from typing import Dict, Optional, List, Union

import torch
import torch.nn.functional


def set_temperature(m: torch.nn.Module, temp: torch.Tensor):
    if isinstance(m, MixedModule):
        m.gumbel_temperature.copy_(temp)


def get_named_arch_params(module: torch.nn.Module):
    named_parameters = {n: p for n, p in module.named_parameters() if n.endswith('gumble_arch_params')}
    return named_parameters


def get_named_model_params(module: torch.nn.Module):
    named_parameters = {n: p for n, p in module.named_parameters() if not n.endswith('gumble_arch_params')}
    return named_parameters


class SuperNetwork(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *xs, **kwargs):
        assert not DynamicCostContext.in_context()
        with DynamicCostContext() as context:
            output = self.module(*xs, **kwargs)
        cost = context.get_cost()
        return output, cost

    def set_temperature(self, temp: Union[torch.Tensor, float]):
        if isinstance(temp, float):
            temp = torch.tensor(temp)
        self.apply(lambda x: set_temperature(x, temp))

    def get_named_arch_params(self):
        return get_named_arch_params(self)

    def get_named_model_params(self):
        return get_named_model_params(self)

    def sample_genotype(self):
        genotype_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, MixedModule):
                assert name not in genotype_dict
                genotype_dict[name] = module.sample_genotype_index().item()
        return genotype_dict

    def get_arch_values(self):
        genotype_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, MixedModule):
                assert name not in genotype_dict
                genotype_dict[name] = module.gumble_arch_params.data.cpu().numpy().tolist()
        return genotype_dict

    def thop_estimate_flops_as_cost(self, *inputs):
        import thop
        thop.profile(self, inputs, keep_buffers=True)

        def recursive_sum_total_ops(m):
            sum = 0
            if hasattr(m, 'total_ops'):
                sum += m.total_ops
            
            for children in m.children():
                if not isinstance(children, MixedModule):
                    sum += recursive_sum_total_ops(children)
            return sum

        def set_mixed_module_static_cost_from_flops(module):
            assert isinstance(module, MixedModule)
            for i, op in enumerate(module.ops.values()):
                module.ops_cost_static[i] = recursive_sum_total_ops(op)

        for module in self.modules():
            if isinstance(module, MixedModule):
                set_mixed_module_static_cost_from_flops(module)


class DynamicCostContext:
    contexts: List["DynamicCostContext"] = []

    def __init__(self):
        self.costs: List[torch.Tensor] = []

    def __enter__(self):
        DynamicCostContext.contexts.append(self)
        return self

    def __exit__(self, exc_type, exc_value, trace):
        last = DynamicCostContext.contexts.pop()
        assert last == self

    def add(self, cost):
        self.costs.append(cost)

    def get_cost(self):
        if len(self.costs) == 0:
            return 0
        else:
            return torch.sum(torch.stack(self.costs), dim=0)

    @classmethod
    def in_context(cls):
        return len(cls.contexts)

    @classmethod
    def current_context(cls):
        return cls.contexts[-1]

    @classmethod
    def add_to_current_context(cls, cost):
        assert cls.in_context(), "To use MixedModule please wrap model forward with DynamicCostContext"
        cls.current_context().add(cost)


class MixedModule(torch.nn.Module):
    def __init__(self, ops: Union[List[torch.nn.Module], Dict[str, torch.nn.Module]]):
        super().__init__()
        if isinstance(ops, list):
            ops = {str(i): op for i, op in enumerate(ops)}
        assert len(ops) > 1
        self.ops = torch.nn.ModuleDict(ops)
        self.keys = list(self.ops.keys())
        self.register_buffer('ops_cost_static', torch.zeros(len(self.ops)))
        self.gumble_arch_params = torch.nn.Parameter(torch.ones(len(self.ops), 1))
        self.register_buffer('gumbel_temperature', torch.ones(1))

    def forward(self, x, *xs, weights: Optional[torch.Tensor] = None, gene: Optional[int] = None):
        assert (weights is None) or (gene is None)

        input_size = x.size()
        batch_size = input_size[0]
        if weights is None and gene is None:
            weights = gumbel_softmax_sample(self.gumble_arch_params.expand(-1, batch_size),
                                            temperature=self.gumbel_temperature, dim=0)

        if gene is not None:
            with DynamicCostContext() as context:
                output = self.ops[gene](x, *xs)
            dynamic_cost = context.get_cost()

            index = self.keys.index(gene)
            static_cost = self.ops_cost_static[index].expand(x.shape[0])
            cost = dynamic_cost + static_cost
        else:
            weighted_costs = []
            weighted_outputs = []
            for w, op, static_cost in zip(weights, self.ops.values(), self.ops_cost_static):
                with DynamicCostContext() as context:
                    output = op(x, *xs)
                dynamic_cost = context.get_cost()
                cost = dynamic_cost + static_cost
                weighted_output = w.view(-1, *([1] * (len(input_size) - 1))) * output
                weighted_outputs.append(weighted_output)
                weighted_cost = w * cost
                weighted_costs.append(weighted_cost)

            cost = torch.sum(torch.stack(weighted_costs), dim=0)
            output = torch.sum(torch.stack(weighted_outputs), dim=0)

        DynamicCostContext.add_to_current_context(cost)
        return output

    def sample_genotype_index(self):
        with torch.no_grad():

            sampled_alphas = gumbel_softmax_sample(self.gumble_arch_params.squeeze(),
                                                   temperature=self.gumbel_temperature, dim=0)
            best_sampled_alphas = torch.argmax(sampled_alphas, dim=0)
            return best_sampled_alphas.detach()


def gumbel_softmax_sample(logits, temperature, dim=-1, std=1.0):
    return torch.nn.functional.gumbel_softmax(logits, tau=temperature, dim=dim)
