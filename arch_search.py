from typing import Optional, List, Union, Any

import torch
import torch.nn.functional as F
from torch.distributions import Gumbel

gumbel_dist = Gumbel(0.0, 1.0)

def sample_gumbel(logits=None, shape=None, eps=1e-20, std=1.0):
    if input:
        U = torch.rand_like(logits)
    else:
        U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps) * std


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


class MixedModule(torch.nn.Module):
    def __init__(self, ops: List[torch.nn.Module]):
        super().__init__()
        self.ops = torch.nn.ModuleList(ops)
        self.register_buffer('ops_cost', torch.zeros(len(self.ops)))
        self.gumble_arch_params = torch.nn.Parameter(torch.ones(len(self.ops), 1))
        self.register_buffer('gumbel_temperature', torch.ones(1))

    def forward(self, x, *xs, weights: Optional[torch.Tensor] = None, gene: Optional[int] = None):
        # weights.size = [len(self.ops), batch_size]
        assert (weights is None) or (gene is None)

        input_size = x.size()
        batch_size = input_size[0]
        if weights is None and gene is None:
            weights = gumbel_softmax_sample(self.gumble_arch_params.expand(-1, batch_size),
                                            temperature=self.gumbel_temperature, dim=0)


        # if torch.sum(torch.isnan(weights)) > 0:
        #     print(weights)
        #     print(self.gumble_arch_params)
        #     raise Exception("Nan")
        # weights.size = [len(self.ops), batch_size]

        # if len(self.ops) == 1:
        #     gene = 0
        if gene is not None:
            output = self.ops[gene](x, *xs)
            flops = self.ops_cost[gene].expand(x.shape[0])
        else:
            output = sum(w.view(-1, *([1] * (len(input_size) - 1))) * op(x, *xs) for w, op in zip(weights, self.ops))
            flops = torch.sum(weights * self.ops_cost.view(-1, 1), dim=0)

        # if torch.sum(torch.isnan(output)) > 0:
        #     print(output)
        #     print(flops)
        #     raise Exception("Nan")
        return output, flops

    def use_flops(self, delete=False):
        for i, op in enumerate(self.ops):
            total_flop = 0
            for m in op.modules():
                if hasattr(m, 'total_ops'):
                    total_flop += m.total_ops.item()
                    if delete:
                        del m.total_ops

                if hasattr(m, 'total_params'):
                    if delete:
                        del m.total_params
            self.ops_cost[i].copy_(torch.tensor(total_flop / 1e6))
        print(self.ops_cost)

    def sample_genotype_index(self):
        with torch.no_grad():
            sampled_alphas = gumbel_softmax_sample(self.gumble_arch_params.squeeze(),
                                                   temperature=self.gumbel_temperature, dim=0)
            best_sampled_alphas = torch.argmax(sampled_alphas, dim=0)
            return best_sampled_alphas.detach()


def gumbel_softmax_sample(logits, temperature, dim=None, std=1.0):
    y = logits + gumbel_dist.sample(logits.shape).to(device=logits.device, dtype=logits.dtype)
    # y = logits + sample_gumbel(logits=logits, std=std)
    return F.softmax(y / temperature, dim=dim)
