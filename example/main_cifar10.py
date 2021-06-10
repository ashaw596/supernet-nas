'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import torchvision.models as models

import simple_model
from arch_search import SuperNetwork
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--arch_lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
#net = models.resnet18(num_classes=10)
# net = simple_model.SimpleNet()
net = SuperNetwork(simple_model.SimpleSearch()) # NAS Specific
net.thop_estimate_flops_as_cost(torch.rand(1, 3, 32, 32)) # NAS Specific
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
model_optimizer = optim.SGD(net.get_named_model_params().values(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
                      
arch_optimizer = optim.SGD(net.get_named_arch_params().values(), lr=args.arch_lr, momentum=0) # NAS Specific

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_cost = 0
    train_criterion_loss = 0
    correct = 0
    total = 0
    cost_multiplier = 1e-6 # NAS Specific
    temperature = max(5.0 - (5.0 - 1.0) * (epoch / 150) , 1.0) # NAS Specific. temperature goes down from 5 to 1 linearly
    net.set_temperature(temperature) # NAS Specific
    print(f"Gumbel Softmax Temperature: {temperature}") # NAS Specific
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        model_optimizer.zero_grad() # NAS Specific
        arch_optimizer.zero_grad() # NAS Specific
        outputs, cost = net(inputs) # NAS Specific
        criterion_loss = criterion(outputs, targets) # NAS Specific
        cost_loss = torch.mean(cost) * cost_multiplier # NAS Specific
        loss = cost_loss + criterion_loss # NAS Specific
        loss.backward()
        model_optimizer.step() # NAS Specific
        arch_optimizer.step() # NAS Specific

        train_criterion_loss += criterion_loss.item()
        train_cost += torch.mean(cost_loss).item()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 
            f'Loss: {train_loss/(batch_idx+1):.3f} CE_Loss: {train_criterion_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total}) | Cost Loss: {train_cost/(batch_idx+1)}')


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    train_cost = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, cost = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            train_cost += torch.mean(cost).item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 
            f'Loss: {test_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total}) | Cost: {train_cost/(batch_idx+1)}')

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()