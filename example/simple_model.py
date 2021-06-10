import torch
import torch.nn.functional as F
from torch import nn
from arch_search import MixedModule

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc = nn.Sequential(nn.Linear(16 * 5 * 5, 120) , nn.ReLU(), nn.Linear(120, 84), nn.ReLU())
        self.last_layer = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        x = self.last_layer(x)
        return x

class SimpleSearch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc = MixedModule({
            'option_1': nn.Sequential(nn.Linear(16 * 5 * 5, 120) , nn.ReLU(), nn.Linear(120, 84), nn.ReLU()),
            'option_2': nn.Sequential(nn.Linear(16 * 5 * 5, 240) , nn.ReLU(), nn.Linear(240, 84), nn.ReLU()),
        })
        self.last_layer = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        x = self.last_layer(x)
        return x