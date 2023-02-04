import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, m1, m2, n):
        super().__init__()
        self.regression = nn.Sequential()
        self.regression.add_module('fc1', nn.Linear(m1 + m2, 2048))
        self.regression.add_module('bn1', nn.BatchNorm1d(2048))
        self.regression.add_module('relu1', nn.ReLU(inplace=True))
        self.regression.add_module('fc2', nn.Linear(2048, 1024))
        self.regression.add_module('bn2', nn.BatchNorm1d(1024))
        self.regression.add_module('relu2', nn.ReLU(inplace=True))
        self.regression.add_module('fc3', nn.Linear(1024, 512))
        self.regression.add_module('relu3', nn.ReLU(inplace=True))
        self.regression.add_module('fc4', nn.Linear(512, 128))
        self.regression.add_module('relu4', nn.ReLU(inplace=True))
        self.regression.add_module('fc5', nn.Linear(128, n))

    def forward(self, x1, x2):

        return self.regression(torch.cat((x1, x2), dim=1))
