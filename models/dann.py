import torch
import torch.nn as nn


class GradientReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb

        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        new_grad = gradient.neg() * ctx.lamb

        return new_grad, None


class DANN(nn.Module):

    def __init__(self, m, r, n):
        super().__init__()
        self.fc1_n = int(0.9 * m)
        self.fc2_n = int(0.85 * m)
        self.fc3_n = int(r * m)
        self.Gf = torch.nn.Sequential()
        self.Gf.add_module('gf_fc1', nn.Linear(m, self.fc1_n))
        self.Gf.add_module('gf_relu1', nn.ReLU(True))
        self.Gf.add_module('gf_fc2', nn.Linear(self.fc1_n, self.fc2_n))
        self.Gf.add_module('gf_relu2', nn.ReLU(True))
        self.Gf.add_module('gf_fc3', nn.Linear(self.fc2_n, self.fc3_n))
        self.Gf.add_module('gf_relu3', nn.ReLU(True))

        self.Gy = torch.nn.Sequential()
        self.Gy.add_module('gy_fc1', nn.Linear(self.fc3_n, 100))
        self.Gy.add_module('gy_relu1', nn.ReLU(True))
        self.Gy.add_module('gy_fc2', nn.Linear(100, 64))
        self.Gy.add_module('gy_relu2', nn.ReLU(True))
        self.Gy.add_module('gy_fc3', nn.Linear(64, n))

        self.Gd = torch.nn.Sequential()
        self.Gd.add_module('gd_fc1', nn.Linear(self.fc3_n, 100))
        self.Gd.add_module('gd_relu1', nn.ReLU(True))
        self.Gd.add_module('gd_fc2', nn.Linear(100, 2))
        self.Gd.add_module('gd_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x, lamb):
        feature = self.Gf(x)
        reverse = GradientReverse.apply(feature, lamb)
        regression = self.Gy(feature)
        domain_classification = self.Gd(reverse)

        return regression, domain_classification
