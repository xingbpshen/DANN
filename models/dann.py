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


class DANNConv2d(nn.Module):

    def __init__(self, m, r, n):
        super(DANNConv2d, self).__init__()
        self.Gf = nn.Sequential()
        self.Gf.add_module('f_conv1', nn.Conv2d(1, 64, (5, 5)))
        self.Gf.add_module('f_bn1', nn.BatchNorm2d(64))
        self.Gf.add_module('f_pool1', nn.MaxPool2d(2))
        self.Gf.add_module('f_relu1', nn.ReLU(True))
        self.Gf.add_module('f_conv2', nn.Conv2d(64, 50, (5, 5)))
        self.Gf.add_module('f_bn2', nn.BatchNorm2d(50))
        self.Gf.add_module('f_drop1', nn.Dropout2d())
        self.Gf.add_module('f_pool2', nn.MaxPool2d(2))
        self.Gf.add_module('f_relu2', nn.ReLU(True))

        self.Gy = nn.Sequential()
        self.Gy.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.Gy.add_module('c_bn1', nn.BatchNorm1d(100))
        self.Gy.add_module('c_relu1', nn.ReLU(True))
        self.Gy.add_module('c_drop1', nn.Dropout2d())
        self.Gy.add_module('c_fc2', nn.Linear(100, 100))
        self.Gy.add_module('c_bn2', nn.BatchNorm1d(100))
        self.Gy.add_module('c_relu2', nn.ReLU(True))
        self.Gy.add_module('c_fc3', nn.Linear(100, 10))
        self.Gy.add_module('c_relu3', nn.ReLU(True))
        self.Gy.add_module('c_fc4', nn.Linear(10, 1))

        self.Gd = nn.Sequential()
        self.Gd.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.Gd.add_module('d_bn1', nn.BatchNorm1d(100))
        self.Gd.add_module('d_relu1', nn.ReLU(True))
        self.Gd.add_module('d_fc2', nn.Linear(100, 2))
        self.Gd.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x, lamb):
        feature = self.Gf(x)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse = GradientReverse.apply(feature, lamb)
        regression = self.Gy(feature)
        domain_classification = self.Gd(reverse)

        return regression, domain_classification


class DANN(nn.Module):

    def __init__(self, m, r, n):
        super(DANN, self).__init__()
        self.fc1_n = int(0.9 * m)
        self.fc2_n = int(0.85 * m)
        self.fc3_n = int(r * m)
        self.Gf = torch.nn.Sequential()
        self.Gf.add_module('gf_fc1', nn.Linear(m, self.fc1_n))
        self.Gf.add_module('gf_tanh1', nn.ReLU(True))
        self.Gf.add_module('gf_fc2', nn.Linear(self.fc1_n, self.fc2_n))
        self.Gf.add_module('gf_tanh2', nn.ReLU(True))
        self.Gf.add_module('gf_fc3', nn.Linear(self.fc2_n, self.fc3_n))
        self.Gf.add_module('gf_tanh3', nn.ReLU(True))

        self.Gy = torch.nn.Sequential()
        self.Gy.add_module('gy_fc1', nn.Linear(self.fc3_n, 64))
        self.Gy.add_module('gy_tanh1', nn.ReLU(True))
        self.Gy.add_module('gy_fc2', nn.Linear(64, 32))
        self.Gy.add_module('gy_tanh2', nn.ReLU(True))
        self.Gy.add_module('gy_fc3', nn.Linear(32, n))

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
