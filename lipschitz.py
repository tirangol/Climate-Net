"""lipmlp (https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/lipmlp.py)"""
import torch
import math


class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        # compute lipschitz constant of initial weight to initialize self.c
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max()  # just a rough initialization

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)
