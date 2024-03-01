import torch
from torch import nn, Tensor

import numpy as np

class FFNN(nn.Module):
    def __init__(self, input_nodes: Tensor, hidden_nodes: Tensor, output_nodes: Tensor, act_func: nn.Module, N: int) -> None:
        super(FFNN, self).__init__()

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.act_func = act_func
        self.N = N

        self.fc1 = nn.Linear(input_nodes, hidden_nodes, bias=True)
        self.fc2 = nn.Linear(hidden_nodes, output_nodes, bias=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.fc1.weight.copy_(torch.rand(self.hidden_nodes, self.input_nodes) - 1)
        self.fc1.bias.copy_(2*torch.rand(self.hidden_nodes) - 1)
        self.fc2.weight.copy_(torch.rand(self.output_nodes, self.hidden_nodes))

    def forward(self, k):
        x=self.fc1(k)
        x=self.act_func(x)
        x=self.fc2(x)
        s_state, d_state = x[...,0], x[...,1]
        return s_state.unsqueeze(-1), d_state.unsqueeze(-1)