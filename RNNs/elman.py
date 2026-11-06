import torch
import torch.nn as nn

class ElmanCell(nn.Module):
    """h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)"""

    def __init__(self, input_dim, units):
        super().__init__()
        self.input_dim = input_dim
        self.units = units
        self.w_xh = nn.Linear(input_dim, units)
        self.w_hh = nn.Linear(units, units, bias=False)
        nn.init.xavier_uniform_(self.w_xh.weight)
        nn.init.orthogonal_(self.w_hh.weight)

    def forward(self, x_t, h_prev):
        return torch.tanh(self.w_xh(x_t) + self.w_hh(h_prev))

    def count_parameters(self):
        '''Count total learnable parameters.'''
        return self.units * (self.input_dim + 1) + self.units * self.units
