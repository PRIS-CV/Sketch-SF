import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter, scatter_softmax
import torch_geometric as tg


class GA_Module(nn.Module):
    def __init__(self, channels, qk_channels, v_channels):
        super(GA_Module, self).__init__()
        self.temperature = qk_channels ** 0.5
        self.q_conv = nn.Conv1d(channels, qk_channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, qk_channels, 1, bias=False)
        self.q_conv.weight.data = self.k_conv.weight.data.clone()

        self.v_conv = nn.Conv1d(channels, v_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        # b, n, c
        x_v = self.v_conv(x).permute(0, 2, 1)
        # b, n, n
        energy = torch.bmm(x_q, x_k) / self.temperature

        attention = self.softmax(energy)
        # b, c, n
        x_r = torch.bmm(attention, x_v).permute(0, 2, 1)
        return x_r


class SA_Module(MessagePassing):
    def __init__(self, channels, qk_channels, v_channels, aggr='add', **kwargs):
        super(SA_Module, self).__init__(aggr=aggr, **kwargs)
        # self.nn = nn
        self.temperature = qk_channels ** 0.5
        self.channels = channels
        self.q_matrix = nn.Linear(channels, qk_channels, bias=False)
        self.k_matrix = nn.Linear(channels, qk_channels, bias=False)
        self.q_matrix.weight.data = self.k_matrix.weight.data.clone()
        # self.q_matrix.weight = self.k_matrix.weight
        # self.q_matrix.bias = self.k_matrix.bias
        self.v_matrix = nn.Linear(channels, v_channels, bias=False)
        self.aggr = aggr
        self.edge_index = None

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        self.edge_index = edge_index
        out = self.propagate(edge_index, x=x)
        self.edge_index = None
        # x = x + out

        return out

    def message(self, x_i, x_j):
        x_q = self.q_matrix(x_i)
        x_k = self.k_matrix(x_j)
        x_v = self.v_matrix(x_j)
        energy = x_q * x_k
        energy = torch.sum(energy, dim=1, keepdim=True)
        energy = energy / self.temperature
        attention = tg.utils.softmax(energy, self.edge_index[1])

        self.edge_index = None

        return x_v*attention


