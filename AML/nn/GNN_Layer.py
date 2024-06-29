import torch
from utils.init_func import softmax
from torch_sparse import SparseTensor
import math
from torch_geometric.nn.inits import glorot, zeros

class SAGEConv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 concat=False,
                 normalize=True,
                 ):
        super(SAGEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.concat = concat
        self.normalize = normalize

        self.lin_l = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_r = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        if self.lin_l.bias is not None:
            zeros(self.lin_l.bias)
        if self.lin_r.bias is not None:
            zeros(self.lin_r.bias)

    def forward(self, x, graph=None):
        if graph is not None:
            out_l = self.lin_l(graph @ x)
            out_r = self.lin_r(x[:out_l.size(0)])
            output = torch.cat([out_l, out_r], dim=1) if self.concat else (out_l + out_r)
        else:
            out_l = self.lin_l(x)
            out_r = self.lin_r(x)
            output = torch.cat([out_l, out_r], dim=1) if self.concat else (out_l + out_r)
        return output

class GATConv(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_heads=1,
                 bias = True,
                 concat=True,
                 ):
        super(GATConv, self).__init__()
        self.n_heads = n_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.concat=concat

        self.a1 = torch.nn.Linear(in_dim, n_heads, bias=True)
        self.a2 = torch.nn.Linear(in_dim, n_heads, bias=True)
        self.lin_l = torch.nn.Linear(in_dim, out_dim * n_heads, bias=bias)
        self.lin_r = torch.nn.Linear(in_dim, out_dim * n_heads if concat else out_dim, bias=bias)
        self.leakyrelu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        if self.lin_l.bias is not None:
            zeros(self.lin_l.bias)
        if self.lin_r.bias is not None:
            zeros(self.lin_r.bias)
        glorot(self.a1.weight)
        glorot(self.a2.weight)
        if self.a1.bias is not None:
            zeros(self.a1.bias)
        if self.a2.bias is not None:
            zeros(self.a2.bias)

    def forward(self, x, graph=None):
        if graph is not None:
            is_ST = True if isinstance(graph, SparseTensor) else False
            if is_ST:
                row, col, norm = graph.storage.row(), graph.storage.col(), graph.storage.value()
            else:
                row, col = graph._indices()
                norm = graph._values()

            num_nodes = graph.size(0)
            attn1 = self.a1(x).view(-1, self.n_heads)
            attn2 = self.a2(x).view(-1, self.n_heads)
            attn = attn1[row] + attn2[col]
            attn_d = softmax(attn, row, num_nodes=num_nodes)

            xw = self.lin_l(x).view(-1, self.n_heads, self.out_dim)
            out_l = []
            for i in range(self.n_heads):
                attn_adj = SparseTensor(row=row, col=col, value=attn_d[:, i], sparse_sizes=(num_nodes, x.size(0)), is_sorted=is_ST)
                out_l.append((attn_adj @ xw[:, i, :]).unsqueeze(1))
            out_l = torch.cat(out_l, dim=1)
            out_l = out_l.flatten(1, -1) if self.concat else out_l.sum(1)
            out = out_l + self.lin_r(x[:out_l.size(0)])
            return out
        else:
            out = self.lin_l(x) + self.lin_r(x)
            return out

