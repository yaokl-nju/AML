import torch
from torch_geometric.nn.inits import glorot, zeros

class MLP_Layer(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 ):
        super(MLP_Layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_l = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        if self.lin_l.bias is not None:
            zeros(self.lin_l.bias)

    def forward(self, x):
        out = self.lin_l(x)
        return out