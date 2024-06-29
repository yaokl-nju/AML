import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from utils.init_func import switch_init

class MLPs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, dataset):
        super(MLPs, self).__init__()
        self.dataset = dataset
        self.lins, self.bns = torch.nn.ModuleList(), torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns.append(BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))
        self.drop = torch.nn.Dropout(dropout) if dropout != 0.0 else torch.nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            switch_init(lin.weight, dim=-1)
            if lin.bias is not None:
                switch_init(lin.bias)
            # lin.reset_parameters()

    def forward(self, feat):
        x, x_last = feat, None
        for i in range(len(self.lins)):
            x = self.lins[i](x)
            x = self.drop(F.relu(self.bns[i](x), inplace=True))
            if x_last is not None:
                x = x + x_last
            if self.dataset != 'ogbl-ddi':
                x = F.normalize(x, p=2., dim=-1)
            x_last = x
        return x