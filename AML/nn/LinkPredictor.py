import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, dataset):
        super(LinkPredictor, self).__init__()

        self.lins, self.bns, self.bns_r = torch.nn.ModuleList(), torch.nn.ModuleList(), torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns.append(BatchNorm1d(hidden_channels))
        self.bns_r.append(BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))
            self.bns_r.append(BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout
        self.dataset = dataset

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j, left=True):
        x = x_i * x_j
        for i in range(len(self.lins) - 1):
            x = self.lins[i](x)
            if left:
                x = F.dropout(F.relu(self.bns[i](x)), p=self.dropout, training=self.training)
            else:
                x = F.dropout(F.relu(self.bns_r[i](x)), p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x.squeeze(1)

