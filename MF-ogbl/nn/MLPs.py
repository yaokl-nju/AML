import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d

import nn
from nn.MLP_Layer import *
from nn.LinkPredictor import LinkPredictor

class MLPs(torch.nn.Module):
    def __init__(self, args):
        super(MLPs, self).__init__()
        self.args = args
        feat_dim = args.node_emb if args.node_emb > 0 else args.num_features
        in_dims = [feat_dim] + [args.hidden_dim] * (args.layer_num - 1)

        self.convs_l, self.bns_l, self.node_emb_l, self.bns_in_l = self.get_module(in_dims, args)
        self.predictor = LinkPredictor(in_dims[-1], args.hidden_dim, 1, args.layer_num, args.drop, args.dataset)
        self.drop_func = torch.nn.Dropout(args.drop) if args.drop != 0.0 else torch.nn.Identity()

        if args.directed:
            self.convs_r, self.bns_r, self.node_emb_r, self.bns_in_r = self.get_module(in_dims, args)
        else:
            self.convs_r, self.bns_r, self.node_emb_r, self.bns_in_r = \
                self.convs_l, self.bns_l, self.node_emb_l, self.bns_in_l

    def get_module(self, in_dims, args):
        convs, bns = torch.nn.ModuleList([]), torch.nn.ModuleList([])
        for i in range(args.layer_num):
            convs.append(MLP_Layer(in_dims[i], args.hidden_dim))
            bns.append(BatchNorm1d(args.hidden_dim) if args.bn else torch.nn.Identity())
        if args.node_emb > 0:
            node_emb = torch.nn.Embedding(args.num_nodes, args.node_emb)
            bns_in = BatchNorm1d(args.node_emb) if args.bn else torch.nn.Identity()
        else:
            node_emb, bns_in = None, None
        return convs, bns, node_emb, bns_in

    def forward_single(self, feat, ids_map, axis='l'):
        if self.args.node_emb > 0:
            if ids_map is not None:
                feat = self.drop_func(self.__getattr__(f'bns_in_{axis}')(self.__getattr__(f'node_emb_{axis}').weight[ids_map]))
            else:
                feat = self.drop_func(self.__getattr__(f'bns_in_{axis}')(self.__getattr__(f'node_emb_{axis}').weight))
        x, x_last = feat, None
        for i in range(self.args.layer_num):
            x = self.__getattr__(f'convs_{axis}')[i](x)
            if i < self.args.layer_num - 1:
                x = self.drop_func(F.relu(self.__getattr__(f'bns_{axis}')[i](x), inplace=True))
            else:
                x = self.drop_func(self.__getattr__(f'bns_{axis}')[i](x))
            x = x + x_last if x_last is not None else x
            x = F.normalize(x, p=2., dim=-1) if self.args.dataset != 'ogbl-ddi' else x
            x_last = x
        return feat, x

    def forward(self, feat, links, ids_map):
        x_l = self.forward_single(feat, ids_map, 'l')[1]
        # x_r = self.forward_single(feat, ids_map, 'r')[1] if self.args.directed else x_l
        x_r = x_l
        out = self.predictor(x_l[links[0]], x_r[links[1]])
        return out

    @torch.no_grad()
    def get_node_embedding(self):
        self.eval()
        feat_l, _ = self.forward_single(None, None, 'l')
        # if self.args.directed:
        #     feat_r, _ = self.forward_single(None, None, 'r')
        # else:
        #     feat_r = feat_l
        return feat_l