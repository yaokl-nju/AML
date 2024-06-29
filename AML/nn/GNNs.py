import torch
from nn.GNN_Layer import *
import torch.nn.functional as F
import nn
from nn.GraphNorm import GraphNorm
from torch_sparse import matmul
from utils.init_func import switch_init
from nn.LinkPredictor import LinkPredictor
from nn.MLPs import MLPs
from torch.nn import BatchNorm1d
from torch.nn import Parameter

class GNN(torch.nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.args = args
        in_feat_dim = args.node_emb if args.node_emb > 0 and not args.use_mf_embedding else args.num_features
        if args.model == 'SAGE':
            hidden_dim = args.hidden_dim * 2 if args.concat else args.hidden_dim
            in_dims = [in_feat_dim] + [hidden_dim] * (args.layer_num - 1)
            kwargs = [{'concat': args.concat}] * (args.layer_num - 1) + \
                     [{'concat': args.concat}]
            modelname = 'SAGEConv_dgl'
        elif args.model == 'GAT':
            hidden_dim = args.hidden_dim * args.n_heads if args.concat else args.hidden_dim
            in_dims = [in_feat_dim] + [hidden_dim] * (args.layer_num - 1)
            kwargs = [{'n_heads': args.n_heads, 'concat': args.concat}] * (args.layer_num - 1) + \
                     [{'n_heads': args.n_heads, 'concat': args.concat}]
            modelname = 'GATConv_dgl'

        self.drop = torch.nn.ModuleList([])
        for i in range(args.layer_num):
            self.drop.append(torch.nn.Dropout(args.drop) if args.drop != 0.0 else torch.nn.Identity())

        GNNConv = getattr(nn, modelname)
        self.convs_l, self.bns_l, self.bns_pe = torch.nn.ModuleList([]), torch.nn.ModuleList([]), torch.nn.ModuleList([])
        self.convs_r, self.bns_r = torch.nn.ModuleList([]), torch.nn.ModuleList([])
        for i in range(args.layer_num):
            self.convs_l.append(GNNConv(in_dims[i], args.hidden_dim, **kwargs[i]))
            self.bns_l.append(GraphNorm(hidden_dim, momentum=0.9) if args.bn else torch.nn.Identity())
            self.bns_pe.append(GraphNorm(hidden_dim, momentum=0.9) if args.bn else torch.nn.Identity())
            self.convs_r.append(GNNConv(in_dims[i] , args.hidden_dim, **kwargs[i]))
            self.bns_r.append(GraphNorm(hidden_dim, momentum=0.9) if args.bn else torch.nn.Identity())
        self.predictor = LinkPredictor(in_dims[-1], args.hidden_dim, 1, args.layer_num, args.drop, args.dataset)
        if args.node_emb > 0 and not args.use_mf_embedding:
            self.bns_in_l = GraphNorm(in_dims[0], momentum=0.9) if args.bn else torch.nn.Identity()
        self.drop_in = torch.nn.Dropout(args.drop) if args.drop != 0.0 else torch.nn.Identity()

        if args.node_emb > 0 and not args.use_mf_embedding and not args.use_feature:
            self.node_emb_l = torch.nn.Embedding(args.num_nodes, args.node_emb)
            self.node_emb_r = torch.nn.Embedding(args.num_nodes, args.node_emb)
        self.reset_parameters()

    def reset_parameters(self):
        if self.args.node_emb > 0 and not self.args.use_mf_embedding:
            switch_init(self.node_emb_l.weight, dim=-1)
            switch_init(self.node_emb_r.weight, dim=-1)

    def forward_gnn(self, feat, graph, ids_map, axis='l'):     # axis='l' for head, axis='r' for tail
        if self.args.node_emb > 0 and not self.args.use_mf_embedding:
            bn_in, node_emb = self.__getattr__(f'bns_in_{axis}'), self.__getattr__(f'node_emb_{axis}')
            feat = self.drop_in(F.relu(bn_in(node_emb.weight[ids_map]), inplace=True))
        x, x_last = feat, None
        convs, bns = self.__getattr__(f'convs_{axis}'), self.__getattr__(f'bns_{axis}')
        for i in range(self.args.layer_num):
            x = convs[i](x, graph[i]) if isinstance(graph, list) else convs[i](x, graph)
            if i < self.args.layer_num - 1:
                x = self.drop[i](F.relu(bns[i](x), inplace=True))
            else:
                x = self.drop[i](bns[i](x))
            if x_last is not None:
                x = x + x_last[:x.size(0)]
            if self.args.dataset not in ['ogbl-ddi']:
                x = F.normalize(x, p=2., dim=-1)
            x_last = x
        return x

    def forward_mlp(self, feat, feat_v, feat_d):
        x_pe = feat_v
        x_res = feat - feat_v if feat_d is None else feat_d
        x_last_pe, x_last_res = None, None
        for i in range(self.args.layer_num):
            x_pe = self.convs_l[i](x_pe)
            x_res = self.convs_r[i](x_res)
            if i < self.args.layer_num - 1:
                x_pe = self.drop[i](F.relu(self.bns_pe[i](x_pe), inplace=True))
                x_res = self.drop[i](F.relu(self.bns_r[i](x_res), inplace=True))
            else:
                x_pe = self.drop[i](self.bns_pe[i](x_pe))
                x_res = self.drop[i](self.bns_r[i](x_res))
            if x_last_res is not None:
                x_pe = x_pe + x_last_pe[:x_pe.size(0)]
                x_res = x_res + x_last_res[:x_res.size(0)]
            if self.args.dataset not in ['ogbl-ddi'] and i < self.args.layer_num - 1:
                x_pe = F.normalize(x_pe, p=2., dim=-1)
                x_res = F.normalize(x_res, p=2., dim=-1)
            x_last_pe = x_pe
            x_last_res = x_res
        x_pe = x_pe + x_res
        if self.args.dataset not in ['ogbl-ddi']:
            x_pe = F.normalize(x_pe, p=2., dim=-1)
        return x_pe

    def forward(self, feat, graph, links, ids_map, feat_v, feat_d):
        if self.args.node_emb > 0 and not self.args.use_mf_embedding and not self.args.use_feature:
            # typically strategy is symmetric under this condition
            feat = self.bns_in_l(self.node_emb_l.weight)
        x_l = self.forward_gnn(feat, graph, ids_map, axis='l')
        if self.args.strategy == 'symmetric':
            # x_r = self.forward_gnn(feat, graph, ids_map, axis='r') if self.args.directed else x_l
            x_r = x_l
        else:
            x_r = self.forward_mlp(feat, feat_v, feat_d)
            x_l = x_l + x_r[:x_l.size(0)]

        out = self.predictor(x_l[links[0]], x_r[links[1]])
        return out

    @torch.no_grad()
    def forward_predict(self, x_l, x_r):
        out = self.predictor(x_l, x_r)
        return out

    def get_node_embedding(self):
        self.eval()
        return self.bns_in_l(self.node_emb_l.weight)