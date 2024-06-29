import gc
import joblib
import numpy as np
from ogb.linkproppred import PygLinkPropPredDataset
import os.path as osp
import time, torch
from torch_geometric.utils import add_self_loops, negative_sampling, to_undirected
from torch_scatter import scatter_add
from torch_sparse import SparseTensor

from global_val import *
import samplers
from utils.init_func import row_normalize

class ogbl_dataset(PygLinkPropPredDataset):
    def __init__(self, args, name, root='dataset'):
        super(ogbl_dataset, self).__init__(name, root)
        self.args = args
        self.num_nodes = self[0].num_nodes
        self.edge_dict = self._get_edge_split()
        self.load()
        self.num_feature = 0 if self.data.x is None else self.data.x.size(1)

        self.batch = {}
        for phase in self.edge_dict.keys():
            if args.bsize > 0:
                esize = self.edge_dict[phase]['edge_pos'].size(1) * 2 if phase == 'train' else self.edge_dict[phase]['edge'].size(1)
                self.batch[phase] = esize // args.bsize if phase == 'train' else esize // 256000
                if self.batch[phase] < 1:
                    self.batch[phase] = 1
            else:
                self.batch[phase] = 1
        self.sampler = getattr(samplers, args.method)

        print("data info:")
        print("\tnum_train", self.edge_dict['train']['edge_pos'].size())
        print("\tnum_valid", self.edge_dict['valid']['edge'].size())
        print("\tnum_test", self.edge_dict['test']['edge'].size())
        print("\tnum_nodes", self.num_nodes)
        print("\tnum_features", self.num_feature)
        print("\tnum_edges", self.data.edge_index.size())


    def row_normalize(self, src, row_index, num_nodes):
        out = src / (scatter_add(src, row_index, dim=0, dim_size=num_nodes)[row_index] + 1e-16)
        return out

    def coalesc(self, edges):
        perm = torch.argsort(edges[0] * self.num_nodes + edges[1])
        edges = edges[:, perm]
        eids = edges[0] * self.num_nodes + edges[1]
        mask = eids[1:] > eids[:-1]
        edges = torch.cat([edges[:, (0,)], edges[:, 1:][:, mask]], dim=1)
        return edges

    def _get_edge_split(self):
        edge_split = self.get_edge_split()
        edge_dict = {}
        for stage in ['train', 'valid', 'test']:
            edge_dict[stage] = {}
            if 'edge' in edge_split['train']:
                pos_edge = edge_split[stage]['edge'].t()
                assert pos_edge.size(0) == 2
                label = torch.ones(pos_edge.size(1))

                if stage != 'train':
                    edge_dict[stage]['edge_pos'] = pos_edge
                    edge_dict[stage]['label_pos'] = label

                    neg_edge = edge_split[stage]['edge_neg'].t()
                    assert neg_edge.size(0) == 2
                    label = torch.cat([label, torch.zeros(neg_edge.size(1))])
                    edge_dict[stage]['edge'] = torch.cat([pos_edge, neg_edge], dim=1)
                    edge_dict[stage]['label'] = label
                else:
                    edge_dict[stage]['edge_pos'] = pos_edge
                    edge_dict[stage]['label_pos'] = label

            elif 'source_node' in edge_split['train']:
                source = edge_split[stage]['source_node']
                target = edge_split[stage]['target_node']
                pos_edge = torch.stack([source, target], dim=0)
                label = torch.ones(source.size(0))
                if stage != 'train':
                    edge_dict[stage]['edge_pos'] = pos_edge
                    edge_dict[stage]['label_pos'] = label

                    target_neg = edge_split[stage]['target_node_neg']
                    neg_edge = torch.stack([source.repeat_interleave(target_neg.size(1)), target_neg.view(-1)])
                    label = torch.cat([label, torch.zeros(neg_edge.size(1))])
                    edge_dict[stage]['edge'] = torch.cat([pos_edge, neg_edge], dim=1)
                    edge_dict[stage]['label'] = label
                else:
                    edge_dict[stage]['edge_pos'] = pos_edge
                    edge_dict[stage]['label_pos'] = label
            else:
                assert False
        return edge_dict

    def process_for_dataset(self):
        if self.data.x is not None:
            if self.args.use_feature and self.args.pre_encode:
                edge_index = self.data.edge_index
                edge_weight = row_normalize(torch.ones(edge_index.size(1), 1), edge_index, num_nodes=self.num_nodes).view(-1)
                graph_v = SparseTensor(
                    row=edge_index[0], col=edge_index[1], value=edge_weight,
                    sparse_sizes=(self.num_nodes, self.num_nodes)
                ).to(self.args.device)
                x = self.data.x.to(torch.float).to(self.args.device)
                for _ in range(self.args.layer_num):
                    x = graph_v @ x
                self.x_v = x
                del graph_v
                gc.collect()
                torch.cuda.empty_cache()

                if self.args.use_valedges_as_input:
                    val_edge = self.edge_dict['valid']['edge_pos']
                    edge_index = torch.cat([self.data.edge_index, val_edge], dim=1)
                    edge_index = to_undirected(edge_index, self.num_nodes) if not self.args.directed else edge_index
                    edge_weight = row_normalize(torch.ones(edge_index.size(1), 1), edge_index, num_nodes=self.num_nodes).view(-1)
                    graph_t = SparseTensor(
                        row=edge_index[0], col=edge_index[1], value=edge_weight,
                        sparse_sizes=(self.num_nodes, self.num_nodes)
                    ).to(self.args.device)
                    x = self.data.x.to(torch.float).to(self.args.device)
                    for _ in range(self.args.layer_num):
                        x = graph_t @ x
                    self.x_t = x
                    del graph_t
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    self.x_t = self.x_v
            else:
                self.x_v, self.x_t = self.data.x, self.data.x

    def load_mf_embedding(self):
        if self.args.use_feature and self.args.use_mf_embedding:
            path = osp.join(self.root, 'processed/MF_emb.pt')
            self.data.x = torch.FloatTensor(joblib.load(path))

    def load_node2vec(self):
        if self.args.use_feature and self.args.use_node2vec:
            path = osp.join(self.root, 'processed/node2vec.pt')
            self.data.x = torch.FloatTensor(joblib.load(path))
        if self.args.use_node2vec and self.args.use_mf_embedding:
            raise NameError('Use node2vec conflicts with use mf embedding!')

    def handle_dataleakage(self):
        edge_index = self.data.edge_index
        eids = edge_index[0] * self.num_nodes + edge_index[1]
        edges_v = self.edge_dict['valid']['edge_pos']
        edges_v = to_undirected(edges_v, self.num_nodes) if not self.args.directed else edges_v
        eids_v = edges_v[0] * self.num_nodes + edges_v[1]
        edges_t = self.edge_dict['test']['edge_pos']
        edges_t = to_undirected(edges_t, self.num_nodes) if not self.args.directed else edges_t
        eids_t = edges_t[0] * self.num_nodes + edges_t[1]
        mask1 = np.isin(eids.numpy(), eids_v.numpy())
        mask2 = np.isin(eids.numpy(), eids_t.numpy())
        mask = np.logical_not(np.logical_or(mask1, mask2))
        self.data.edge_index = edge_index[:, torch.BoolTensor(mask)]

    def load(self):
        self.handle_dataleakage()
        if self.args.use_graph_edges:
            edges = self.data.edge_index
            mask = edges[0] <= edges[1]
            edges = edges[:, mask]
            self.edge_dict['train']['edge_pos'] = edges
            self.edge_dict['train']['label_pos'] = torch.ones(edges.size(1))

        self.load_mf_embedding()
        self.load_node2vec()
        self.process_for_dataset()
        if not self.args.use_feature:
            self.data.x = None
    # ===============================================

    # ===============================================
    # funcs of sampling data
    def reset_iter(self):
        self.perm_idx = {}
        for phase in self.edge_dict.keys():
            self.perm_idx[phase] = self.iter_idx(phase)
        self.time = [0., 0., 0.]

    def iter_idx(self, phase):
        if phase == 'train':
            num_pos = self.edge_dict[phase]['edge_pos'].size(1)
            # bsize = num_pos // self.batch[phase]
            bsize = self.args.bsize // 2
            perm = torch.randperm(num_pos).split(bsize)
            perm = perm if num_pos % bsize == 0 else perm[:-1]

            if self.args.eval_metric != 'mrr':
                nlinks, num_neg = None, 0
                while num_neg < num_pos:
                    nlinks_i = negative_sampling(self.data.edge_index, num_nodes=self.num_nodes, num_neg_samples=num_pos - num_neg)
                    nlinks = torch.cat([nlinks, nlinks_i], dim=1) if nlinks is not None else nlinks_i
                    num_neg = nlinks.size(1)
            else:
                neg_dst = torch.randint(0, self.num_nodes, (num_pos,), dtype=torch.long)
                nlinks = torch.stack([self.edge_dict[phase]['edge_pos'][0], neg_dst], dim=0)
            self.edge_dict['train']['edge_neg'] = nlinks
            self.edge_dict['train']['label_neg'] = torch.zeros(nlinks.size(1))
        else:
            bsize = int(np.ceil(self.edge_dict[phase]['edge'].size(1) * 1.0 / self.batch[phase]))
            perm = torch.arange(self.edge_dict[phase]['edge'].size(1)).split(bsize)
        for ids in perm:
            yield ids
        yield None

    def get_links(self, phase):
        index = next(self.perm_idx[phase])
        if index is None:
            self.perm_idx[phase] = self.iter_idx(phase)
            index = next(self.perm_idx[phase])
        if phase != 'train':
            links, labels = self.edge_dict[phase]['edge'][:, index], self.edge_dict[phase]['label'][index]
        else:
            s_time = time.time()
            plinks, plabels = self.edge_dict[phase]['edge_pos'][:, index], self.edge_dict[phase]['label_pos'][index]
            nlinks, nlabels = self.edge_dict[phase]['edge_neg'][:, index], self.edge_dict[phase]['label_neg'][index]
            self.time[1] = self.time[1] + (time.time() - s_time)

            links = torch.cat([plinks, nlinks], dim=1)
            labels = torch.cat([plabels, -torch.ones(nlinks.size(1))])
        return links, labels

    def sample(self, phase):
        s_time = time.time()
        links, labels = self.get_links(phase)
        nlinks, ids_map, links, labels = self.sampler(links, labels, self)
        labels = labels.to(self.args.device)
        if phase == 'train':
            self.time[2] = self.time[2] + (time.time() - s_time)

        x = self.x_v if phase != 'test' else self.x_t
        feat = x[ids_map].to(self.args.device) if x is not None else None
        return labels, nlinks.to(self.args.device), ids_map.to(self.args.device), feat