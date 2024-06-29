import gc
import joblib
import numpy as np
from ogb.linkproppred import PygLinkPropPredDataset
import os.path as osp
import random
import time, torch
from torch_geometric.utils import add_self_loops, negative_sampling, to_undirected
from torch_scatter import scatter_add
from torch_sparse import SparseTensor

from global_val import *
import samplers
from utils.init_func import row_normalize, neigh_index, choice_multi_range


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
            esize = self.edge_dict[phase]['edge_pos'].size(1) * 2 if phase == 'train' else self.edge_dict[phase]['edge'].size(1)
            self.batch[phase] = esize // args.bsize if phase == 'train' else esize // 256000
            if self.batch[phase] < 1:
                self.batch[phase] = 1
        self.batch['node'] = 1 if args.method_te == 'FULL' else self.num_nodes // (args.bsize * 2)
        if self.batch['node'] < 1:
            self.batch['node'] = 1

        self.samplers = {}
        self.samplers['train'] = getattr(samplers, args.method+"_edge")
        self.samplers['node'] = getattr(samplers, args.method_te+"_node")
        self.methods = {'train':args.method, 'node':args.method_te}

        print("data info:")
        # print("\tnum_train", self.edge_dict['train']['edge'].size())
        print("\tnum_train", self.edge_dict['train']['edge_pos'].size())
        print("\tnum_valid", self.edge_dict['valid']['edge'].size())
        print("\tnum_test", self.edge_dict['test']['edge'].size())
        print("\tnum_nodes", self.num_nodes)
        print("\tnum_features", self.num_feature)
        print("\tnum_edges", self.data.edge_index.size())
        print("\tbatch", self.batch)

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
                    edge_dict[stage]['edge'] = torch.cat([pos_edge, neg_edge], dim=1)
                    edge_dict[stage]['label'] = torch.cat([label, torch.zeros(neg_edge.size(1))])
                else:
                    if self.args.strategy == 'asymmetric':
                        pos_edge = to_undirected(pos_edge, num_nodes=self.num_nodes) if not self.args.directed else pos_edge
                        label = torch.ones(pos_edge.size(1))
                    if self.args.reverse:
                        pos_edge = torch.stack([pos_edge[1], pos_edge[0]], dim=0)
                    perm = torch.argsort(pos_edge[0] * self.num_nodes + pos_edge[1])
                    pos_edge = pos_edge[:, perm]
                    count = torch.zeros(self.num_nodes, dtype=torch.long)
                    count.scatter_add_(0, pos_edge[0], torch.ones(pos_edge.size(1), dtype=torch.long))
                    edge_dict[stage]['edge_pos'] = pos_edge
                    edge_dict[stage]['label_pos'] = label
                    edge_dict[stage]['count'] = count
            elif 'source_node' in edge_split['train']:
                source = edge_split[stage]['source_node']
                target = edge_split[stage]['target_node']
                pos_edge = torch.stack([source, target], dim=0)

                label = torch.ones(pos_edge.size(1))
                if stage != 'train':
                    edge_dict[stage]['edge_pos'] = pos_edge
                    edge_dict[stage]['label_pos'] = label
                    target_neg = edge_split[stage]['target_node_neg']
                    neg_edge = torch.stack([source.repeat_interleave(target_neg.size(1)), target_neg.view(-1)])
                    edge_dict[stage]['edge'] = torch.cat([pos_edge, neg_edge], dim=1)
                    edge_dict[stage]['label'] = torch.cat([label, torch.zeros(neg_edge.size(1))])
                else:
                    # pos_edge = self.coalesc(pos_edge)
                    # label = torch.ones(pos_edge.size(1))
                    if self.args.strategy == 'asymmetric':
                        pos_edge = to_undirected(pos_edge, num_nodes=self.num_nodes) if not self.args.directed else pos_edge
                        label = torch.ones(pos_edge.size(1))
                    if self.args.reverse:
                        pos_edge = torch.stack([pos_edge[1], pos_edge[0]], dim=0)
                    perm = torch.argsort(pos_edge[0] * self.num_nodes + pos_edge[1])
                    pos_edge = pos_edge[:, perm]
                    count = torch.zeros(self.num_nodes, dtype=torch.long)
                    count.scatter_add_(0, pos_edge[0], torch.ones(pos_edge.size(1), dtype=torch.long))
                    edge_dict[stage]['edge_pos'] = pos_edge
                    edge_dict[stage]['label_pos'] = label
                    edge_dict[stage]['count'] = count
            else:
                assert False
        return edge_dict

    def process_for_dataset(self):
        # edge_weight, graph
        edge_index = self.data.edge_index
        self.edge_weight = row_normalize(torch.ones(edge_index.size(1), 1), edge_index, num_nodes=self.num_nodes).view(-1)
        self.graph = SparseTensor(
            row=edge_index[0], col=edge_index[1], value=self.edge_weight,
            sparse_sizes=(self.num_nodes, self.num_nodes), is_sorted=True
        )
        self.graph = self.graph.to(self.args.device) if self.args.method == 'FULL' else self.graph

        if self.args.use_valedges_as_input:
            val_edge = self.edge_dict['valid']['edge_pos']
            edge_index = torch.cat([self.data.edge_index, val_edge], dim=1)
            edge_index = to_undirected(edge_index, self.num_nodes)

            perm = torch.argsort(edge_index[0] * self.num_nodes + edge_index[1])
            edge_index = edge_index[:, perm]
            self.deg_test = torch.zeros(self.num_nodes, dtype=torch.long)
            self.deg_test.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
            self.rowptr_test = torch.zeros(self.num_nodes + 1, dtype=torch.long)
            torch.cumsum(self.deg_test, 0, out=self.rowptr_test[1:])
            self.edge_index_test = edge_index
            gc.collect()

            self.edge_weight_test = row_normalize(torch.ones(edge_index.size(1), 1), edge_index, num_nodes=self.num_nodes).view(-1)
            self.test_graph = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                value=self.edge_weight_test, sparse_sizes=(self.num_nodes, self.num_nodes), is_sorted=True
            )
            self.test_graph = self.test_graph.to(self.args.device) if self.args.method_te == 'FULL' else self.test_graph
        else:
            self.deg_test = self.deg
            self.rowptr_test = self.rowptr
            self.edge_index_test = self.data.edge_index
            self.edge_weight_test = self.edge_weight
            self.test_graph = self.graph

        if self.data.x is not None:
            self.data.x = self.data.x.to(torch.float).to(self.args.device)
            graph = self.graph.to(self.args.device)
            x = self.data.x
            for _ in range(self.args.layer_num):
                x = graph @ x
            self.x_v = x
            del graph
            torch.cuda.empty_cache()

            if self.args.use_valedges_as_input:
                x, x_last = self.data.x, None
                graph = self.test_graph.to(self.args.device)
                for _ in range(self.args.layer_num):
                    x = graph @ x
                self.x_v_test = x
                del graph
                torch.cuda.empty_cache()
            else:
                self.x_v_test = self.x_v

    def load_mf_embedding(self):
        if self.args.use_mf_embedding:
            path = osp.join(self.root, 'processed/MF_emb.pt')
            self.data.x = torch.FloatTensor(joblib.load(path))

    def handle_dataleakage(self):
        # self.data.edge_index = self.coalesc(self.data.edge_index)
        edge_index = self.data.edge_index
        edge_index = to_undirected(edge_index, num_nodes=self.num_nodes)
        eids = edge_index[0] * self.num_nodes + edge_index[1]
        edges_v = self.edge_dict['valid']['edge_pos']
        edges_v = to_undirected(edges_v, num_nodes=self.num_nodes) if not self.args.directed else edges_v
        eids_v = edges_v[0] * self.num_nodes + edges_v[1]
        edges_t = self.edge_dict['test']['edge_pos']
        edges_t = to_undirected(edges_t, num_nodes=self.num_nodes) if not self.args.directed else edges_t
        eids_t = edges_t[0] * self.num_nodes + edges_t[1]
        mask1 = np.isin(eids.numpy(), eids_v.numpy())
        mask2 = np.isin(eids.numpy(), eids_t.numpy())
        mask = np.logical_not(np.logical_or(mask1, mask2))
        self.data.edge_index = edge_index[:, torch.BoolTensor(mask)]
        print("\tnum of valid in train graph", mask1.sum())
        print("\tnum of test in train graph", mask2.sum())

    def load(self):
        start_time = time.time()
        self.load_mf_embedding()
        self.handle_dataleakage()

        self.data.edge_index, _ = add_self_loops(self.data.edge_index, num_nodes=self.num_nodes)
        perm = torch.argsort(self.data.edge_index[0] * self.num_nodes + self.data.edge_index[1])
        self.data.edge_index = self.data.edge_index[:, perm]
        self.deg = torch.zeros(self.num_nodes, dtype=torch.long)
        self.deg.scatter_add_(0, self.data.edge_index[0], torch.ones(self.data.edge_index.size(1), dtype=torch.long))
        self.rowptr = torch.zeros(self.num_nodes + 1, dtype=torch.long)
        torch.cumsum(self.deg, 0, out=self.rowptr[1:])
        gc.collect()
        self.process_for_dataset()

        pd_time = time.time() - start_time
        print("\tgraph.nnz, average", self.data.edge_index.size(1), self.data.edge_index.size(1) // self.num_nodes)
        print('preprocess step 3, time: {:f}s'.format(pd_time))
    # ===============================================

    # ===============================================
    # funcs of sampling data
    def reset_iter(self):
        self.times = [0., 0., 0.]
        self.perm_idx = {}
        for phase in self.edge_dict.keys():
            self.perm_idx[phase] = self.iter_idx(phase)
        self.perm_idx['node'] = self.iter_idx('node')

    def iter_idx(self, phase):
        if phase == 'train':
            num_pos = self.edge_dict[phase]['edge_pos'].size(1)
            bsize = self.args.bsize // 2
            if self.args.strategy == 'symmetric':
                perm = torch.randperm(num_pos).split(bsize)
                perm = perm if num_pos % bsize == 0 else perm[:-1]

                num_neg_sampled = num_pos * self.args.num_neg
                if self.args.eval_metric != 'mrr':
                    nlinks, num_neg = None, 0
                    while num_neg < num_neg_sampled:
                        nlinks_i = negative_sampling(self.data.edge_index, num_nodes=self.num_nodes, num_neg_samples=num_neg_sampled - num_neg)
                        nlinks = torch.cat([nlinks, nlinks_i], dim=1) if nlinks is not None else nlinks_i
                        num_neg = nlinks.size(1)
                else:
                    neg_dst = torch.randint(0, self.num_nodes, (num_neg_sampled,), dtype=torch.long)
                    if self.args.num_neg > 1:
                        neg_src = torch.repeat_interleave(self.edge_dict[phase]['edge_pos'][0], self.args.num_neg)
                    else:
                        neg_src = self.edge_dict[phase]['edge_pos'][0]
                    nlinks = torch.stack([neg_src, neg_dst], dim=0)
                self.edge_dict['train']['edge_neg'] = nlinks
                self.edge_dict['train']['label_neg'] = torch.zeros(nlinks.size(1))
            else:
                # 1. disrupt the order of elements within a group, but keep group order unchanged.
                #    [group refers to a row node, which has a number of neighbors]
                # 2. disrupt the order of groups
                # step 1
                perm = torch.randperm(num_pos)
                count = self.edge_dict[phase]['count']
                cids = torch.repeat_interleave(torch.arange(count.size(0)), count)
                cperm = torch.argsort(cids[perm])
                perm = perm[cperm]
                # step 2
                cids = torch.randperm(count.size(0))
                pcids = torch.repeat_interleave(cids, count)
                pcperm = torch.argsort(pcids)
                perm = perm[pcperm]

                num_neg_sampled = num_pos * self.args.num_neg
                if self.args.eval_metric != 'mrr':
                    # 1. randomly sample negative links and order them according to head node id.
                    # 2. keep negative group order the same as positive group order.
                    # 3. disrupt the order of negative links within a batch.
                    nlinks, num_neg = None, 0
                    # step 1
                    while num_neg < num_neg_sampled:
                        nlinks_i = negative_sampling(self.data.edge_index, num_nodes=self.num_nodes,
                                                     num_neg_samples=num_neg_sampled - num_neg)
                        nlinks = torch.cat([nlinks, nlinks_i], dim=1) if nlinks is not None else nlinks_i
                        num_neg = nlinks.size(1)
                    if self.args.reverse:
                        nlinks = torch.stack([nlinks[1], nlinks[0]], dim=0)
                else:
                    neg_dst = torch.randint(0, self.num_nodes, (num_neg_sampled,), dtype=torch.long)
                    if self.args.num_neg > 1:
                        neg_src = torch.repeat_interleave(self.edge_dict[phase]['edge_pos'][0], self.args.num_neg)
                    else:
                        neg_src = self.edge_dict[phase]['edge_pos'][0]
                    nlinks = torch.stack([neg_src, neg_dst], dim=0)
                    if self.args.reverse:
                        nlinks = torch.stack([nlinks[1], nlinks[0]], dim=0)
                    num_neg = nlinks.size(1)

                if self.args.eval_metric != 'mrr' or self.args.reverse:
                    nperm = torch.randperm(nlinks.size(1))
                    nlinks = nlinks[:, nperm]
                    sidx = torch.argsort(nlinks[0])
                    nlinks = nlinks[:, sidx]
                    ncount = torch.zeros(self.num_nodes, dtype=torch.long)
                    ncount.scatter_add_(0, nlinks[0], torch.ones(nlinks.size(1), dtype=torch.long))
                    # step 2
                    ncids = torch.repeat_interleave(cids, ncount)
                    ncperm = torch.argsort(ncids)
                    nlinks = nlinks[:, ncperm]
                    num_neg = num_neg - (num_pos % bsize) * self.args.num_neg
                    nlinks_sub = nlinks[:, :num_neg]
                    # step3
                    bcids = torch.repeat_interleave(torch.arange(perm.size(0) // bsize), bsize * self.args.num_neg)
                    bperm = torch.randperm(nlinks_sub.size(1))
                    bcperm = torch.argsort(bcids[bperm])
                    bperm = bperm[bcperm]
                    nlinks_sub = nlinks_sub[:, bperm]
                    if self.args.num_neg > 1:
                        perm_neg = torch.repeat_interleave(perm * self.args.num_neg, self.args.num_neg) \
                                   + torch.repeat_interleave(torch.arange(self.args.num_neg), perm.size(0))
                    else:
                        perm_neg = perm
                    nlinks[:, perm_neg[:num_neg]] = nlinks_sub

                self.edge_dict['train']['edge_neg'] = nlinks
                self.edge_dict['train']['label_neg'] = torch.zeros(nlinks.size(1))

                perm = perm.split(bsize)
                perm = perm if num_pos % bsize == 0 else perm[:-1]
                # self.times[1] += (time.time() - stime)
        elif phase == 'test' or phase == 'valid':
            bsize = int(np.ceil(self.edge_dict[phase]['edge'].size(1) * 1.0 / self.batch[phase]))
            perm = torch.arange(self.edge_dict[phase]['edge'].size(1)).split(bsize)
        else:
            bsize = int(np.ceil(self.num_nodes * 1.0 / self.batch['node']))
            perm = torch.arange(self.num_nodes).split(bsize)
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
            plinks, plabels = self.edge_dict[phase]['edge_pos'][:, index], self.edge_dict[phase]['label_pos'][index]
            index_neg = torch.repeat_interleave(index, self.args.num_neg) if self.args.num_neg > 1 else index
            nlinks = self.edge_dict[phase]['edge_neg'][:, index_neg]
            links = torch.cat([plinks, nlinks], dim=1)
            labels = torch.cat([plabels, torch.zeros(nlinks.size(1))])
        return links, labels

    def sample(self, phase):
        if self.args.fast_version:
            while True:
                global batch
                if len(batch[phase]) > 0:
                    params = batch[phase].pop(0)
                    break
                else:
                    time.sleep(0.1)
            graph, nlinks, ids_map, links, labels, ids_map_tail = params
        else:
            ### nlinks: links w new nids; links: w old nids
            if self.methods[phase] in ['BNS', 'LBNS']:
                s_n_1 = self.args.s_n_1 if phase == 'train' else self.args.s_n_1_test
                s_n = self.args.s_n if phase == 'train' else self.args.s_n_test
            else:
                s_n_1 = None
                s_n = None
            links, labels = self.get_links(phase)
            graph, nlinks, ids_map, links, labels, ids_map_tail = self.samplers[phase](links, labels, self, s_n_1, s_n, phase)

        if self.data.x is not None:
            x, x_v = self.data.x, self.x_v_test if phase == 'test' else self.x_v
            if self.methods[phase] != 'FULL':
                x_v = x_v[ids_map_tail].to(self.args.device)
                x_d = x[ids_map_tail] - x_v
                x = x[ids_map].to(self.args.device)
            else:
                x_d = None
        else:
            x, x_v, x_d = None, None, None
        labels = labels.to(self.args.device)
        if self.methods[phase] == 'FULL':
            graph = self.test_graph if phase == 'test' else self.graph
        if graph is not None:
            graph = [gi.to(self.args.device) for gi in graph] if isinstance(graph, list) else graph.to(self.args.device)
        return graph, x, labels, nlinks.to(self.args.device), ids_map.to(self.args.device), x_v, x_d

    def sample_eval(self, phase):
        links, labels = self.get_links(phase)
        return links, labels

    def get_nodeids(self):
        index = next(self.perm_idx['node'])
        if index is None:
            self.perm_idx['node'] = self.iter_idx('node')
            index = next(self.perm_idx['node'])
        return index

    def sample_node(self, phase, is_node=True):
        if is_node:
            if self.args.fast_version:
                while True:
                    global batch
                    if len(batch['node']) > 0:
                        params = batch['node'].pop(0)
                        break
                    else:
                        time.sleep(0.1)
                graph, ids_map, ids_0 = params
            else:
                if self.methods['node'] in ['BNS', 'LBNS']:
                    s_n_1 = self.args.s_n_1_test
                    s_n = self.args.s_n_test
                else:
                    s_n_1 = None
                    s_n = None
                ids_0 = self.get_nodeids()
                graph, ids_map, _ = self.samplers['node'](ids_0, self, s_n_1, s_n, phase)
            if self.methods['node'] == 'FULL':
                graph = self.test_graph if phase == 'test' else self.graph
            graph = [gi.to(self.args.device) for gi in graph] if isinstance(graph, list) else graph.to(self.args.device)
            x = self.data.x[ids_map].to(self.args.device) if self.data.x is not None else None
            return graph, x, ids_map
        else:
            ids_0 = self.get_nodeids()
            if self.data.x is not None:
                x_v = self.x_v_test[ids_0] if phase == 'test' else self.x_v[ids_0]
                x_d = self.data.x[ids_0] - x_v
            else:
                x_v, x_d = None, None
            return x_v, x_d

    def neigh_index(self, ids, num_nb=None, phase='train'):
        deg = self.deg_test[ids] if phase == 'test' and self.args.use_valedges_as_input else self.deg[ids]
        rowptr_graph = self.rowptr_test if phase == 'test' and self.args.use_valedges_as_input else self.rowptr
        rowptr_i = deg.new_zeros(ids.shape[0] + 1)
        torch.cumsum(deg, 0, out=rowptr_i[1:])

        deg_sum = deg.sum()
        if self.args.dataset != 'ogbl-':
            index = torch.arange(deg_sum)
            index -= torch.repeat_interleave(rowptr_i[:-1], deg)
            index += torch.repeat_interleave(rowptr_graph[ids], deg)
        else:
            index = neigh_index(rowptr_i.numpy(), rowptr_graph[ids].numpy(), deg.numpy())
            index = torch.LongTensor(index)

        if num_nb is None:
            return index
        else:
            idx_i_split = torch.LongTensor(choice_multi_range(deg.numpy(), num_nb.numpy())) \
                          + torch.repeat_interleave(rowptr_i[:-1], num_nb)

            mask = torch.zeros(index.size(0), dtype=torch.bool)
            mask[idx_i_split] = True
            sample_neigh_num_b = deg - num_nb
            ratio = sample_neigh_num_b * 1.0 / num_nb
            ratio[ratio < 1] = 1.0
            weight_1 = torch.repeat_interleave(torch.ones(ids.size(0)) * ratio, num_nb)
            weight_2 = torch.repeat_interleave(torch.ones(ids.size(0)), sample_neigh_num_b)
            return index[mask], index[torch.logical_not(mask)], weight_1, weight_2

    def neigh_index_v2(self, ids, num_nb=None, phase='train'):
        deg = self.deg_test[ids] if phase == 'test' and self.args.use_valedges_as_input else self.deg[ids]
        rowptr_graph = self.rowptr_test if phase == 'test' and self.args.use_valedges_as_input else self.rowptr

        ptr = deg.new_zeros(ids.shape[0] + 1)
        torch.cumsum(deg, 0, out=ptr[1:])

        deg_sum = deg.sum()
        if self.args.dataset != 'ogbn-':
            index = torch.arange(deg_sum)
            index -= torch.repeat_interleave(ptr[:-1], deg)
            index += torch.repeat_interleave(rowptr_graph[ids], deg)
        else:
            index = neigh_index(ptr.numpy(), rowptr_graph[ids].numpy(), deg.numpy())
            index = torch.LongTensor(index)

        if num_nb is None:
            return index
        else:
            idx = torch.LongTensor(choice_multi_range(deg.numpy(), num_nb.numpy())) + torch.repeat_interleave(ptr[:-1], num_nb)
            mask = torch.zeros(index.size(0), dtype=torch.bool)
            mask[idx] = True
            return index[mask], index[torch.logical_not(mask)]

    def negative_sampling(self, uids, edge_index, num_nodes, num_neg_samples):
        def sample(high: int, size: int, device=None):
            size = min(high, size)
            return torch.tensor(random.sample(range(high), size), device=device)

        if uids is not None:
            edgenum_sub = self.deg[uids].sum()
            size = uids.size(0) * num_nodes
            num_neg_samples = min(num_neg_samples, size - edgenum_sub)

            row, col = edge_index
            idx = row * num_nodes + col
            alpha = abs(1 / (1 - 1.1 * (edgenum_sub / size)))

            perm = sample(size, int(alpha * num_neg_samples))
            row = perm // num_nodes
            row = uids[row]
            col = perm % num_nodes
            perm = row * num_nodes + col
        else:
            size = num_nodes * num_nodes
            num_neg_samples = min(num_neg_samples, size - edge_index.size(1))

            row, col = edge_index
            idx = row * num_nodes + col
            alpha = abs(1 / (1 - 1.1 * (edge_index.size(1) / size)))

            perm = sample(size, int(alpha * num_neg_samples))

        mask = torch.from_numpy(np.isin(perm, idx.to('cpu'))).to(torch.bool)
        perm = perm[~mask][:num_neg_samples].to(edge_index.device)

        row = perm // num_nodes
        col = perm % num_nodes
        neg_edge_index = torch.stack([row, col], dim=0).long()
        return neg_edge_index
