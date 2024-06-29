import threading
import time
from global_val import *
import torch
import samplers
import numpy as np
from torch_geometric.utils import negative_sampling

class _Sampler(threading.Thread):
    def __init__(self, dataset, phase='train', buffer_size=20, daemon=True):
        super(_Sampler, self).__init__(daemon=daemon)
        self.__is_running = True
        self.daemon = daemon
        self.init_params(dataset, phase, buffer_size)
        self.start()

    def terminate(self):
        self.__is_running = False

    def init_params(self, dataset, phase='train', buffer_size=20):
        if phase == 'train':
            self.sample_idx = torch.arange(dataset.edge_dict[phase]['edge_pos'].size(1))
        else:
            self.sample_idx = torch.arange(dataset.num_nodes)
        self.b_num = dataset.batch[phase]

        self.buffer_size = buffer_size
        self.phase = phase
        self.dataset = dataset
        self.b_count = 0
        self.perm = self.iter_idx()
        method = dataset.methods[phase]
        self.sampler = getattr(samplers, method+'_edge' if phase != 'node' else method+'_node')

        if dataset.methods[phase] in ['BNS', 'LBNS']:
            if phase == 'train' or phase == 'valid':
                self.s_n_1 = dataset.args.s_n_1
                self.s_n = dataset.args.s_n
            else:
                self.s_n_1 = dataset.args.s_n_1_te
                self.s_n = dataset.args.s_n_te
        else:
            self.s_n_1 = None
            self.s_n = None

    def iter_idx(self):
        if self.phase == 'train':
            bsize = self.dataset.args.bsize // 2
            if self.dataset.args.strategy == 'symmetric':
                num_pos = self.dataset.edge_dict[self.phase]['edge_pos'].size(1)
                perm = torch.randperm(num_pos).split(bsize)
                perm = perm if num_pos % bsize == 0 else perm[:-1]

                num_neg_sampled = num_pos * self.dataset.args.num_neg
                if self.dataset.args.eval_metric != 'mrr':
                    nlinks, num_neg = None, 0
                    while num_neg < num_neg_sampled:
                        nlinks_i = negative_sampling(self.dataset.data.edge_index, num_nodes=self.dataset.num_nodes, num_neg_samples=num_neg_sampled - num_neg)
                        nlinks = torch.cat([nlinks, nlinks_i], dim=1) if nlinks is not None else nlinks_i
                        num_neg = nlinks.size(1)
                else:
                    neg_dst = torch.randint(0, self.dataset.num_nodes, (num_neg_sampled,), dtype=torch.long)
                    neg_src = torch.repeat_interleave(self.dataset.edge_dict[self.phase]['edge_pos'][0], self.dataset.args.num_neg)
                    nlinks = torch.stack([neg_src, neg_dst], dim=0)
                self.edge_neg = nlinks
                self.label_neg = torch.zeros(nlinks.size(1))
            else:
                # 1. disrupt the order of elements within a group, but keep group order unchanged.
                #    [group refers to a row node, which has a number of neighbors]
                # 2. disrupt the order of groups
                # step 1
                num_pos = self.dataset.edge_dict[self.phase]['edge_pos'].size(1)
                perm = torch.randperm(num_pos)
                count = self.dataset.edge_dict[self.phase]['count']
                cids = torch.repeat_interleave(torch.arange(count.size(0)), count)
                cperm = torch.argsort(cids[perm])
                perm = perm[cperm]
                # step 2
                cids = torch.randperm(count.size(0))
                pcids = torch.repeat_interleave(cids, count)
                pcperm = torch.argsort(pcids)
                perm = perm[pcperm]

                num_neg_sampled = num_pos * self.dataset.args.num_neg
                if self.dataset.args.eval_metric != 'mrr':
                    # 1. randomly sample negative links and order them according to head node id.
                    # 2. keep negative group order the same as positive group order.
                    # 3. disrupt the order of negative links within a batch.
                    # step 1
                    nlinks, num_neg = None, 0
                    while num_neg < num_neg_sampled:
                        nlinks_i = negative_sampling(self.dataset.data.edge_index, num_nodes=self.dataset.num_nodes, num_neg_samples=num_neg_sampled - num_neg)
                        nlinks = torch.cat([nlinks, nlinks_i], dim=1) if nlinks is not None else nlinks_i
                        num_neg = nlinks.size(1)
                    if self.dataset.args.reverse:
                        nlinks = torch.stack([nlinks[1], nlinks[0]], dim=0)
                else:
                    neg_dst = torch.randint(0, self.dataset.num_nodes, (num_neg_sampled,), dtype=torch.long)
                    if self.dataset.args.num_neg > 1:
                        neg_src = torch.repeat_interleave(self.dataset.edge_dict[self.phase]['edge_pos'][0], self.dataset.args.num_neg)
                    else:
                        neg_src = self.dataset.edge_dict[self.phase]['edge_pos'][0]
                    nlinks = torch.stack([neg_src, neg_dst], dim=0)
                    if self.dataset.args.reverse:
                        nlinks = torch.stack([nlinks[1], nlinks[0]], dim=0)
                    num_neg = nlinks.size(1)

                if self.dataset.args.eval_metric != 'mrr' or self.dataset.args.reverse:
                    nperm = torch.randperm(nlinks.size(1))
                    nlinks = nlinks[:, nperm]
                    sidx = torch.argsort(nlinks[0])
                    nlinks = nlinks[:, sidx]
                    ncount = torch.zeros(self.dataset.num_nodes, dtype=torch.long)
                    ncount.scatter_add_(0, nlinks[0], torch.ones(nlinks.size(1), dtype=torch.long))
                    # step 2
                    ncids = torch.repeat_interleave(cids, ncount)
                    ncperm = torch.argsort(ncids)
                    nlinks = nlinks[:, ncperm]
                    num_neg = num_neg - (num_pos % bsize) * self.dataset.args.num_neg
                    nlinks_sub = nlinks[:, :num_neg]
                    # step3
                    bcids = torch.repeat_interleave(torch.arange(perm.size(0) // bsize), bsize * self.dataset.args.num_neg)
                    bperm = torch.randperm(nlinks_sub.size(1))
                    bcperm = torch.argsort(bcids[bperm])
                    bperm = bperm[bcperm]
                    nlinks_sub = nlinks_sub[:, bperm]
                    if self.dataset.args.num_neg > 1:
                        perm_neg = torch.repeat_interleave(perm * self.dataset.args.num_neg, self.dataset.args.num_neg) \
                                   + torch.repeat_interleave(torch.arange(self.dataset.args.num_neg), perm.size(0))
                    else:
                        perm_neg = perm
                    nlinks[:, perm_neg[:num_neg]] = nlinks_sub

                self.edge_neg = nlinks
                self.label_neg = torch.zeros(nlinks.size(1))

                perm = perm.split(bsize)
                perm = perm if num_pos % bsize == 0 else perm[:-1]
        else:
            bsize = int(np.ceil(self.dataset.num_nodes * 1.0 / self.dataset.batch['node']))
            perm = torch.arange(self.dataset.num_nodes).split(bsize)
        for ids in perm:
            yield ids
        yield None

    def get_iter_idx(self):
        index = next(self.perm)
        if index is None:
            self.perm = self.iter_idx()
            index = next(self.perm)
        if self.phase == 'node':
            return index
        else:
            plinks, plabels = self.dataset.edge_dict[self.phase]['edge_pos'][:, index], self.dataset.edge_dict[self.phase]['label_pos'][index]
            index_neg = torch.repeat_interleave(index, self.dataset.args.num_neg) if self.dataset.args.num_neg > 1 else index
            nlinks = self.edge_neg[:, index_neg]
            links = torch.cat([plinks, nlinks], dim=1)
            labels = torch.cat([plabels, torch.zeros(nlinks.size(1))])
            return links, labels

    def pass_to_global(self, params):
        global batch
        batch[self.phase].append(params)

    def run(self):
        while self.__is_running:
            global batch
            while len(batch[self.phase]) < self.buffer_size:
                if self.phase != 'node':
                    links, labels = self.get_iter_idx()
                    batch_data = self.sampler(links, labels, self.dataset, self.s_n_1, self.s_n, self.phase)
                else:
                    ids_0 = self.get_iter_idx()
                    batch_data = self.sampler(ids_0, self.dataset, self.s_n_1, self.s_n, self.phase)
                self.pass_to_global(batch_data)
                self.b_count += 1
            time.sleep(0.1)