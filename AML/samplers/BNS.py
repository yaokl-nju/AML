import torch
from utils.init_func import choice_multi_range_multi_sample, choice_multi_range
import pandas as pd
from torch_sparse import SparseTensor
import numpy as np
import time
import torch

def relabel_cpu(reindex, edges_col, oid_all):
    bool_id = torch.zeros(reindex.size(0), dtype=torch.bool)
    bool_id[edges_col] = True
    bool_id[oid_all] = True
    oid = torch.where(bool_id)[0]
    nid = reindex[oid]
    return oid[nid < 0]

def le(ids, deg, threshold):
    mask = deg < threshold
    mask_not = torch.logical_not(mask)
    return ids[mask], ids[mask_not], torch.where(mask)[0], torch.where(mask_not)[0]

def ge(ids, deg, threshold):
    mask = deg > threshold
    mask_not = torch.logical_not(mask)
    return ids[mask], ids[mask_not], torch.where(mask)[0], torch.where(mask_not)[0]

def leq(ids, deg, threshold):
    mask = deg <= threshold
    mask_not = torch.logical_not(mask)
    return ids[mask], ids[mask_not], torch.where(mask)[0], torch.where(mask_not)[0]

def geq(ids, deg, threshold):
    mask = deg >= threshold
    mask_not = torch.logical_not(mask)
    return ids[mask], ids[mask_not], torch.where(mask)[0], torch.where(mask_not)[0]

### original version
def BNS_edge(links, labels, dataset, s_n_1, s_n, phase):
    deg_graph = dataset.deg_test if phase == 'test' and dataset.args.use_valedges_as_input else dataset.deg
    edge_index_graph = dataset.edge_index_test if phase == 'test' and dataset.args.use_valedges_as_input else dataset.data.edge_index
    weight_graph = dataset.edge_weight_test if phase == 'test' and dataset.args.use_valedges_as_input else dataset.edge_weight
    rowptr_graph = dataset.rowptr_test if phase == 'test' and dataset.args.use_valedges_as_input else dataset.rowptr

    num_nodes = dataset.num_nodes
    ids_0 = torch.unique(links[0])
    if dataset.args.strategy == 'symmetric':
        ids_0 = torch.unique(torch.cat([ids_0, links[1]]))
    ids_cur = ids_0
    times = [0., 0., 0.]

    if dataset.args.dataset == 'large-datasets':
        ### for relatively large-scale datasets, this implementation is more efficient
        ids_record_2 = torch.LongTensor([])
        ids_record_1 = ids_0
    else:
        ids_record_1 = torch.zeros(num_nodes, dtype=torch.bool)
        ids_record_1[ids_0] = True
        ids_record_2 = torch.zeros(num_nodes, dtype=torch.bool)
    # print("ids_cur.size", ids_cur.size())

    stime = time.time()
    edges_row, edges_col, edges_weight = [], [], []
    for i in range(dataset.args.layer_num):
        # if ids_cur.size(0) < 0.5 * num_nodes:
        if True:
            if s_n_1 > 0 or i > 0:
                rho = dataset.args.rho
                s_n_nb = s_n_1 if i == 0 else s_n
                s_n_b = s_n_nb * rho

                ids_cur_1, ids_cur_2, _, _ = leq(ids_cur, deg_graph[ids_cur], s_n_nb * (rho + 1))
                ids_cur_1_1, ids_cur_1_2, _, _ = ge(ids_cur_1, deg_graph[ids_cur_1], s_n_nb)
                ids_cur_2_1, ids_cur_2_2, _, _ = le(ids_cur_2, deg_graph[ids_cur_2], s_n_nb * (rho + 1) * 10)

                deg_i_1_1 = torch.zeros(ids_cur_1_1.size(0), dtype=torch.long) + s_n_nb
                idx_i_1_1_1, idx_i_1_1_2, weight_1_1_1, weight_1_1_2 = \
                    dataset.neigh_index(ids_cur_1_1, deg_i_1_1, phase)
                rowids_1_1_1, colids_1_1_1 = edge_index_graph[:, idx_i_1_1_1]    # non-block
                rowids_1_1_2, colids_1_1_2 = edge_index_graph[:, idx_i_1_1_2]    # block
                idx_i_1_2 = dataset.neigh_index(ids_cur_1_2, phase=phase)
                rowids_1_2, colids_1_2 = edge_index_graph[:, idx_i_1_2]          # non-block

                idx_i_2_1_1, idx_i_2_1_2 = choice_multi_range_multi_sample(
                    deg_graph[ids_cur_2_1].numpy(), s_n_nb, s_n_b, rowptr_graph[ids_cur_2_1].numpy())
                rand_i_2_2_1 = (torch.rand(s_n_nb * ids_cur_2_2.size(0)) *
                                torch.repeat_interleave(deg_graph[ids_cur_2_2], s_n_nb)).to(torch.long)
                idx_i_2_2_1 = rand_i_2_2_1 + torch.repeat_interleave(rowptr_graph[ids_cur_2_2], s_n_nb)
                rand_i_2_2_2 = (torch.rand(s_n_b * ids_cur_2_2.size(0)) *
                                torch.repeat_interleave(deg_graph[ids_cur_2_2], s_n_b)).to(torch.long)
                idx_i_2_2_2 = rand_i_2_2_2 + torch.repeat_interleave(rowptr_graph[ids_cur_2_2], s_n_b)
                idx_i_2_1 = torch.cat([torch.LongTensor(idx_i_2_1_1), idx_i_2_2_1])
                idx_i_2_2 = torch.cat([torch.LongTensor(idx_i_2_1_2), idx_i_2_2_2])
                rowids_2_1, colids_2_1 = edge_index_graph[:, idx_i_2_1]           # non-block
                rowids_2_2, colids_2_2 = edge_index_graph[:, idx_i_2_2]           # block


                weight_1_2 = torch.ones(rowids_1_2.size(0))
                weight_2_1 = torch.ones(rowids_2_1.size(0)) * rho
                weight_2_2 = torch.ones(rowids_2_2.size(0))

                rowid_i = torch.cat([rowids_1_1_1, rowids_1_2, rowids_2_1, rowids_1_1_2, rowids_2_2])
                colid_i = torch.cat([colids_1_1_1, colids_1_2, colids_2_1, colids_1_1_2, colids_2_2])
                weight_i = torch.cat([weight_1_1_1, weight_1_2, weight_2_1, weight_1_1_2, weight_2_2])
                size_1st = rowids_1_1_1.size(0) + rowids_1_2.size(0) + rowids_2_1.size(0)
            else:
                idx_i = dataset.neigh_index(ids_cur, phase=phase)
                rowid_i, colid_i = edge_index_graph[:, idx_i]
                weight_i = weight_graph[idx_i]
                ids_cur_2 = torch.LongTensor([])
                size_1st = rowid_i.size(0)
        else:
            # idx_i = dataset.neigh_index(ids_cur, phase=phase)
            # rowid_i, colid_i = edge_index_graph[:, idx_i]
            # weight_i = weight_graph[idx_i]
            # ids_cur_2 = torch.LongTensor([])
            # size_1st = rowid_i.size(0)
            rowid_i, colid_i = edge_index_graph
            weight_i = weight_graph
            ids_cur_2 = torch.LongTensor([])
            size_1st = rowid_i.size(0)

        if i > 0:
            if dataset.args.dataset == 'large-datasets':
                ids_record_2 = torch.cat([ids_record_2, ids_cur_2])
                selfloop_id = torch.LongTensor(pd.unique(ids_record_2.numpy()))
                ids_record_2 = selfloop_id
            else:
                ids_record_2[ids_cur_2] = True
                selfloop_id = torch.where(ids_record_2)[0]
            rowid_i = torch.cat([rowid_i, selfloop_id])
            colid_i = torch.cat([colid_i, selfloop_id])
            if i == dataset.args.layer_num - 1:
                weight_i = torch.ones(rowid_i.size(0))
            else:
                weight_i = torch.cat([weight_i, torch.ones(selfloop_id.size(0))])
        else:
            rowid_i = torch.cat([rowid_i, ids_cur_2])
            colid_i = torch.cat([colid_i, ids_cur_2])
            weight_i = torch.cat([weight_i, torch.ones(ids_cur_2.size(0))])

        edges_row.append(rowid_i)
        edges_col.append(colid_i)
        if dataset.args.model != 'GAT':
            weight_i = dataset.row_normalize(weight_i, edges_row[-1], num_nodes)
        edges_weight.append(weight_i)

        if dataset.args.dataset == 'large-datasets':
            ids_record_2 = torch.cat([ids_record_2, edges_col[-1][size_1st:]])
            ids_record_1 = torch.cat([ids_record_1, ids_record_2])
            ids_cur = torch.LongTensor(pd.unique(edges_col[-1][:size_1st].numpy()))
            ids_record_1 = torch.cat([ids_record_1, ids_cur])
        else:
            ids_record_1[edges_col[-1]] = True
            ids_record_2[edges_col[-1][size_1st:]] = True
            ids_record_i_1 = torch.zeros(num_nodes, dtype=torch.bool)
            ids_record_i_1[edges_col[-1][:size_1st]] = True
            ids_cur = torch.where(ids_record_i_1)[0]
    times[0] += (time.time() - stime)

    stime = time.time()
    ### reindex ids in top-down manner
    if dataset.args.dataset == 'large-datasets':
        ids_all = torch.LongTensor(pd.unique(ids_record_1.numpy()))
    else:
        ids_all = torch.where(ids_record_1)[0]
    remap = torch.full((num_nodes, ), -1, dtype=torch.long)
    remap[ids_all] = torch.arange(ids_all.size(0))
    edges_col_new, edges_row_new, ids_0_re = [], [], remap[ids_0]
    for i in range(dataset.args.layer_num):
        edges_row_new.append(remap[edges_row[i]])
        edges_col_new.append(remap[edges_col[i]])

    reindex = torch.zeros(ids_all.size(0), dtype=torch.long) - 1
    reindex[ids_0_re] = torch.arange(ids_0_re.size(0))
    edge_row_i, oid_list, graph = reindex[edges_row_new[0]], [ids_0], []
    offset, oid_all = ids_0_re.size(0), ids_0_re
    for i in range(dataset.args.layer_num):
        # print("layer", i, oid_all.size(0))
        if i < dataset.args.layer_num - 1:
            # 这个主要考虑到if ids_cur.size(0) >= 0.9 * num_nodes，edges_row_new[i+1]会存在部分节点没有出现在edges_col_new[i]中
            oid = relabel_cpu(reindex, torch.cat([edges_col_new[i], edges_row_new[i+1]]), oid_all)
        else:
            oid = relabel_cpu(reindex, edges_col_new[i], oid_all)
        reindex[oid] = offset + torch.arange(oid.size(0))
        offset += oid.size(0)
        oid_all = torch.cat([oid_all, oid])
        edge_index_i = torch.stack([edge_row_i, reindex[edges_col_new[i]]], dim=0)
        # graph.append(SparseTensor(row=edge_index_i[0], col=edge_index_i[1], value=edges_weight[i].view(-1),
        #                           sparse_sizes=(oid_list[-1].size(0), offset)))
        graph.append(torch.sparse.FloatTensor(edge_index_i, edges_weight[i].view(-1),
                                              (oid_list[-1].size(0), offset)))
        if i < dataset.args.layer_num - 1:
            edge_row_i = reindex[edges_row_new[i + 1]]
        oid_list.append(ids_all[oid_all])
    graph.reverse()
    # print("layer", dataset.args.layer_num, oid_all.size(0))

    ids_all = oid_list[-1]
    rowids = reindex[remap[links[0]]]
    if dataset.args.strategy == 'asymmetric':
        uids = torch.unique(torch.cat([ids_0, links[1]]))
        reindex = torch.full((num_nodes, ), -1, dtype=torch.long)
        reindex[ids_0] = torch.arange(ids_0.size(0))
        tids = uids[reindex[uids] < 0]
        reindex[tids] = torch.arange(tids.size(0)) + ids_0.size(0)
        colids = reindex[links[1]]
        ids_all_tail = torch.cat([ids_0, tids])
        # assert ids_all_tail.size(0) == uids.size(0)
    else:
        colids = reindex[remap[links[1]]]
        ids_all_tail = ids_all
    nlinks = torch.stack([rowids, colids], dim=0)
    return [graph, nlinks, ids_all, links, labels, ids_all_tail]



def BNS_node(ids_0, dataset, s_n_1, s_n, phase):
    deg_graph = dataset.deg_test if phase == 'test' and dataset.args.use_valedges_as_input else dataset.deg
    edge_index_graph = dataset.edge_index_test if phase == 'test' and dataset.args.use_valedges_as_input else dataset.data.edge_index
    weight_graph = dataset.edge_weight_test if phase == 'test' and dataset.args.use_valedges_as_input else dataset.edge_weight
    rowptr_graph = dataset.rowptr_test if phase == 'test' and dataset.args.use_valedges_as_input else dataset.rowptr

    num_nodes = dataset.num_nodes
    ids_cur = ids_0
    times = [0., 0., 0.]

    if dataset.args.dataset == 'large-datasets':
        ### for relatively large-scale datasets, this implementation is more efficient
        ids_record_2 = torch.LongTensor([])
        ids_record_1 = ids_0
    else:
        ids_record_1 = torch.zeros(num_nodes, dtype=torch.bool)
        ids_record_1[ids_0] = True
        ids_record_2 = torch.zeros(num_nodes, dtype=torch.bool)
    # print("ids_cur.size", ids_cur.size())

    stime = time.time()
    edges_row, edges_col, edges_weight = [], [], []
    for i in range(dataset.args.layer_num):
        # if ids_cur.size(0) < 0.5 * num_nodes:
        if True:
            if s_n_1 > 0 or i > 0:
                rho = dataset.args.rho
                s_n_nb = s_n_1 if i == 0 else s_n
                s_n_b = s_n_nb * rho

                ids_cur_1, ids_cur_2, _, _ = leq(ids_cur, deg_graph[ids_cur], s_n_nb * (rho + 1))
                ids_cur_1_1, ids_cur_1_2, _, _ = ge(ids_cur_1, deg_graph[ids_cur_1], s_n_nb)
                ids_cur_2_1, ids_cur_2_2, _, _ = le(ids_cur_2, deg_graph[ids_cur_2], s_n_nb * (rho + 1) * 10)

                deg_i_1_1 = torch.zeros(ids_cur_1_1.size(0), dtype=torch.long) + s_n_nb
                idx_i_1_1_1, idx_i_1_1_2, weight_1_1_1, weight_1_1_2 = \
                    dataset.neigh_index(ids_cur_1_1, deg_i_1_1, phase)
                rowids_1_1_1, colids_1_1_1 = edge_index_graph[:, idx_i_1_1_1]    # non-block
                rowids_1_1_2, colids_1_1_2 = edge_index_graph[:, idx_i_1_1_2]    # block
                idx_i_1_2 = dataset.neigh_index(ids_cur_1_2, phase=phase)
                rowids_1_2, colids_1_2 = edge_index_graph[:, idx_i_1_2]          # non-block

                idx_i_2_1_1, idx_i_2_1_2 = choice_multi_range_multi_sample(
                    deg_graph[ids_cur_2_1].numpy(), s_n_nb, s_n_b, rowptr_graph[ids_cur_2_1].numpy())
                rand_i_2_2_1 = (torch.rand(s_n_nb * ids_cur_2_2.size(0)) *
                                torch.repeat_interleave(deg_graph[ids_cur_2_2], s_n_nb)).to(torch.long)
                idx_i_2_2_1 = rand_i_2_2_1 + torch.repeat_interleave(rowptr_graph[ids_cur_2_2], s_n_nb)
                rand_i_2_2_2 = (torch.rand(s_n_b * ids_cur_2_2.size(0)) *
                                torch.repeat_interleave(deg_graph[ids_cur_2_2], s_n_b)).to(torch.long)
                idx_i_2_2_2 = rand_i_2_2_2 + torch.repeat_interleave(rowptr_graph[ids_cur_2_2], s_n_b)
                idx_i_2_1 = torch.cat([torch.LongTensor(idx_i_2_1_1), idx_i_2_2_1])
                idx_i_2_2 = torch.cat([torch.LongTensor(idx_i_2_1_2), idx_i_2_2_2])
                rowids_2_1, colids_2_1 = edge_index_graph[:, idx_i_2_1]           # non-block
                rowids_2_2, colids_2_2 = edge_index_graph[:, idx_i_2_2]           # block


                weight_1_2 = torch.ones(rowids_1_2.size(0))
                weight_2_1 = torch.ones(rowids_2_1.size(0)) * rho
                weight_2_2 = torch.ones(rowids_2_2.size(0))

                rowid_i = torch.cat([rowids_1_1_1, rowids_1_2, rowids_2_1, rowids_1_1_2, rowids_2_2])
                colid_i = torch.cat([colids_1_1_1, colids_1_2, colids_2_1, colids_1_1_2, colids_2_2])
                weight_i = torch.cat([weight_1_1_1, weight_1_2, weight_2_1, weight_1_1_2, weight_2_2])
                size_1st = rowids_1_1_1.size(0) + rowids_1_2.size(0) + rowids_2_1.size(0)
            else:
                idx_i = dataset.neigh_index(ids_cur, phase=phase)
                rowid_i, colid_i = edge_index_graph[:, idx_i]
                weight_i = weight_graph[idx_i]
                ids_cur_2 = torch.LongTensor([])
                size_1st = rowid_i.size(0)
        else:
            # idx_i = dataset.neigh_index(ids_cur, phase=phase)
            # rowid_i, colid_i = edge_index_graph[:, idx_i]
            # weight_i = weight_graph[idx_i]
            # ids_cur_2 = torch.LongTensor([])
            # size_1st = rowid_i.size(0)
            rowid_i, colid_i = edge_index_graph
            weight_i = weight_graph
            ids_cur_2 = torch.LongTensor([])
            size_1st = rowid_i.size(0)

        if i > 0:
            if dataset.args.dataset == 'large-datasets':
                ids_record_2 = torch.cat([ids_record_2, ids_cur_2])
                selfloop_id = torch.LongTensor(pd.unique(ids_record_2.numpy()))
                ids_record_2 = selfloop_id
            else:
                ids_record_2[ids_cur_2] = True
                selfloop_id = torch.where(ids_record_2)[0]
            rowid_i = torch.cat([rowid_i, selfloop_id])
            colid_i = torch.cat([colid_i, selfloop_id])
            if i == dataset.args.layer_num - 1:
                weight_i = torch.ones(rowid_i.size(0))
            else:
                weight_i = torch.cat([weight_i, torch.ones(selfloop_id.size(0))])
        else:
            rowid_i = torch.cat([rowid_i, ids_cur_2])
            colid_i = torch.cat([colid_i, ids_cur_2])
            weight_i = torch.cat([weight_i, torch.ones(ids_cur_2.size(0))])

        edges_row.append(rowid_i)
        edges_col.append(colid_i)
        if dataset.args.model != 'GAT':
            weight_i = dataset.row_normalize(weight_i, edges_row[-1], num_nodes)
        edges_weight.append(weight_i)

        if dataset.args.dataset == 'large-datasets':
            ids_record_2 = torch.cat([ids_record_2, edges_col[-1][size_1st:]])
            ids_record_1 = torch.cat([ids_record_1, ids_record_2])
            ids_cur = torch.LongTensor(pd.unique(edges_col[-1][:size_1st].numpy()))
            ids_record_1 = torch.cat([ids_record_1, ids_cur])
        else:
            ids_record_1[edges_col[-1]] = True
            ids_record_2[edges_col[-1][size_1st:]] = True
            ids_record_i_1 = torch.zeros(num_nodes, dtype=torch.bool)
            ids_record_i_1[edges_col[-1][:size_1st]] = True
            ids_cur = torch.where(ids_record_i_1)[0]
    times[0] += (time.time() - stime)

    stime = time.time()
    ### reindex ids in top-down manner
    if dataset.args.dataset == 'large-datasets':
        ids_all = torch.LongTensor(pd.unique(ids_record_1.numpy()))
    else:
        ids_all = torch.where(ids_record_1)[0]
    remap = torch.full((num_nodes, ), -1, dtype=torch.long)
    remap[ids_all] = torch.arange(ids_all.size(0))
    edges_col_new, edges_row_new, ids_0_re = [], [], remap[ids_0]
    for i in range(dataset.args.layer_num):
        edges_row_new.append(remap[edges_row[i]])
        edges_col_new.append(remap[edges_col[i]])

    reindex = torch.zeros(ids_all.size(0), dtype=torch.long) - 1
    reindex[ids_0_re] = torch.arange(ids_0_re.size(0))
    edge_row_i, oid_list, graph = reindex[edges_row_new[0]], [ids_0], []
    offset, oid_all = ids_0_re.size(0), ids_0_re
    for i in range(dataset.args.layer_num):
        # print("layer", i, oid_all.size(0))
        if i < dataset.args.layer_num - 1:
            # 这个主要考虑到if ids_cur.size(0) >= 0.9 * num_nodes，edges_row_new[i+1]会存在部分节点没有出现在edges_col_new[i]中
            oid = relabel_cpu(reindex, torch.cat([edges_col_new[i], edges_row_new[i+1]]), oid_all)
        else:
            oid = relabel_cpu(reindex, edges_col_new[i], oid_all)
        reindex[oid] = offset + torch.arange(oid.size(0))
        offset += oid.size(0)
        oid_all = torch.cat([oid_all, oid])
        edge_index_i = torch.stack([edge_row_i, reindex[edges_col_new[i]]], dim=0)
        # graph.append(SparseTensor(row=edge_index_i[0], col=edge_index_i[1], value=edges_weight[i].view(-1),
        #                           sparse_sizes=(oid_list[-1].size(0), offset)))
        graph.append(torch.sparse.FloatTensor(edge_index_i, edges_weight[i].view(-1),
                                              (oid_list[-1].size(0), offset)))
        if i < dataset.args.layer_num - 1:
            edge_row_i = reindex[edges_row_new[i + 1]]
        oid_list.append(ids_all[oid_all])
    graph.reverse()
    ids_all = oid_list[-1]
    return [graph, ids_all, ids_0]





def split(index, ids, s_n, s_n_nb=1):
    if s_n_nb == 1:
        if s_n > 1:
            assert index.size(0) == ids.size(0) * s_n
            rand = (torch.rand(ids.size(0) * s_n_nb) * s_n).to(torch.long)
            # idx = rand + torch.repeat_interleave(torch.arange(0, index.size(0) - 1, s_n, dtype=torch.long), s_n_nb)
            idx = rand + torch.arange(ids.size(0), dtype=torch.long) * s_n
            mask = torch.zeros(index.size(0), dtype=torch.bool)
            mask[idx] = True
        else:
            mask = torch.ones(index.size(0), dtype=torch.bool)
    else:
        assert index.size(0) == ids.size(0) * s_n
        num_all = torch.full((ids.size(0), ), s_n, dtype=torch.long)
        num_nb = torch.full((ids.size(0), ), s_n_nb, dtype=torch.long)
        idx = torch.LongTensor(choice_multi_range(num_all.numpy(), num_nb.numpy()))
        idx = idx + torch.repeat_interleave(torch.arange(ids.size(0), dtype=torch.long) * s_n, s_n_nb)
        mask = torch.zeros(index.size(0), dtype=torch.bool)
        mask[idx] = True
    return index[mask], index[torch.logical_not(mask)]

def uniform_quant(x, alpha=1.0, b=4):
    x = x.div(alpha)
    x_c = x.clamp(max=1)
    xdiv = x_c.mul(2 ** b - 1)
    xhard = xdiv.round().div(2 ** b - 1).mul(alpha)
    return xhard

def binquant(x):
    m = x - x.mean()
    return torch.sign(m)
