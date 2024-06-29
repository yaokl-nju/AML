import torch
from torch_geometric.utils import negative_sampling
from utils.init_func import row_normalize
from torch_sparse import SparseTensor

def FULL(links, labels, dataset):
    ids_all = torch.unique(torch.cat([links[0], links[1]]))
    reindex = torch.zeros(dataset.num_nodes, dtype=torch.long)
    reindex[ids_all] = torch.arange(ids_all.size(0))
    nlinks = torch.stack([reindex[links[0]], reindex[links[1]]], dim=0)
    return [nlinks, ids_all, links, labels]


