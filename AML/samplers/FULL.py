import torch
from torch_geometric.utils import negative_sampling
from utils.init_func import row_normalize
from torch_sparse import SparseTensor

def FULL_edge(links, labels, dataset, s_n_1=None, s_n=None, phase='train'):
    # ids_all = torch.unique(links[0])
    ids_all = torch.arange(dataset.num_nodes)
    times = [0., 0., 0.]
    return [None, links, ids_all, links, labels, ids_all]

def FULL_node(ids_0, dataset, s_n_1=None, s_n=None, phase='train'):
    ids_all = torch.arange(dataset.num_nodes)
    return [None, ids_all, ids_0]


