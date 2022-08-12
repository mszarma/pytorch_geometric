import os.path as osp

from hetero_gat import HeteroGAT
from hetero_sage import HeteroGraphSAGE
from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import SparseTensor

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG, Reddit
from torch_geometric.nn.models.basic_gnn import GAT, GCN, PNA, EdgeCNN

models_dict = {
    'edge_cnn': EdgeCNN,
    'gat': GAT,
    'gcn': GCN,
    'pna': PNA,
    'rgat': HeteroGAT,
    'rgcn': HeteroGraphSAGE,
}


def get_dataset(name, root, use_sparse_tensor=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), root, name)
    if name == 'ogbn-mag':
        transform = T.ToUndirected(merge=True)
        dataset = OGB_MAG(root=path, preprocess='metapath2vec',
                          transform=transform)
    elif name == 'ogbn-products':
        dataset = PygNodePropPredDataset('ogbn-products', root=path)
    elif name == 'Reddit':
        dataset = Reddit(root=path)

    data = dataset[0]
    if use_sparse_tensor:
        if name == 'ogbn-mag':
            data.adj_t_dict = {}
            for edge_type, edge_index in data.edge_index_dict.items():
                src_type, _, dst_type = edge_type
                data.adj_t_dict[edge_type] = SparseTensor(
                    row=edge_index[0], col=edge_index[1],
                    sparse_sizes=(data.x_dict[src_type].size(0),
                                  data.x_dict[dst_type].size(0))).t()
        else:
            data = T.ToSparseTensor()(data)

    return data, dataset.num_classes


def get_model(name, params, metadata=None):
    Model = models_dict.get(name, None)
    assert Model is not None, f'Model {name} not supported!'

    if name == 'rgat':
        return Model(metadata, params['hidden_channels'], params['num_layers'],
                     params['output_channels'], params['num_heads'])

    if name == 'rgcn':
        return Model(metadata, params['hidden_channels'], params['num_layers'],
                     params['output_channels'])

    if name == 'gat':
        return Model(params['inputs_channels'], params['hidden_channels'],
                     params['num_layers'], params['output_channels'],
                     heads=params['num_heads'])

    if name == 'pna':
        return Model(params['inputs_channels'], params['hidden_channels'],
                     params['num_layers'], params['output_channels'],
                     aggregators=['mean', 'min', 'max', 'std'],
                     scalers=['identity', 'amplification',
                              'attenuation'], deg=params['degree'])

    return Model(params['inputs_channels'], params['hidden_channels'],
                 params['num_layers'], params['output_channels'])
