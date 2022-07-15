import os.path as osp

from edgeconv import EdgeConvNet
from graphsage import SAGE_HETERO
from ogb.nodeproppred import PygNodePropPredDataset
from rgat import GAT_HETERO

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG, Reddit
from torch_geometric.nn.models.basic_gnn import GAT, GCN, PNA  # , edgeCNN

models_dict = {
    'edge_conv': EdgeConvNet,  # edgeCNN,
    'gat': GAT,
    'gcn': GCN,
    'pna_conv': PNA,
    'rgat': GAT_HETERO,
    'rgcn': SAGE_HETERO,
}


def get_dataset(name, root):
    path = osp.dirname(osp.realpath(__file__))

    if name == 'ogbn-mag':
        transform = T.ToUndirected(merge=True)
        dataset = OGB_MAG(root=osp.join(path, root, 'mag'),
                          preprocess='metapath2vec', transform=transform)
    elif name == 'ogbn-products':
        dataset = PygNodePropPredDataset('ogbn-products',
                                         root=osp.join(path, root, 'products'))
    elif name == 'reddit':
        dataset = Reddit(root=osp.join(path, root, 'reddit'))

    return dataset


def get_model(name, params, metadata=None):
    try:
        model_type = models_dict[name]
    except KeyError:
        print(f'Model {name} not supported!')

    if name in ['rgat', 'rgcn']:
        if name == 'rgat':
            model = model_type(params['hidden_channels'], params['num_layers'],
                               params['output_channels'], params['num_heads'])
        elif name == 'rgcn':
            model = model_type(params['hidden_channels'],
                               params['output_channels'], params['num_layers'])
        model.create_hetero(metadata)

    elif name == 'gat':
        kwargs = {}
        kwargs['heads'] = params['num_heads']
        model = model_type(params['inputs_channels'],
                           params['hidden_channels'], params['num_layers'],
                           params['output_channels'], **kwargs)

    elif name == 'pna_conv':
        kwargs = {}
        kwargs['aggregators'] = ['mean', 'min', 'max', 'std']
        kwargs['scalers'] = ['identity', 'amplification', 'attenuation']
        kwargs['deg'] = params['degree']
        model = model_type(params['inputs_channels'],
                           params['hidden_channels'], params['num_layers'],
                           params['output_channels'], **kwargs)

    else:
        model = model_type(params['inputs_channels'],
                           params['hidden_channels'], params['num_layers'],
                           params['output_channels'])
    return model
