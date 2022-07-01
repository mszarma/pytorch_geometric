import os.path as osp


import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import OGB_MAG, Reddit
import torch_geometric.transforms as T
from inference.rgcn import RGCN
from inference.rgat import GAT_HETERO
from inference.graphsage import SAGE_HETERO
from inference.pna_net import PNANet
from torch_geometric.utils import degree
from inference.edgeConv_net import EdgeConvNet


models_dict = {
    'rgcn': SAGE_HETERO,
    'rgat': GAT_HETERO,
    'pna_conv': PNANet,
    'edge_conv': EdgeConvNet,
}


def get_dataset(name, root):
    path = osp.dirname(osp.realpath(__file__))

    if name == 'ogbn-mag':
        transform = T.ToUndirected(merge=True)
        dataset = OGB_MAG(root=osp.join(path, root, "mag"),
                          preprocess='metapath2vec',
                          transform=transform)
    elif name == 'ogbn-products':
        dataset = PygNodePropPredDataset("ogbn-products",
                                         root=osp.join(path, root, "products"))
    elif name == 'reddit':
        dataset = Reddit(root=osp.join(path, root, "reddit"))

    return dataset


def get_model(name, params, device, metadata=None):
    try:
        model_type = models_dict[name]
    except KeyError:
        print(f"Model '{name}' not supported!")

    if name in ['rgcn', 'rgat']:
        if name == 'rgat':
            model = model_type(params['hidden_channels'],
                            params['output_channels'],
                            params['num_layers'],
                            params['num_heads'],)
        elif name == 'rgcn':
            model = model_type(params['hidden_channels'],
                    params['output_channels'],
                    params['num_layers'],)
        model.create_hetero(device, metadata)

    elif name == 'pna_conv':
        model = model_type(params['inputs_channels'],
                           params['hidden_channels'],
                           params['output_channels'],
                           params['num_layers'],
                           params['degree'])

    else:
        model = model_type(params['inputs_channels'],
                           params['hidden_channels'],
                           params['output_channels'],
                           params['num_layers'],)
    return model


def get_degree(loader):
    max_degree = -1
    for data in loader:
        d = degree(data.edge_index[1],
                   num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in loader:
        d = degree(data.edge_index[1],
                   num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg
