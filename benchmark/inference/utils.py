import os.path as osp

import torch
from torch_geometric.datasets import OGB_MAG, Reddit
import torch_geometric.transforms as T
from gat_net import GATNet
from rgcn import RGCN
from rgat import GAT_HETERO
from graphsage import SAGE_HETERO
from pna_net import PNANet
from edgeConv_net import EdgeConvNet
from torch_geometric.utils import degree

models_dict = {
    'gat': GATNet,
    'rgcn': SAGE_HETERO,
    'rgat': GAT_HETERO,
    'pna_conv': PNANet,
    'edge_conv': EdgeConvNet,
}


# TODO: remove ogb_node_dataset; it's a hack to fix hang on ogb import
def get_dataset(name, root, ogb_node_dataset=None):
    path = osp.dirname(osp.realpath(__file__))

    if name == 'ogbn-mag':
        transform = T.ToUndirected(merge=True)
        dataset = OGB_MAG(root=osp.join(path, root, "mag"),
                          preprocess='metapath2vec', transform=transform)
    elif name == 'ogbn-products':
        dataset = ogb_node_dataset("ogbn-products",
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
            model = model_type(
                params['hidden_channels'],
                params['output_channels'],
                params['num_layers'],
                params['num_heads'],
            )
        elif name == 'rgcn':
            model = model_type(
                params['hidden_channels'],
                params['output_channels'],
                params['num_layers'],
            )
        model.create_hetero(device, metadata)

    elif name == 'gat':
        model = model_type(params['inputs_channels'],
                           params['hidden_channels'],
                           params['output_channels'], params['num_heads'],
                           params['num_layers'])

    elif name == 'pna_conv':
        model = model_type(params['inputs_channels'],
                           params['hidden_channels'],
                           params['output_channels'], params['num_layers'],
                           params['degree'])

    else:
        model = model_type(
            params['inputs_channels'],
            params['hidden_channels'],
            params['output_channels'],
            params['num_layers'],
        )
    return model


def get_degree(loader):
    max_degree = -1
    for data in loader:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes,
                   dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in loader:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes,
                   dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg
