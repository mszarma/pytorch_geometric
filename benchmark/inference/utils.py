import os.path as osp

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import OGB_MAG, Reddit
import torch_geometric.transforms as T
from rgcn import RGCN
from rgat import GAT_HETERO
from graphsage import SAGE_HETERO
import copy
#from torch_geometric.nn import to_hetero

models_dict = {
    'rgcn': SAGE_HETERO,
    'rgat': GAT_HETERO,
}

def get_dataset(name, root):
    path = osp.dirname(osp.realpath(__file__))

    if name == 'ogbn-mag':
        transform = T.ToUndirected(merge=True)
        #dataset = PygNodePropPredDataset("ogbn-mag",
    #                 root=osp.join(path, root, "mag"))
        dataset = OGB_MAG(root=osp.join(path, root, "mag"), transform=transform)
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
        model = model_type(params['hidden_channels'],
                           params['output_channels'],
                           params['num_layers'],)
        model.create_hetero(device, metadata)
        print(model)
    # if name in ['rgcn', 'rgat']:
    #     model = model_type(params['inputs_channels'],
    #                        params['hidden_channels'],
    #                        params['output_channels'],
    #                        params['num_layers'],
    #                        params['num_nodes_dict'],
    #                        params['x_types'],
    #                        params['edge_types'])
    else:
        model = model_type(params['inputs_channels'],
                           params['hidden_channels'],
                           params['output_channels'],
                           params['num_layers'],)
    return model