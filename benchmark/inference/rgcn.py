

import torch
import torch.nn.functional as F
from torch.nn import Parameter, ModuleDict, ModuleList, Linear, ParameterDict
from tqdm import tqdm
import copy
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import RGCNConv # current implementation don't support newest API?

class RGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, node_types, edge_types):
        super(RGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # `ModuleDict` does not allow tuples :(
        self.rel_lins = ModuleDict({
            f'{key[0]}_{key[1]}_{key[2]}': Linear(in_channels, out_channels,
                                                  bias=False)
            for key in edge_types
        })

        self.root_lins = ModuleDict({
            key: Linear(in_channels, out_channels, bias=True)
            for key in node_types
        })

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins.values():
            lin.reset_parameters()
        for lin in self.root_lins.values():
            lin.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        out_dict = {}
        for key, x in x_dict.items():
            out_dict[key] = self.root_lins[key](x)

        for key, adj_t in adj_t_dict.items():
            key_str = f'{key[0]}_{key[1]}_{key[2]}'
            x = x_dict[key[0]]
            out = self.rel_lins[key_str](adj_t.matmul(x, reduce='mean'))
            out_dict[key[2]].add_(out)

        return out_dict

class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 num_nodes_dict, x_types, edge_types, dropout=0.5):
        super(RGCN, self).__init__()
        print(num_nodes_dict)
        print(x_types)
        print(num_layers)
        num_layers
        print(type(num_nodes_dict))
        node_types = list(num_nodes_dict.keys())

        self.embs = ParameterDict({
            key: Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        self.convs = ModuleList()
        self.convs.append(
            RGCNConv(in_channels, hidden_channels, node_types, edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, node_types,
                         edge_types))
        self.convs.append(
            RGCNConv(hidden_channels, out_channels, node_types, edge_types))

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embs.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        x_dict = copy.copy(x_dict)
        for key, emb in self.embs.items():
            x_dict[key] = emb

        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, adj_t_dict)
            for key, x in x_dict.items():
                x_dict[key] = F.relu(x)
                x_dict[key] = F.dropout(x, p=self.dropout,
                                        training=self.training)
        return self.convs[-1](x_dict, adj_t_dict)

    @torch.no_grad()
    def inference(self, loader, device, data=None):
        out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
        edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
        for batch in tqdm(loader):
            batch = batch.to(device, 'edge_index')
            batch_size = batch['paper'].batch_size
            # out = self(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
            out = self(batch.x_dict, batch.adj_t_dict)['paper'][:batch_size]
            pred = out.argmax(dim=-1)
