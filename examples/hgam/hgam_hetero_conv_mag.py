from typing import List, Optional, Tuple, Union
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, Linear, SAGEConv
from torch_geometric.utils import trim_x, trim_adj, trim_to_layer
from torch_geometric.typing import Adj, OptPairTensor, Size

use_sparse_tensor = False
trim = True

transform = T.ToSparseTensor(
    remove_edge_index=False) if use_sparse_tensor else None
if transform is None:
    transform = T.ToUndirected(merge=True)
else:
    transform = T.Compose([T.ToUndirected(merge=True), transform])

dataset = OGB_MAG(root='../../data', preprocess='metapath2vec',
                  transform=transform)

data = dataset[0]
device = 'cpu'


class HierarchicalSAGEConv(SAGEConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, num_sampled_nodes_per_hop=None, num_sampled_edges_per_hop=None, layer=None, edge_index_key=None, edge_indexes=None) -> Tensor:
#    #     print(f' Kwargs SAGECONV: {kwargs}' )
        if num_sampled_nodes_per_hop and num_sampled_edges_per_hop:
#             # print(layer)
#             # print(num_sampled_nodes_per_hop)
#             print("fw before ", id(edge_index))
            print("aaa")
            print(id(edge_indexes[edge_index_key]))
            print("aaa")
            edge_indexes[edge_index_key] = trim_adj(edge_index,layer, num_sampled_nodes_per_hop[1] if len(num_sampled_nodes_per_hop) == 2 else num_sampled_nodes_per_hop, num_sampled_edges_per_hop)
            edge_index = edge_indexes[edge_index_key]
# print("fw after:", id(edge_index))
#             # trim_adj(edge_index,layer, num_sampled_nodes_per_hop[1] if len(num_sampled_nodes_per_hop) == 2 else num_sampled_nodes_per_hop, num_sampled_edges_per_hop)
#             print("edgeeee index adter trim:", edge_index.size())
            x = trim_x(x, layer, num_sampled_nodes_per_hop)
            if edge_index.numel() == 0:
#                 # print("HEEEEELLLLOOOO")
                x = x[1] if isinstance(x, Tuple) else x
                return x

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]

        # if num_sampled_nodes_per_hop and num_sampled_edges_per_hop:
        #     out = trim_x(out, layer, num_sampled_nodes_per_hop[1] if len(num_sampled_nodes_per_hop) == 2 else num_sampled_nodes_per_hop)
        #     x_r = trim_x(x_r, layer, num_sampled_nodes_per_hop[1] if len(num_sampled_nodes_per_hop) == 2 else num_sampled_nodes_per_hop) if self.root_weight else None

        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

class HierarchicalHeteroGraphSage(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, trim):
        super().__init__()
        self.trim = trim
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ('paper', 'cites', 'paper'):
                    HierarchicalSAGEConv((-1, -1), hidden_channels),
                    ('author', 'writes', 'paper'):
                    HierarchicalSAGEConv((-1, -1), hidden_channels),
                    ('paper', 'rev_writes', 'author'):
                    HierarchicalSAGEConv((-1, -1), hidden_channels),
                    ('author', 'affiliated_with', 'institution'):
                    HierarchicalSAGEConv((-1, -1), hidden_channels),
                    ('institution', 'rev_affiliated_with', 'author'):
                    HierarchicalSAGEConv((-1, -1), hidden_channels),
                    ('paper', 'has_topic', 'field_of_study'):
                    HierarchicalSAGEConv((-1, -1), hidden_channels),
                    ('field_of_study', 'rev_has_topic', 'paper'):
                    HierarchicalSAGEConv((-1, -1), hidden_channels),
                }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, num_sampled_edges_dict=None,
                num_sampled_nodes_dict=None):
     #   print("num_sampled_nodes_dict: ", num_sampled_nodes_dict)
       
        for i, conv in enumerate(self.convs):
            kwargs_dict = {}
            # if not use_sparse_tensor and trim:
            #     x_dict, edge_index_dict = trim_to_layer(
            #         layer=i, node_attrs=x_dict, edge_index=edge_index_dict,
            #         num_nodes_per_layer=num_sampled_nodes,
            #         num_edges_per_layer=num_sampled_edges)
       #     print(f' Kwargs: {kwargs_dict}' )
            #print(edge_index_dict)
            if self.trim:
                kwargs_dict.update({"num_sampled_edges_per_hop_dict":num_sampled_edges_dict})
                kwargs_dict.update({"num_sampled_nodes_per_hop_dict":num_sampled_nodes_dict})
                layer = {k : i for k in edge_index_dict.keys()}
                edge_index_key = {k : k for k in edge_index_dict.keys()}
                kwargs_dict.update({"layer_dict":layer})
                kwargs_dict.update({"edge_index_key_dict":edge_index_key})
                edge_indexes = {k : edge_index_dict for k in edge_index_dict.keys()}
                kwargs_dict.update({"edge_indexes_dict":edge_indexes})
                print("...")
                for k in edge_index_dict.keys():
                    print(id(edge_index_dict[k]))
                print("...")
                # x_dict, edge_index_dict,_ = trim_to_layer(i,num_sampled_nodes_dict,num_sampled_edges_dict,x_dict,edge_index_dict)
            x_dict = conv(x_dict, edge_index_dict, **kwargs_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return self.lin(x_dict['paper'])


model = HierarchicalHeteroGraphSage(hidden_channels=64, out_channels=dataset.num_classes,
                  num_layers=3, trim=trim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

kwargs = {'batch_size': 1024, 'num_workers': 0}
train_loader = NeighborLoader(data, num_neighbors=[10] * 3, shuffle=True,
                              input_nodes=('paper', data['paper'].train_mask),
                              **kwargs)

val_loader = NeighborLoader(data, num_neighbors=[10] * 3, shuffle=False,
                            input_nodes=('paper', data['paper'].val_mask),
                            **kwargs)


def train():
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        edge_index_dict = (batch.adj_t_dict
                           if use_sparse_tensor else batch.edge_index_dict)
        optimizer.zero_grad()
        batch_size = batch['paper'].batch_size
        #print("nodes: ", batch.num_sampled_nodes_dict)
        #print("Edges:", batch.num_sampled_edges_dict)
        if trim:
            out = model(batch.x_dict, edge_index_dict,
                        num_sampled_nodes_dict=batch.num_sampled_nodes_dict,
                        num_sampled_edges_dict=batch.num_sampled_edges_dict)
        else:
            out = model(batch.x_dict, edge_index_dict)

        loss = F.cross_entropy(out[:batch_size], batch['paper'].y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    total_examples = total_correct = 0
    for batch in tqdm(loader):
        batch = batch.to(device, 'edge_index')
        batch_size = batch['paper'].batch_size
        edge_index_dict = (batch.adj_t_dict
                           if use_sparse_tensor else batch.edge_index_dict)
        if trim:
            out = model(
                batch.x_dict, edge_index_dict,
                num_sampled_edges=batch.num_sampled_edges_dict,
                num_sampled_nodes=batch.num_sampled_nodes_dict)[:batch_size]
        else:
            out = model(batch.x_dict, edge_index_dict)[:batch_size]

        pred = out.argmax(dim=-1)

        total_examples += batch_size
        total_correct += int((pred == batch['paper'].y[:batch_size]).sum())

    return total_correct / total_examples


for epoch in range(1, 6):
    print("Train")
    loss = train()
    print("Test")
    val_acc = test(val_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')