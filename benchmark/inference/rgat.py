
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import GATConv
from torch_geometric.nn import to_hetero

class GAT_HETERO:
    def __init__(self, hidden_channels, output_channels, num_layers) -> None:
        self.model = None
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_layers = num_layers

    def create_hetero(self, device, metadata):
        model = GAT_FOR_HETERO(self.hidden_channels, self.output_channels, self.num_layers)
        # print(model)
        self.model = to_hetero(model, metadata, aggr='sum')
        # print(self.model)

    def inference(self, loader, device):
        self.model.eval()
        for batch in tqdm(loader):
            batch = batch.to(device)
            batch_size = batch['paper'].batch_size
            print(batch)
            print(self.model)
            out = self.model(batch.x_dict, batch.edge_index_dict)
            #out['paper'][:batch_size]
            print(out)

class GAT_FOR_HETERO(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv((-1, -1), hidden_channels, add_self_loops=False))
        for i in range(num_layers - 2):
            self.convs.append(GATConv((-1, -1), hidden_channels, add_self_loops=False))
        self.convs.append(GATConv((-1, -1), out_channels, add_self_loops=False))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
        return x

# class RGAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels,
#                  num_relations, num_layers):
#         super().__init__()
#         self.conv1 = RGATConv(in_channels, hidden_channels, num_relations)
#         self.convs = torch.nn.ModuleList()
#         for i in range(num_layers - 1):
#             self.convs.append(RGATConv(hidden_channels, hidden_channels, num_relations))
#         self.conv2 = RGATConv(hidden_channels, hidden_channels, num_relations)
#         self.lin = torch.nn.Linear(hidden_channels, out_channels)

#     def forward(self, x, edge_index, edge_type):
#         x = self.conv1(x, edge_index, edge_type).relu()
#         for conv in self.convs:
#             x = conv(x, edge_index, edge_type).relu()
#         x = self.lin(x)
#         return F.log_softmax(x, dim=-1)

#     @torch.no_grad()
#     def inference(self, loader, device, data=None):
#         for batch in tqdm(loader):
#             batch = batch.to(device, 'edge_index')
#             batch_size = batch['paper'].batch_size
#             out = self(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
#             pred = out.argmax(dim=-1)