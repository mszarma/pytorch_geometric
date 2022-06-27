
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import RGATConv

class RGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_relations, num_layers):
        super().__init__()
        self.conv1 = RGATConv(in_channels, hidden_channels, num_relations)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(RGATConv(hidden_channels, hidden_channels, num_relations))
        self.conv2 = RGATConv(hidden_channels, hidden_channels, num_relations)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        for conv in self.convs:
            x = conv(x, edge_index, edge_type).relu()
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)

    @torch.no_grad()
    def inference(self, loader, device, data=None):
        for batch in tqdm(loader):
            batch = batch.to(device, 'edge_index')
            batch_size = batch['paper'].batch_size
            out = self(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
            pred = out.argmax(dim=-1)