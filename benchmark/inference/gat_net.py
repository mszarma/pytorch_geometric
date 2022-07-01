import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import GATConv

from tqdm import tqdm


class GATBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads, last_layer=False,
                 **conv_kwargs):
        super().__init__()

        self.conv = GATConv(in_channels, out_channels, heads, **conv_kwargs)
        self.skip = Linear(
            in_channels, out_channels if last_layer else out_channels * heads)
        self.last_layer = last_layer

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        # TODO: how to use skip connection with NeighborLoader?
        # x = x + self.skip(?)
        return x if self.last_layer else F.elu(x)


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads,
                 num_layers):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(GATBlock(in_channels, hidden_channels, heads))
        for _ in range(num_layers - 2):
            self.layers.append(
                GATBlock(hidden_channels * heads, hidden_channels, heads))
        self.layers.append(
            GATBlock(hidden_channels * heads, out_channels, heads,
                     last_layer=True, concat=False))

    @torch.no_grad()
    def inference(self, subgraph_loader, device, x_all):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.layers))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        # Please note that this approach requires a lot of memory.
        for layer in self.layers:
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = layer(x, batch.edge_index.to(device))
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)

        pbar.close()
        return x_all
