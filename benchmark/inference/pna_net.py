
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import PNAConv
from torch_geometric.nn import to_hetero

class PNANet(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, num_layers, degree):
        super().__init__()
        self.aggregators = ['mean', 'min', 'max', 'std']
        self.scalers = ['identity', 'amplification', 'attenuation']
        self.convs = torch.nn.ModuleList()
        self.convs.append(PNAConv(input_channels, hidden_channels, self.aggregators, self.scalers, degree))
        for i in range(num_layers - 2):
            self.convs.append(PNAConv(hidden_channels, hidden_channels, self.aggregators, self.scalers, degree))
        self.convs.append(PNAConv(hidden_channels, out_channels, self.aggregators, self.scalers, degree))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
        return x

    @torch.no_grad()
    def inference(self, subgraph_loader, device, x_all):
        pbar = tqdm(total=len(subgraph_loader) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all