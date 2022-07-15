import copy
import os.path as osp
from timeit import default_timer

import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm

# from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'products')
dataset = PygNodePropPredDataset('ogbn-products', root=osp.join(path))
# dataset = Reddit(path)

# Already send node features/labels to GPU for faster access during sampling:
data = dataset[0].to(device, 'x', 'y')
print(dataset)
print(data)
kwargs = {'batch_size': 1024, 'num_workers': 2, 'persistent_workers': True}
train_loader = None
# train_loader = NeighborLoader(data, input_nodes=data.train_mask,
#                               num_neighbors=[25, 10], shuffle=True, **kwargs)
# subgraph_loader_layer = None
subgraph_loader_layer = NeighborLoader(copy.copy(data), input_nodes=None,
                                       num_neighbors=[-1], shuffle=False,
                                       **kwargs)

subgraph_loader_batch = NeighborLoader(copy.copy(data), input_nodes=None,
                                       num_neighbors=[-1, -1], shuffle=False,
                                       **kwargs)

# No need to maintain these features during evaluation:
del subgraph_loader_layer.data.x, subgraph_loader_layer.data.y
# # Add global node index information.
subgraph_loader_layer.data.num_nodes = data.num_nodes
subgraph_loader_layer.data.n_id = torch.arange(data.num_nodes)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
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


model = SAGE(dataset.num_features, 128, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)
    pbar.close()

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def test_layer_wise():
    model.eval()
    y_hat = model.inference(data.x, subgraph_loader_layer).argmax(dim=-1)
    y = data.y.to(y_hat.device)
    y = y  # pep
    accs = []
    # for mask in [data.train_mask, data.val_mask, data.test_mask]:
    #     accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    return accs


@torch.no_grad()
def test_batch_wise():

    model.eval()

    total_examples = total_correct = 0
    for batch in tqdm(subgraph_loader_batch):
        batch = batch.to(device)
        batch_size = batch.batch_size
        out = model(batch.x, batch.edge_index)[:batch_size]
        pred = out.argmax(dim=-1)

        total_examples += batch_size
        total_correct += int((pred == batch.y[:batch_size]).sum())

    return total_correct / total_examples


for epoch in range(1, 2):
    start = default_timer()
    acc = test_layer_wise()
    stop = default_timer()
    # print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
    #       f'Test: {test_acc:.4f}')
    print(f"Inference layer-wise time: {stop - start}")
    start = default_timer()
    total_acc = test_batch_wise()
    stop = default_timer()
    print(f'Epoch: {epoch:02d}, Total acc: {total_acc:.4f}')
    print(f"Inference batch-wise time: {stop - start}")
