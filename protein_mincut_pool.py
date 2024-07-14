# %%
import time

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score as NMI
from torch.nn import Linear
from torch_geometric import utils
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, Sequential, dense_mincut_pool
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.gnn.datasets import PhaseAssociationDataset

torch.manual_seed(0)  # for reproducibility

# Load data
dataset = PhaseAssociationDataset('data', force_reload=False)
n = (len(dataset) + 9) // 10

for d in dataset:
    d.edge_index, d.edge_weight = gcn_norm(
        d.edge_index, d.edge_weight, d.num_nodes,
        add_self_loops=False, dtype=d.x.dtype)

test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DataLoader(test_dataset, batch_size=20)
val_loader = DataLoader(val_dataset, batch_size=20)
train_loader = DataLoader(train_dataset, batch_size=20)

# %%
# Normalized adjacency matrix
# data.edge_index, data.edge_weight = gcn_norm(
#     data.edge_index, data.edge_weight, data.num_nodes,
#     add_self_loops=False, dtype=data.x.dtype)

# %%


class Net(torch.nn.Module):
    def __init__(self,
                 mp_units,
                 mp_act,
                 in_channels,
                 n_clusters,
                 mlp_units=[],
                 mlp_act="Identity"):
        super().__init__()

        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)

        # Message passing layers
        mp = [
            (GraphConv(in_channels, mp_units[0]),
             'x, edge_index, edge_weight -> x'),
            mp_act
        ]
        for i in range(len(mp_units)-1):
            mp.append(
                (GraphConv(mp_units[i], mp_units[i+1]),
                 'x, edge_index, edge_weight -> x'))
            mp.append(mp_act)
        self.mp = Sequential('x, edge_index, edge_weight', mp)
        out_chan = mp_units[-1]

        # MLP layers
        self.mlp = torch.nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)
        self.mlp.append(Linear(out_chan, n_clusters))

    def forward(self, x, edge_index, edge_weight):

        # Propagate node feats
        x = self.mp(x, edge_index, edge_weight)

        # Cluster assignments (logits)
        s = self.mlp(x)

        # Obtain MinCutPool losses
        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight)
        _, _, mc_loss, o_loss = dense_mincut_pool(x, adj, s)

        return torch.softmax(s, dim=-1), mc_loss, o_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data = data.to(device)
model = Net([2], "ELU", dataset.num_features, dataset.num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# %%


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        for i in range(10):
            optimizer.zero_grad()
            _, mc_loss, o_loss = model(
                data.x, data.edge_index, data.edge_weight)
            loss = mc_loss + o_loss
            loss.backward()
            loss_all += data.y.size(0) * float(loss)
            optimizer.step()
    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    nmi = 0
    loss_all = 0
    for data in loader:
        data = data.to(device)
        clust, mc_loss, o_loss = model(
            data.x, data.edge_index, data.edge_weight)
        loss = mc_loss + o_loss
        loss_all += data.y.size(0) * float(loss)
        nmi += NMI(clust.max(1)[1].cpu(), data.y.cpu())
    rand_idx = np.random.randint(1, 1001, size=15)
    print(
        torch.stack((clust.max(1)[1][rand_idx].cpu(),
                     data.y[rand_idx].cpu()), 0))
    return loss_all / len(loader.dataset), nmi / len(loader.dataset)


times = []
best_val_nmi = test_nmi = 0
best_val_loss = float('inf')
patience = start_patience = 50

for epoch in range(1, 15001):
    start = time.time()
    train_loss = train(epoch)
    _, train_nmi = test(train_loader)
    val_loss, val_nmi = test(val_loader)
    if val_loss < best_val_loss:
        test_loss, test_nmi = test(test_loader)
        best_val_nmi = val_nmi
        patience = start_patience
    else:
        patience -= 1
        if patience == 0:
            break
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, '
          f'Train NMI: {train_nmi:.3f}, Val Loss: {val_loss:.3f}, '
          f'Val NMI: {val_nmi:.3f}, Test Loss: {test_loss:.3f}, '
          f'Test NMI: {test_nmi:.3f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

# %%
