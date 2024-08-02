# %%
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from src.gnn.datasets import PhaseAssociationGraphDataset

# %%
# Load data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = PhaseAssociationGraphDataset('data', force_reload=False)
n = len(dataset)

# %%
test_dataset = dataset[:int(n*0.2)]
train_dataset = dataset[int(n*0.2):]

test_loader = DataLoader(test_dataset, batch_size=1)
train_loader = DataLoader(train_dataset, batch_size=1)

pairs = []
for i, ds in enumerate(train_loader):
    a1, p, a2, n = lmu.get_all_pairs_indices(ds.y)
    pos = torch.randperm(len(a1))[:1000]
    neg = torch.randperm(len(a2))[:10000]

    select = a1[pos], p[pos], a2[neg], n[neg]

    pairs.append(tuple(p.to(device) for p in select))

# %%


class NN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, out_feats),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class GNN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h


# %%
# model = GNN(dataset.num_features, 32, 2).to(device)
model = NN(dataset.num_features, 32, 16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# loss_func = losses.NTXentLoss(embedding_regularizer=LpRegularizer())
loss_func = losses.SupConLoss(embedding_regularizer=LpRegularizer())


# %%


def train(loader):
    model.train()
    train_loss = 0

    for i, data in enumerate(loader):
        data.to(device)
        optimizer.zero_grad()
        embeddings = model(data.x)
        loss = loss_func(embeddings, data.y, pairs[i])
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if i % 100 == 0:
            print(f"Batch {i}, Loss: {loss.item()}")

    return train_loss / len(loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    ari = 0
    for data in loader:
        data.to(device)
        embeddings = model(data.x)

        # Cluster embeddings
        kmeans = KMeans(n_clusters=15, random_state=0).fit(
            embeddings.cpu().numpy())
        labels = kmeans.labels_

        ari += adjusted_rand_score(data.y.cpu(), labels)

    return ari / len(loader.dataset)


for epoch in range(1):
    train_loss = train(train_loader)
    test_ari = test(test_loader)
    print(f"Epoch {epoch}, Train Loss: {train_loss}, Test ARI: {test_ari}")

# %%
# without Contrastive Step
ari = 0
for data in train_loader:

    # Cluster embeddings
    kmeans = KMeans(n_clusters=15, random_state=0).fit(data.x.numpy())
    labels = kmeans.labels_

    ari += adjusted_rand_score(data.y.cpu(), labels)

print(f"ARI: {ari / len(train_loader.dataset)}")

# %%

for i, data in enumerate(test_loader):
    data.to(device)
    embeddings = model(data.x).cpu().detach()

    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=data.y.cpu())

    if i == 2:
        break
# %%
