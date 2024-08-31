# %%
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torchvision import transforms

from src.clustering.dataset import (NDArrayTransform, NDArrayTransformX,
                                    PhasePicksDataset)

# %%
# Load data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def scale_data(sample):
    scaler_std = StandardScaler()
    return torch.tensor(scaler_std.fit_transform(sample),
                        dtype=torch.float64,
                        device=device)


ds = PhasePicksDataset(
    root_dir='data/raw',
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv',
    transform=transforms.Compose([
        NDArrayTransformX(drop_cols=['station'],
                          cat_cols=['phase']),
        scale_data]),
    target_transform=NDArrayTransform(),
    catalog_transform=NDArrayTransform(),
)

n_feats = ds[0].x.shape[1]

# %%
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(ds, [0.7, 0.3], generator=generator)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

pairs = []
for i, ds in enumerate(train_loader):
    a1, p, a2, n = lmu.get_all_pairs_indices(ds.y.squeeze()[:10000])
    pos = torch.randperm(len(a1))[:2500]
    neg = torch.randperm(len(a2))[:25000]

    select = a1[pos], p[pos], a2[neg], n[neg]

    pairs.append(tuple(p.to(device) for p in select))

# %%


class NN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_feats, h_feats, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(h_feats, out_feats, dtype=torch.float64),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


# %%
model = NN(n_feats, 64, 32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_func = losses.NTXentLoss(embedding_regularizer=LpRegularizer())
# loss_func = losses.SupConLoss(embedding_regularizer=LpRegularizer())


# %%


def train(loader):
    model.train()
    train_loss = 0

    for i, data in enumerate(loader):

        data.x.to(device)
        optimizer.zero_grad()
        embeddings = model(data.x)

        loss = loss_func(embeddings.squeeze(), data.y.squeeze(), pairs[i])
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if i % 100 == 0:
            print(f"Batch {i}, Loss: {np.round(loss.item(), 3)}")

    return train_loss / len(loader)


@torch.no_grad()
def test(loader):
    model.eval()
    ari = 0
    for data in loader:
        data.x.to(device)
        embeddings = model(data.x)

        # Cluster embeddings
        n_clusters = len(data.y.squeeze().unique())
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
            embeddings.cpu().squeeze().numpy())
        labels = kmeans.labels_

        ari += adjusted_rand_score(data.y.squeeze().cpu(), labels)

    return ari / len(loader)


# %%
start = time()
for epoch in range(2):
    train_loss = train(train_loader)
    test_ari = test(test_loader)
    end = time() - start
    print(f"Epoch {epoch+1}, "
          f"Train Loss: {np.round(train_loss, 3)}, "
          f"Test ARI: {np.round(test_ari, 3)}, "
          f"Time: {np.round(end/60, 2)} min")

# %%
# Cluster without Contrastive Step
ari = 0
for data in train_loader:
    n_clusters = len(data.y.squeeze().unique())
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
        data.x.cpu().squeeze().numpy())
    labels = kmeans.labels_

    ari += adjusted_rand_score(data.y.squeeze().cpu(), labels)

print(f"ARI: {np.round(ari / len(train_loader),3)}")

# %%

for i, data in enumerate(test_loader):
    data.x.to(device)
    embeddings = model(data.x).cpu().detach().squeeze()

    plt.scatter(embeddings.squeeze()[:, 1], embeddings.squeeze()[
                :, 4], c=data.y.squeeze().cpu())

    if i == 2:
        break
# %%

torch.save(model.state_dict(), 'model2')
