# %%
import matplotlib.pyplot as plt
import torch
from pytorch_metric_learning import distances, losses, miners, reducers  # noqa
from pytorch_metric_learning.utils.accuracy_calculator import \
    AccuracyCalculator
from pytorch_metric_learning.utils.inference import FaissKMeans
from sklearn.cluster import KMeans
# from pytorch_metric_learning.regularizers import LpRegularizer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler  # , StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torchvision import transforms

from src.clustering.dataset import (NDArrayTransform, NDArrayTransformX,
                                    PhasePicksDataset)

# %% Load data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def scale_data(sample):
    # scaler_std = StandardScaler()
    # return torch.tensor(scaler_std.fit_transform(sample),
    #                     dtype=torch.float32,
    #                     device=device)
    scaler = MinMaxScaler()
    return torch.tensor(scaler.fit_transform(sample),
                        dtype=torch.float32,
                        device=device)


ds = PhasePicksDataset(
    root_dir='data/reference/5s_5hz',
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv',
    transform=transforms.Compose([
        NDArrayTransformX(drop_cols=['station'],
                          cat_cols=['phase']),
        scale_data
    ]),
    target_transform=NDArrayTransform(),
    catalog_transform=NDArrayTransform(),
)

# number of features: ds[0].x.shape[1]
# number of catalogs: len(ds) = 5000
# number of picks per catalog: ds[0].x.shape[0]
# number of events per catalog: len(np.unique(ds[0].y)) - 1

generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset, _ = random_split(
    ds, [0.9, 0.09, 0.01], generator=generator)

test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)

print('Data Loader Prepared')

# %%


class NN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_feats, h_feats, dtype=torch.float32),
            nn.BatchNorm1d(h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, 2 * h_feats, dtype=torch.float32),
            nn.BatchNorm1d(2 * h_feats),
            nn.ReLU(),
            nn.Linear(2 * h_feats, h_feats, dtype=torch.float32),
            nn.BatchNorm1d(h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, out_feats, dtype=torch.float32)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return F.normalize(logits, p=2, dim=1)


n_features = ds[0].x.shape[1]
model = NN(n_features, 64, 16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=1000, gamma=0.9)


# %%


def train(model, loss_func, mining_func, device,
          train_loader, optimizer, epoch,
          scheduler: torch.optim.lr_scheduler.StepLR):
    model.train()
    for i, data in enumerate(train_loader):
        mask = data.y != -1
        x = data.x[mask].squeeze().to(device)
        y = data.y[mask].squeeze().to(device).long()

        optimizer.zero_grad()
        embeddings = model(x)
        indices_tuple = mining_func(embeddings, y)
        loss = loss_func(embeddings, y, indices_tuple)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 250 == 0:
            num_triplets = mining_func.num_triplets if hasattr(
                mining_func, "num_triplets") else None
            print("Epoch {} Iteration {}: Loss = {}, "
                  "Number of mined triplets = {}, "
                  "Learning Rate = {}".format(
                      epoch, i, loss, num_triplets,
                      scheduler.get_last_lr()))


# %% Testing Function

@torch.no_grad()
def test_cluster(loader, model, accuracy_calculator, device):
    model.eval()
    ari = 0
    # pat1 = 0
    for data in loader:
        mask = data.y.squeeze() != -1
        x = data.x.squeeze().to(device)
        y = data.y.squeeze().to(device).long()

        embeddings = model(x)

        acc = accuracy_calculator.get_accuracy(
            embeddings[mask], y[mask])
        ari += acc["ari"]
        # pat1 += acc["precision_at_1"]

    print(f"Test set ARI: {ari / len(loader)}")
    # print(f"Test set Precision@1: {pat1 / len(loader)}")

    return ari / len(loader)


# %% pytorch-metric-learning stuff
distance = distances.CosineSimilarity()
# reducer = reducers.ThresholdReducer(low=0)
reducer = reducers.AvgNonZeroReducer()
loss_func = losses.TripletMarginLoss(
    margin=0.2, distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="semihard")

# loss_func = losses.NTXentLoss(embedding_regularizer=LpRegularizer())
# loss_func = losses.SupConLoss(embedding_regularizer=LpRegularizer())
# mining_func = miners.BatchEasyHardMiner()


cluster_function_faiss = FaissKMeans(
    niter=20, gpu=device == "cuda", min_points_per_centroid=10)


def cluster_function_sklearn(embeddings, num_clusters):
    try:
        embeddings = embeddings.cpu().numpy()
    except AttributeError:
        pass
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(
        embeddings)
    return kmeans.labels_


class CustomAccuracyCalculator(AccuracyCalculator):
    def calculate_ari(self, query_labels, cluster_labels, **kwargs):
        try:
            query_labels = query_labels.cpu().numpy()
            cluster_labels = cluster_labels.cpu().numpy()
        except AttributeError:
            pass
        return adjusted_rand_score(query_labels, cluster_labels)

    def requires_clustering(self):
        return super().requires_clustering() + ["ari"]


accuracy_calculator = CustomAccuracyCalculator(
    include=("ari",), kmeans_func=cluster_function_sklearn)

# %%
num_epochs = 6
for epoch in range(1, num_epochs + 1):
    loss_func = losses.TripletMarginLoss(
        margin=0.1*epoch, distance=distance, reducer=reducer)
    train(model, loss_func, mining_func, device,
          train_loader, optimizer, epoch, scheduler)
    test_cluster(test_loader, model, accuracy_calculator, device)

# %%
# scatter plot of embeddings

data = next(iter(test_loader))
x = data.x.squeeze().to(device)
embeddings = model(x)
embeddings = embeddings.cpu().squeeze().detach().numpy()
labels = data.y.squeeze().cpu().numpy()
plt.figure(figsize=(10, 10))
plt.scatter(embeddings[:, 0], embeddings[:, 2], c=labels, s=1)
plt.colorbar()
plt.title('Embeddings')
plt.xlabel('Embedding 1')
plt.ylabel('Embedding 2')
plt.show()
