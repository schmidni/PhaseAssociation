# %%
import torch
from pytorch_metric_learning import distances, losses, miners, reducers  # noqa
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning.utils.accuracy_calculator import \
    AccuracyCalculator
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torchvision import transforms

from src.clustering.dataset import (NDArrayTransform, NDArrayTransformX,
                                    PhasePicksDataset)

# %% Load data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def scale_data(sample):
    scaler_std = StandardScaler()
    return torch.tensor(scaler_std.fit_transform(sample),
                        dtype=torch.float64,
                        device=device)


ds = PhasePicksDataset(
    root_dir='data/reference/30s_2hz/',
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

# number of features: ds[0].x.shape[1]
# number of catalogs: len(ds) = 5000
# number of picks per catalog: ds[0].x.shape[0]
# number of events per catalog: len(np.unique(ds[0].y)) - 1

generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset, _ = random_split(
    ds, [0.2, 0.1, 0.7], generator=generator)

test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)

print('Data Loader Prepared')

# %%


class NN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_feats, h_feats, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(h_feats, 2*h_feats, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(2*h_feats, h_feats, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(h_feats, out_feats, dtype=torch.float64)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


n_features = ds[0].x.shape[1]
model = NN(n_features, 128, 8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# %%


def train(model, loss_func, mining_func, device,
          train_loader, optimizer, epoch):
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
        if i % 50 == 0:
            num_triplets = mining_func.num_triplets if hasattr(
                mining_func, "num_triplets") else None
            print("Epoch {} Iteration {}: Loss = {}, "
                  "Number of mined triplets = {}".format(
                      epoch, i, loss, num_triplets
                  ))


def get_all_embeddings(dataloader, model, device=None):
    model.eval()
    all_embeddings = []
    all_labels = []

    if device is None:
        device = next(model.parameters()).device  # Default to model's device

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.x.squeeze().to(device)
            labels = batch.y.squeeze().to(device).long()

            embeddings = model(inputs)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    return all_embeddings, all_labels


def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels
    test_labels = test_labels
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    print(
        "Test set accuracy (Precision@1) = {}".format(
            accuracies["precision_at_1"]))


# %% pytorch-metric-learning stuff

# distance = distances.CosineSimilarity()
# reducer = reducers.ThresholdReducer(low=0)
# loss_func = losses.TripletMarginLoss(
#     margin=0.2, distance=distance, reducer=reducer)
# mining_func = miners.TripletMarginMiner(
#     margin=0.2, distance=distance, type_of_triplets="semihard")

# loss_func = losses.NTXentLoss(embedding_regularizer=LpRegularizer())
loss_func = losses.SupConLoss(embedding_regularizer=LpRegularizer())
mining_func = miners.BatchEasyHardMiner()

accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

# %%
num_epochs = 2
for epoch in range(1, num_epochs + 1):
    train(model, loss_func, mining_func, device,
          train_loader, optimizer, epoch)
    # test(train_loader, test_loader, model, accuracy_calculator)

# %%


@torch.no_grad()
def test_cluster(loader, model, device):
    model.eval()
    ari = 0
    i = 0
    for data in loader:
        data.x.to(device)
        embeddings = model(data.x)

        # Cluster embeddings
        n_clusters = len(data.y.squeeze().unique())
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
            embeddings.cpu().squeeze().numpy())
        labels = kmeans.labels_

        ari += adjusted_rand_score(data.y.squeeze().cpu(), labels)
        i += 1
        if i % 10 == 0:
            print(f"Batch {i}, ARI: {ari / i}")

    return ari / len(loader)


print(test_cluster(test_loader, model, device))

# %%
