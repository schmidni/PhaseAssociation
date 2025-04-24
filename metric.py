# %%
import itertools
from datetime import datetime

import torch
from pytorch_metric_learning import distances, losses, miners, reducers
from pytorch_metric_learning.utils.accuracy_calculator import \
    AccuracyCalculator
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torchvision import transforms

from src.clustering.dataset import (NDArrayTransform, NDArrayTransformX,
                                    PhasePicksDataset, ReduceDatetime)
from src.plotting.embeddings import plot_embeddings_reduced

# %% Load data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running models on {device}.')


def scale_data(sample):
    scaler = MinMaxScaler()
    return torch.tensor(scaler.fit_transform(sample),
                        dtype=torch.float32,
                        device=device)


ds = PhasePicksDataset(
    root_dir='data/reference/low_freq',
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv',
    transform=transforms.Compose([
        ReduceDatetime(datetime=datetime(2025, 1, 1, 0, 0, 0)),
        NDArrayTransformX(drop_cols=['station'],
                          cat_cols=['phase']),
        scale_data,
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
    ds, [0.02, 0.01, 0.97], generator=generator)

test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)

print('Data Loader Prepared')

# %% DEFINITIONS
###############################################################################


class PickSetTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 embed_dim=128,
                 num_heads=4,
                 num_layers=2,
                 max_picks=100):
        super(PickSetTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = 16  # Final embedding dimension

        # Project input features to Transformer embedding dim
        self.feature_proj = nn.Linear(input_dim, embed_dim)

        # Learnable positional encoding
        self.positional_enc = nn.Parameter(torch.rand(max_picks, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=256
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # MLP projection head to 16D embedding
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )

        # L2 normalization
        self.normalize = nn.functional.normalize

    def forward(self, pick_features, pick_times):
        """
        pick_features: Tensor (batch, N, feature_dim)
        pick_times: Tensor (batch, N) — used for sorting
        """
        batch_size, num_picks, _ = pick_features.shape

        # Sort picks by time
        sorted_idx = torch.argsort(pick_times, dim=1)
        batch_indices = torch.arange(
            batch_size).unsqueeze(1).expand(-1, num_picks)
        sorted_features = pick_features[batch_indices, sorted_idx]

        # Project features
        x = self.feature_proj(sorted_features)  # (batch, N, embed_dim)

        # Add positional encodings
        # (1, N, embed_dim)
        x = x + self.positional_enc[:num_picks].unsqueeze(0)

        # Transformer expects (N, batch, embed_dim)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch, N, embed_dim)

        # Reorder to original pick order
        inv_idx = torch.argsort(sorted_idx, dim=1)
        x = x[batch_indices, inv_idx]

        # Project to 16D embedding
        x = self.output_proj(x)  # (batch, N, 16)

        # Normalize
        emb = self.normalize(x, p=2, dim=-1)
        return emb


# %%
# # Example usage:
# batch_size = 4
# max_picks = 50
# feature_dim = ds[0].x.shape[1]
# model = PickSetTransformer(input_dim=feature_dim, max_picks=1000)
# # Dummy input
# # pick_feats = torch.randn(batch_size, max_picks, feature_dim)
# pick_times = pick_feats[..., 0]  # assuming last feature is pick_time
# # output shape: (batch, max_picks, embed_dim)
# embeddings = model(pick_feats, pick_times)
# print(embeddings.shape)  # (4, 50, 128) for this example

# %% MODEL


class NN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.BatchNorm1d(h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, 2 * h_feats),
            nn.BatchNorm1d(2 * h_feats),
            nn.ReLU(),
            nn.Linear(2 * h_feats, h_feats),
            nn.BatchNorm1d(h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, out_feats)
        )

    def forward(self, pick_features, *args, **kwargs):
        """
        pick_features: Tensor of shape (batch, picks, feature_dim)
        """
        batch_size, num_picks, _ = pick_features.shape
        # (batch * picks, feature_dim)
        x = pick_features.view(-1, pick_features.size(-1))
        # (batch * picks, out_feats)
        x = self.linear_relu_stack(x)
        x = F.normalize(x, p=2, dim=1)
        # back to (batch, picks, out_feats)
        x = x.view(batch_size, num_picks, -1)
        return x


def train(model, loss_func, mining_func, device,
          train_loader, optimizer, epoch,
          scheduler: torch.optim.lr_scheduler.StepLR):
    model.train()
    for i, data in enumerate(train_loader):
        mask = data.y.squeeze() != -1

        # (batch, picks, feat)
        x = data.x[:, mask, :].to(device)
        # assuming first feature is pick_time
        pick_times = x[..., 0]
        # (batch, picks)
        y = data.y[:, mask].to(device).long()

        optimizer.zero_grad()

        # (batch, picks, out_feats)
        embeddings = model(x, pick_times)

        # Flatten batch * picks
        flat_embeddings = embeddings.view(-1, embeddings.size(-1))
        flat_labels = y.view(-1)

        indices_tuple = mining_func(flat_embeddings, flat_labels)
        loss = loss_func(flat_embeddings, flat_labels, indices_tuple)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 1000 == 0:
            num_triplets = mining_func.num_triplets if hasattr(
                mining_func, 'num_triplets') else None
            print(f"Epoch {epoch} Iteration {i}: Loss = {loss:.4f}, "
                  f"Mined triplets = {num_triplets}, "
                  f"Learning Rate = {scheduler.get_last_lr()}")


@torch.no_grad()
def test_cluster(loader, model, accuracy_calculator, device):
    model.eval()
    ari = 0

    for data in loader:
        mask = data.y.squeeze() != -1

        # (batch, picks, feature_dim)
        x = data.x[:, mask, :].to(device)
        y = data.y[:, mask].to(device).long()    # (batch, picks)

        embeddings = model(x)                 # (batch, picks, out_feats)
        flat_embeddings = embeddings.view(-1, embeddings.size(-1))
        flat_labels = y.view(-1)

        acc = accuracy_calculator.get_accuracy(flat_embeddings, flat_labels)
        ari += acc['ari']

    print(f'Test set ARI: {ari / len(loader):.4f}')
    return ari / len(loader)


def cluster_function_sklearn(embeddings, num_clusters):
    # clustering function for testing
    try:
        embeddings = embeddings.cpu().numpy()
    except AttributeError:
        pass
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(
        embeddings)
    return kmeans.labels_


class CustomAccuracyCalculator(AccuracyCalculator):
    # custom accuracy calculator
    def calculate_ari(self, query_labels, cluster_labels, **kwargs):
        try:
            query_labels = query_labels.cpu().numpy()
            cluster_labels = cluster_labels.cpu().numpy()
        except AttributeError:
            pass
        return adjusted_rand_score(query_labels, cluster_labels)

    def requires_clustering(self):
        return super().requires_clustering() + ['ari']


# %% INSTANTIATIONS
###############################################################################
distance = distances.CosineSimilarity()
reducer = reducers.AvgNonZeroReducer()
mining_func = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets='semihard')

accuracy_calculator = CustomAccuracyCalculator(
    include=('ari',), kmeans_func=cluster_function_sklearn)
model = NN(ds[0].x.shape[1], 64, 16).to(device)


# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer, step_size=1000, gamma=0.9)


# AdamW optimizer with initial LR (max LR will be set by scheduler)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

num_epochs = 10
steps_per_epoch = len(train_loader)
total_steps = num_epochs * steps_per_epoch

# OneCycleLR scheduler: adjust max_lr and pct_start as needed
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=total_steps, pct_start=0.3,
    anneal_strategy='cos', final_div_factor=30
)
# This will start at max_lr/div_factor (default
# div_factor=25 if not provided, here final_div_factor=30 for end LR)
# and peak at 1e-3 around 30% of training, then cosine anneal down to
# max_lr/final_div_factor.


# %% TRAINING
###############################################################################
for epoch in range(1, num_epochs + 1):
    loss_func = losses.TripletMarginLoss(
        margin=0.2+epoch*0.04, distance=distance, reducer=reducer)
    train(model, loss_func, mining_func, device,
          train_loader, optimizer, epoch, scheduler)
    test_cluster(test_loader, model, accuracy_calculator, device)
torch.save(model.state_dict(), 'model')

# %% VALIDATION
###############################################################################
model.load_state_dict(torch.load(
    'model_2s_5hz', weights_only=True, map_location=torch.device('cpu')))
test_cluster(test_loader, model, accuracy_calculator, device)


# %% VISUALIZATION
###############################################################################
data = next(itertools.islice(test_loader, 44, None))
x = data.x.squeeze().to(device)
embeddings = model(x)
embeddings = embeddings.cpu().squeeze().detach().numpy()
labels = data.y.squeeze().cpu().numpy()

plot_embeddings_reduced(embeddings, labels, data.catalog, method='tsne')
