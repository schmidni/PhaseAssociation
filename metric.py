# %%
import itertools

import torch
from pytorch_metric_learning import distances, losses, miners, reducers
from pytorch_metric_learning.utils.accuracy_calculator import \
    AccuracyCalculator
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.dataset import (ColumnsTransform, PhasePicksDataset, ReduceDatetime,
                         ScaleTransform, collate_fn, collate_fn_validate)
from src.models import PositionalEncoding
from src.plotting.embeddings import plot_embeddings_reduced

# %% Load data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running models on {device}.')

ds = PhasePicksDataset(
    root_dir='data/reference/low_freq',
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv',
    transform=transforms.Compose([
        ReduceDatetime(),
        ColumnsTransform(drop_cols=['id'],
                         cat_cols=['type']),
        ScaleTransform(columns=['timestamp', 'amp', 'e', 'n', 'u']),
    ])
)

generator = torch.Generator()  # .manual_seed(42)
train_dataset, test_dataset, validate_dataset = random_split(
    ds, [0.02, 0.01, 0.97], generator=generator)

test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
train_loader = DataLoader(train_dataset, batch_size=1,
                          num_workers=0, shuffle=True, collate_fn=collate_fn)
validate_loader = DataLoader(validate_dataset, batch_size=1,
                             num_workers=0, collate_fn=collate_fn_validate)

print('Data Loader Prepared')


class PhasePickTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_stations: int,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 output_dim: int = 16,
                 max_picks: int = 100):
        super(PhasePickTransformer, self).__init__()
        self.embed_dim = embed_dim  # Transformer embedding dimension
        self.output_dim = output_dim  # Final embedding dimension

        # Project input features to Transformer embedding dim
        self.feature_proj = nn.Linear(input_dim, embed_dim)

        # Learnable station embedding
        self.station_embedding = nn.Embedding(num_stations, embed_dim)

        # Learnable positional encoding
        self.positional_encoding = \
            PositionalEncoding(embed_dim,
                               max_len=max_picks,
                               encoding_type='sinusoidal')

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=256,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)

        # MLP projection head to output dimension
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )

        # L2 normalization
        self.normalize = nn.functional.normalize

    def forward(self,
                pick_features: torch.tensor,
                station_ids: torch.tensor) -> torch.tensor:
        """
        pick_features: Tensor (batch, N, feature_dim)
        pick_times: Tensor (batch, N) — used for sorting
        """
        # Project features
        # (batch, N, embed_dim)
        x = self.feature_proj(pick_features)

        # Lookup and add station embeddings
        # (batch, N, embed_dim)
        station_embs = self.station_embedding(station_ids)
        x = x + station_embs

        # Add positional encodings
        # (1, N, embed_dim)
        x = self.positional_encoding(x)

        x = self.transformer(x)  # (batch, N, embed_dim)

        # Project to output embedding
        x = self.output_proj(x)  # (batch, N, output_dim)

        # Normalize
        emb = self.normalize(x, p=2, dim=-1)
        return emb


def remap_noise_labels(labels):
    labels = labels.clone()  # avoid modifying in-place
    max_label = labels[labels != -
                       1].max().item() if (labels != -1).any() else -1
    next_label = max_label + 1

    noise_mask = (labels == -1)
    num_noise = noise_mask.sum().item()

    # Assign each -1 a unique ID
    labels[noise_mask] = torch.arange(
        next_label, next_label + num_noise, device=labels.device)

    return labels


def train(model, loss_func, mining_func, device,
          train_loader, optimizer, epoch,
          scheduler: torch.optim.lr_scheduler.StepLR):
    model.train()
    for i, (x, y, st) in enumerate(train_loader):
        # (batch, picks, feat)
        x = x.to(device)
        # (batch, picks)
        y = y.to(device)
        # (batch, picks)
        st = st.to(device)

        optimizer.zero_grad()

        # (batch, picks, out_feats)
        embeddings = model(x, st)

        # Flatten batch * picks
        flat_embeddings = embeddings.view(-1, embeddings.size(-1))
        flat_labels = y.view(-1)

        # Mask out padding
        valid_mask = flat_labels != -2
        flat_embeddings = flat_embeddings[valid_mask]
        flat_labels = flat_labels[valid_mask]

        # Remap noise labels to unique IDs
        remapped_labels = remap_noise_labels(flat_labels)

        indices_tuple = mining_func(flat_embeddings, remapped_labels)
        loss = loss_func(flat_embeddings, remapped_labels, indices_tuple)
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

    for x, y, st in loader:
        x = x.to(device)
        y = y.to(device)
        st = st.to(device)

        embeddings = model(x, st)
        flat_embeddings = embeddings.view(-1, embeddings.size(-1))
        flat_labels = y.view(-1)

        # Mask out padding (-2) and noise (-1)
        valid_mask = (flat_labels != -2) & (flat_labels != -1)
        flat_embeddings = flat_embeddings[valid_mask]
        flat_labels = flat_labels[valid_mask]

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
# ## Metric Learning
distance = distances.CosineSimilarity()
reducer = reducers.AvgNonZeroReducer()

# ## Testing Performance
accuracy_calculator = CustomAccuracyCalculator(
    include=('ari',), kmeans_func=cluster_function_sklearn)

# ## Model
# model = NN(ds[0].x.shape[1], 64, 16).to(device)
model = PhasePickTransformer(
    input_dim=ds[0].x.shape[1]-1, num_stations=len(ds.stations), embed_dim=128,
    num_heads=4, num_layers=2, max_picks=1000).to(device)

# ## Epochs
num_epochs = 1
steps_per_epoch = len(train_loader)
total_steps = num_epochs * steps_per_epoch


# ## Scheduler and Optimizer
# AdamW optimizer with initial LR (max LR will be set by scheduler)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# This will start at max_lr/div_factor (default
# div_factor=25 if not provided, here final_div_factor=30 for end LR)
# and peak at 1e-3 around 30% of training, then cosine anneal down to
# max_lr/final_div_factor.
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=total_steps, pct_start=0.3,
    anneal_strategy='cos', final_div_factor=30
)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer, step_size=total_steps/20, gamma=0.9)


# %% TRAINING
###############################################################################
for epoch in range(1, num_epochs + 1):
    margin = 0.2 + (epoch * 0.4/num_epochs)

    mining_func = miners.TripletMarginMiner(
        margin=margin, distance=distance, type_of_triplets='semihard')
    loss_func = losses.TripletMarginLoss(
        margin=margin, distance=distance, reducer=reducer)

    train(model, loss_func, mining_func, device,
          train_loader, optimizer, epoch, scheduler)
    test_cluster(test_loader, model, accuracy_calculator, device)

    torch.save(model.state_dict(), 'model')


# %% VALIDATION
###############################################################################
# model.load_state_dict(torch.load(
#     'model_2s_5hz', weights_only=True, map_location=torch.device('cpu')))
# test_cluster(test_loader, model, accuracy_calculator, device)


# %% VISUALIZATION
###############################################################################
x, y, st, cat = next(itertools.islice(validate_loader, 44, None))
x = x.to(device)
st = st.to(device)
embeddings = model(x, st)
embeddings = embeddings.cpu().squeeze().detach().numpy()
labels = y.squeeze().cpu().numpy()

plot_embeddings_reduced(embeddings, labels, cat, method='tsne')

# %%
