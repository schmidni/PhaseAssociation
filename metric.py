# %%
import itertools

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src import losses
from src.dataset import (ColumnsTransform, PhasePicksDataset, ReduceDatetime,
                         ScaleTransform, collate_fn, collate_fn_validate)
from src.metrics import ClusterStatistics
from src.models import PositionalEncoding
from src.plotting.embeddings import plot_embeddings_reduced

# %% Load data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running models on {device}.')

ds = PhasePicksDataset(
    root_dir='data/reference/2s_5hz',
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

generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset, validate_dataset = random_split(
    ds, [0.3, 0.1, 0.6], generator=generator)

train_batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                          num_workers=0, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
validate_loader = DataLoader(validate_dataset, batch_size=1,
                             num_workers=0, collate_fn=collate_fn_validate)

print('Data Loader Prepared')

# %%


class PhasePickTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_stations: int,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 output_dim: int = 16,
                 max_picks: int = 1000,
                 encoding_type: str = 'sinusoidal'):
        super(PhasePickTransformer, self).__init__()
        self.embed_dim = embed_dim  # Transformer embedding dimension
        self.output_dim = output_dim  # Final embedding dimension

        # Project input features to Transformer embedding dim
        self.feature_proj = nn.Linear(input_dim-3, embed_dim)

        # Learnable station embedding
        self.station_embedding = nn.Embedding(num_stations, embed_dim)

        # Learnable positional encoding
        self.positional_encoding = \
            PositionalEncoding(embed_dim,
                               max_len=max_picks,
                               encoding_type=encoding_type)

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

        # MLP projection for coordinates
        self.coord_proj = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
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
        x = self.feature_proj(pick_features[:, :, 3:6])

        # Add coordinate embeddings
        # (batch, N, embed_dim)
        coord_embs = self.coord_proj(pick_features[:, :, 0:3])
        x = x + coord_embs

        # Lookup and add station embeddings
        # (batch, N, embed_dim)
        station_embs = self.station_embedding(station_ids)
        x = x + station_embs

        # Add positional encodings
        # (1, N, embed_dim)
        x = self.positional_encoding(x)

        # (batch, N, embed_dim)
        x = self.transformer(x)

        # Project to output embedding
        # (batch, N, output_dim)
        x = self.output_proj(x)

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
    running_loss = 0.0
    running_triplets = 0

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

        num_triplets = getattr(mining_func, 'num_triplets', 0)
        running_loss += loss.item()
        running_triplets += num_triplets

        if i == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'[Epoch {epoch} | Iter {i} | LR: {current_lr:.3e}]')
        if i % 1000 == 0 and i != 0:
            avg_loss = running_loss / 1000
            current_lr = scheduler.get_last_lr()[0]

            print(f"[Epoch {epoch} | Iter {i}] "
                  f"Avg Loss: {avg_loss:.6f} | "
                  f"Total Triplets: {running_triplets:.3e} | "
                  f"LR: {current_lr:.3e}")

            running_loss = 0.0
            running_triplets = 0


@torch.no_grad()
def test_cluster(loader, model, device):
    model.eval()
    ac = ClusterStatistics()

    for x, y, st in loader:
        x = x.to(device)
        y = y.to(device)
        st = st.to(device)

        embeddings = model(x, st)
        flat_embeddings = embeddings.view(-1, embeddings.size(-1))
        flat_labels = y.view(-1)

        # Mask out padding (-2)
        mask = (flat_labels != -2)

        ac.add_embedding(flat_embeddings[mask],
                         flat_labels[mask])

    print(f'Test set ARI: {ac.ari():.4f} | '
          f'Precision: {ac.precision():.4f} | '
          f'Recall: {ac.recall():.4f} | '
          f'Accuracy: {ac.accuracy():.4f}')


# %% INSTANTIATIONS
###############################################################################
# ## Metric Learning

num_epochs = 3
total_steps = int(num_epochs * len(train_loader) / train_batch_size)

# ## Model
model = PhasePickTransformer(
    input_dim=ds[0].x.shape[1]-1, num_stations=len(ds.stations), embed_dim=128,
    num_heads=4, num_layers=2, max_picks=1000).to(device)

# ## Scheduler and Optimizer

optimizer = torch.optim.AdamW(model.parameters())  # LR set by scheduler

# Start at max_lr/div_factor, peak at 30% of training, cosine
# anneal down to max_lr/final_div_factor.
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=total_steps, pct_start=0.3,
    anneal_strategy='cos', final_div_factor=30
)

metric_loss = losses.metric_loss_multi_similarity

# %% TRAINING
###############################################################################
for epoch in range(1, num_epochs + 1):
    mining_func, loss_func = metric_loss(epoch, num_epochs)

    train(model, loss_func, mining_func, device,
          train_loader, optimizer, epoch, scheduler)
    test_cluster(test_loader, model, device)

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
test_cluster(test_loader, model, device)
# %%
