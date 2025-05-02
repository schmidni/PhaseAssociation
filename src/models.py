# %% DEFINITIONS
###############################################################################


import math

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000, encoding_type="learnable"):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoding_type = encoding_type

        self.cached_encoding = None
        self.cached_seq_len = 0

        if encoding_type == "learnable":
            self.positional_encoding = nn.Parameter(
                torch.zeros(max_len, embed_dim))
            nn.init.xavier_uniform_(self.positional_encoding)

    def forward(self, x):
        seq_len, device = x.size(1), x.device

        if self.encoding_type == "learnable":
            pos_enc = self.positional_encoding[:seq_len].unsqueeze(0)
        else:
            if (self.cached_encoding is None
                or self.cached_encoding.device != device
                    or seq_len > self.cached_seq_len):
                pos_enc = self._generate_encoding(seq_len, device)
                self.cached_encoding = pos_enc
                self.cached_seq_len = seq_len
            else:
                pos_enc = self.cached_encoding[:, :seq_len]

        return x + pos_enc

    def _generate_encoding(self, seq_len, device):
        if self.encoding_type == "sinusoidal":
            position = torch.arange(seq_len, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.embed_dim, 2, device=device) *
                -(math.log(10000.0) / self.embed_dim))
            pe = torch.zeros(seq_len, self.embed_dim, device=device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0)
        elif self.encoding_type == "linear":
            pe = torch.linspace(0, 1, steps=seq_len,
                                device=device).unsqueeze(1)
            pe = pe.expand(seq_len, self.embed_dim)
            return pe.unsqueeze(0)
        else:
            raise ValueError(
                f"Unknown positional encoding type: {self.encoding_type}")


class PhasePickMLP(torch.nn.Module):
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


class PrototypicalLoss(torch.nn.Module):
    def __init__(self, distance='euclidean'):
        super().__init__()
        assert distance in ['euclidean', 'cosine']
        self.distance = distance

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor,
                *args, **kwargs):
        """
        embeddings: (N, D)
        labels: (N,)
        """
        device = embeddings.device
        N, D = embeddings.shape
        classes, labels_idx = torch.unique(labels, return_inverse=True)
        C = classes.size(0)

        # Compute class counts
        counts = torch.bincount(
            labels_idx, minlength=C).float().unsqueeze(1)  # (C, 1)

        # Sum embeddings per class
        prototypes = torch.zeros(C, D, device=device).scatter_add_(
            0, labels_idx.unsqueeze(1).expand(-1, D), embeddings)
        prototypes = prototypes / counts  # (C, D)

        if self.distance == 'euclidean':
            dists = torch.cdist(embeddings, prototypes, p=2)
        else:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            prototypes = F.normalize(prototypes, p=2, dim=1)
            dists = 1 - embeddings @ prototypes.T

        # Use index mapping directly
        targets = labels_idx

        return F.cross_entropy(-dists, targets)


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
