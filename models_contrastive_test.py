# %%
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.loader import DataLoader
from torchvision import transforms

from create_synthetic_data import create_synthetic_data
from src.clustering.dataset import (NDArrayTransform, NDArrayTransformX,
                                    PhasePicksDataset)
from src.clustering.utils import ClusterStatistics
from src.synthetics.create_associations import inventory_to_stations

# %%
stations = inventory_to_stations('stations/station_cords_blab_VALTER.csv')
out_dir = Path('data/contrastive')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def scale_data(sample):
    scaler_std = StandardScaler()
    return torch.tensor(scaler_std.fit_transform(sample),
                        dtype=torch.float64,
                        device=device)


events_per_second = np.arange(0, 121, 5)[1:]

events = 30
n_catalogs = 100
fixed_times = False

plot = False
add_noise = True
noise_factor = 1
startdate = pd.to_datetime(datetime.now())

data_config = {}
datasets = {}
stats = {}
last_sample = {}

for j, eps in enumerate(events_per_second):

    # if fixed times is true, add some spacing at start and end
    # and therefore increase the duration
    duration = 1/eps * (events + int(fixed_times))

    data_config[j] = {
        "duration": duration,
        "events": events,
        "ticklabel": eps
    }

for key, value in data_config.items():
    event_times = np.linspace(0, value['duration'], events+2)[1:-1]
    create_synthetic_data(out_dir / Path(str(key)),
                          n_catalogs,
                          value['events'],
                          value['events'],
                          value['duration'],
                          stations,
                          add_noise=add_noise,
                          noise_factor=noise_factor,
                          event_times=event_times if fixed_times else None,
                          startdate=startdate,
                          )

    datasets[key] = PhasePicksDataset(
        root_dir=out_dir / Path(str(key)),
        stations_file='stations.csv',
        file_mask='arrivals_*.csv',
        catalog_mask='catalog_*.csv',
        transform=transforms.Compose([
            NDArrayTransformX(drop_cols=['station',],
                              cat_cols=['phase'],
                              ),
            scale_data]),
        target_transform=transforms.Compose([NDArrayTransform()]),
        catalog_transform=NDArrayTransform(),
    )

    n_feats = datasets[key][0].x.shape[1]


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
model.load_state_dict(torch.load('model', weights_only=True))


@torch.no_grad()
def test(loader):
    model.eval()
    statistics = ClusterStatistics()
    for data in loader:
        data.x.to(device)
        embeddings = model(data.x)

        # Cluster embeddings
        n_clusters = len(data.y.squeeze().unique())
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
            embeddings.cpu().squeeze().numpy())
        labels = kmeans.labels_

        statistics.add(data.y.squeeze().cpu().numpy(), labels)

    return statistics


for key, ds in datasets.items():
    loader = DataLoader(ds, batch_size=1)

    stats[key] = test(loader)

    print(f"Dataset {key}: ARI: {stats[key].ari()}, "
          f"Precision: {stats[key].precision()}, "
          f"Recall: {stats[key].recall()}, ")

# %%
