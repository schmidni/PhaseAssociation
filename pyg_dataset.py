# %%
import os

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import KNNGraph

# %%
stations = pd.read_csv('stations.csv')
arrivals = pd.read_csv('arrivals.csv', parse_dates=['time'])

# %%
arrivals.head()
positions = torch.tensor(arrivals[['e', 'n', 'u']].values, dtype=torch.float)

# %%
time = arrivals['time'] - arrivals['time'].min()
time = torch.tensor(time.astype('int64').values, dtype=torch.float64)

# %%
phase = torch.tensor(arrivals['phase'].replace(
    {'P': '0', 'S': '1'}).astype('int32').values, dtype=torch.int32)

features = torch.vstack((time, phase)).T


# %%
data = Data()
data.pos = positions
data.x = features
data.y = arrivals['event'].nunique()
data = KNNGraph(k=3)(data)
data.validate(raise_on_error=True)


def pre_transform(arrivals):
    positions = torch.tensor(
        arrivals[['e', 'n', 'u']].values, dtype=torch.float)

    time = arrivals['time'] - arrivals['time'].min()
    time = torch.tensor(time.astype('int64').values, dtype=torch.float64)

    phase = torch.tensor(arrivals['phase'].replace(
        {'P': '0', 'S': '1'}).astype('int32').values, dtype=torch.int32)

    features = torch.vstack((time, phase)).T

    data = Data()
    data.pos = positions
    data.x = features
    data.y = arrivals['event'].nunique()
    data = KNNGraph(k=3)(data)
    data.validate(raise_on_error=True)

    return data

# %%


class PhaseAssociationDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        all_files = os.listdir(self.raw_dir)
        all_files = [f for f in all_files if f.startswith('arrivals')]
        all_files = [f for f in all_files if f.endswith('.csv')]

        return all_files

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for raw_path in self.raw_paths:
            data = pd.read_csv(raw_path, parse_dates=['time'])
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


# %%
dataset = PhaseAssociationDataset('data', pre_transform=pre_transform)
