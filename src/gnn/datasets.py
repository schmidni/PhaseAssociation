import os

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import KNNGraph


def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def pre_transform(arrivals):
    positions = torch.tensor(
        arrivals[['e', 'n', 'u']].values, dtype=torch.float)

    features = []

    time = arrivals['time'] - arrivals['time'].min()
    time = torch.tensor(time.astype('int64').values, dtype=torch.float64)
    features.append(min_max_normalize(time))

    phase = torch.tensor(arrivals['phase'].replace(
        {'P': '0', 'S': '1'}).astype('int32').values, dtype=torch.int32)
    features.append(phase)

    # e = torch.tensor(arrivals['e'].values, dtype=torch.float)
    # features.append(min_max_normalize(e))
    # n = torch.tensor(arrivals['n'].values, dtype=torch.float)
    # features.append(min_max_normalize(n))
    # u = torch.tensor(arrivals['u'].values, dtype=torch.float)
    # features.append(min_max_normalize(u))

    features = torch.vstack(features).T

    data = Data()
    data.pos = positions
    data.x = features
    data.y = torch.tensor([[arrivals['event'].nunique()]], dtype=torch.float)
    data = KNNGraph(k=16)(data)
    data.validate(raise_on_error=True)

    return data


class PhaseAssociationDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=pre_transform,
                 pre_filter=None,
                 force_reload=False):
        super().__init__(root, transform, pre_transform,
                         pre_filter, force_reload=force_reload)
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
        data_list = []
        for raw_path in self.raw_paths:
            data = pd.read_csv(raw_path, parse_dates=['time'])
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
