import os

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import KNNGraph


def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def transform_simple_knn(arrivals):
    """
    Uses Arrivals as nodes, and connects them to the k nearest neightbors,
    which usually just are the arrivals of the same station.
    """
    arrivals['time_diff'] = arrivals['time'] - arrivals['time'].min()
    arrivals['time_diff'] = arrivals['time_diff'].astype('int64')

    time = torch.tensor(arrivals['time_diff'].values, dtype=torch.float64)

    positions = torch.tensor(
        arrivals[['e', 'n', 'u', 'time_diff']].values, dtype=torch.float)

    features = []

    features.append(min_max_normalize(time))

    phase = torch.tensor(arrivals['phase'].replace(
        {'P': '0', 'S': '1'}).astype('int32').values, dtype=torch.int32)
    features.append(phase)

    e = torch.tensor(arrivals['e'].values, dtype=torch.float)
    features.append(min_max_normalize(e))
    n = torch.tensor(arrivals['n'].values, dtype=torch.float)
    features.append(min_max_normalize(n))
    u = torch.tensor(arrivals['u'].values, dtype=torch.float)
    features.append(min_max_normalize(u))
    a = torch.tensor(arrivals['amplitude'].values, dtype=torch.float)
    features.append(min_max_normalize(a))

    features = torch.vstack(features).T

    data = Data()
    data.pos = positions
    data.x = features
    data.y = torch.tensor([[arrivals['event'].nunique()]], dtype=torch.float)
    data = KNNGraph(k=32)(data)

    data.validate(raise_on_error=True)

    return data


def find_matching_indices(A, S):
    # Get the unique values in A and their corresponding indices
    unique_values, inverse_indices = np.unique(A, return_inverse=True)

    # Create a dictionary to map each value in A to its indices
    value_to_indices = {val: np.where(A == val)[0] for val in unique_values}

    # Initialize an empty list to collect the results
    result = []

    # Iterate over each element in A with its index
    for idx, a_o in enumerate(A):
        # Get the corresponding row from S
        row_indices = S[a_o]

        # Iterate over each element in the row of S
        for s_value in row_indices:
            # Find all indices in A where the element matches s_value
            matching_indices = value_to_indices.get(s_value, [])

            # For each matching index, append the result
            for match_idx in matching_indices:
                result.append((idx, match_idx))

    # Convert result list to a numpy array with shape (2, j)
    if result:
        R = np.array(result).T
    else:
        R = np.empty((2, 0), dtype=int)

    return R


def transform_knn_stations(arrivals, stations, nearest_stations):
    """
    Uses Arrivals as nodes, and builds the graph by connecting each arrival
    not only to other arrivals of the same station, but also the arrivals
    of the n nearest stations.
    """
    arrivals['time'] = arrivals['time'] - arrivals['time'].min()
    arrivals['time'] = arrivals['time'].astype('long')

    # add column 'station_idx' with the index of the station
    # in the stations dataframe
    arrivals['station_idx'] = arrivals['station'].apply(
        lambda x: stations[stations['id'] == x].index[0])
    arrivals

    features = []

    time_ = torch.tensor(arrivals['time'].values, dtype=torch.long)
    features.append(min_max_normalize(time_))

    phase = torch.tensor(arrivals['phase'].replace(
        {'P': '0', 'S': '1'}).astype('int32').values, dtype=torch.int32)
    features.append(phase)
    e = torch.tensor(arrivals['e'].values, dtype=torch.float)
    features.append(min_max_normalize(e))
    n = torch.tensor(arrivals['n'].values, dtype=torch.float)
    features.append(min_max_normalize(n))
    u = torch.tensor(arrivals['u'].values, dtype=torch.float)
    features.append(min_max_normalize(u))
    a = torch.tensor(arrivals['amplitude'].values, dtype=torch.float)
    features.append(min_max_normalize(a))
    features = torch.vstack(features).T

    data = Data()
    data.x = features
    # build COO representation of the graph edges
    data.edge_index = torch.tensor(find_matching_indices(
        arrivals['station_idx'].values, nearest_stations), dtype=torch.long)

    # graph label: number of unique events
    data.y = torch.tensor([[arrivals['event'].nunique()]], dtype=torch.float)

    # data.validate(raise_on_error=True)
    return data


class PhaseAssociationDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=transform_knn_stations,
                 pre_filter=None,
                 force_reload=False):

        self.stations = pd.read_csv(os.path.join(root, 'raw', 'stations.csv'))
        X = self.stations[['e', 'n', 'u']].values
        nbrs = NearestNeighbors(n_neighbors=6).fit(X)
        _, self.nearest_stations = nbrs.kneighbors(X)

        row_indices = np.arange(self.nearest_stations.shape[0]).reshape(-1, 1)
        self.nearest_stations = np.hstack((self.nearest_stations, row_indices))

        super().__init__(root, transform, pre_transform,
                         pre_filter, force_reload=force_reload)

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
            # replace generation using tqdm loading bar
            new_list = []
            for data in tqdm.tqdm(data_list):
                new_list.append(
                    self.pre_transform(data,
                                       self.stations,
                                       self.nearest_stations))

        self.save(new_list, self.processed_paths[0])
