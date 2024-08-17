import glob
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

Picks = namedtuple('Picks', ['x', 'y', 'catalog'])


class PhasePicksDataset(Dataset):
    def __init__(self,
                 root_dir,
                 stations_file,
                 file_mask='arrivals_*.csv',
                 catalog_mask=None,
                 transform=None,
                 target_transform=None,
                 station_transform=None,
                 catalog_transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.station_transform = station_transform
        self.catalog_transform = catalog_transform
        self.catalog_files = None

        self._stations = pd.read_csv(
            os.path.join(self.root_dir, stations_file))
        self.files = sorted(glob.glob(os.path.join(self.root_dir, file_mask)))

        if catalog_mask:
            self.catalog_files = sorted(glob.glob(
                os.path.join(self.root_dir, catalog_mask)))

    # getter for self._stations
    @property
    def stations(self):
        if self.station_transform:
            return self.station_transform(self._stations)
        return self._stations

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]
        sample = pd.read_csv(file)
        sample = self.convert_datetime(sample)
        sample = sample.join(self._stations.set_index('id'), on='station')
        sample = sample.drop(columns=['longitude', 'latitude', 'altitude'])

        if 'event' in sample.columns:
            events = sample['event']
            sample = sample.drop(columns=['event'])
        else:
            events = None

        if self.catalog_files:
            catalog = pd.read_csv(self.catalog_files[idx], index_col=0)
            catalog = self.convert_datetime(catalog)
            catalog['dx'] = self.get_distance(catalog)
        else:
            catalog = None

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            events = self.target_transform(events)

        if self.catalog_transform:
            catalog = self.catalog_transform(catalog)

        return Picks(x=sample, y=events, catalog=catalog)

    @staticmethod
    def convert_datetime(df):
        df['time'] = pd.to_datetime(df['time'], unit='ns').values.astype(int)
        return df

    @staticmethod
    def get_distance(df, cols=['e', 'n', 'u'], reference=(0, 0, 0)):
        return np.sqrt((df[cols[0]] - reference[0])**2 +
                       (df[cols[1]] - reference[1])**2 +
                       (df[cols[2]] - reference[2])**2)


class GaMMAPickFormat:
    def __init__(self):
        pass

    def __call__(self, sample):
        sample = sample.rename(columns={'station': 'id',
                                        'phase': 'type',
                                        'amplitude': 'amp',
                                        'time': 'timestamp'})
        sample = sample[['id', 'timestamp', 'type', 'amp']]
        sample['timestamp'] = pd.to_datetime(sample['timestamp'], unit='ns')
        sample['prob'] = 1
        return sample


class GaMMAStationFormat:
    def __init__(self):
        pass

    def __call__(self, stations):
        stn = stations.copy()
        stn[['e', 'n', 'u']] = stn[['e', 'n', 'u']] / 1000
        stn['u'] = stn['u'] * -1
        stn = stn.rename(
            columns={'e': 'x(km)', 'n': 'y(km)', 'u': 'z(km)'})
        stn = stn[['id', 'x(km)', 'y(km)', 'z(km)']]
        return stn


class NDArrayTransformX:
    def __init__(self,
                 device=None,
                 drop_cols=['station'],
                 cat_cols=['phase']):
        self.drop_cols = drop_cols
        self.cat_cols = cat_cols
        self.device = device

    def __call__(self, sample):
        sample = sample.drop(columns=self.drop_cols)
        for col in self.cat_cols:
            sample[col] = sample[col].astype('category').cat.codes
        return torch.tensor(sample.to_numpy(),
                            dtype=torch.float64,
                            device=self.device)


class NDArrayTransform:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, sample):
        return torch.tensor(sample.to_numpy(),
                            dtype=torch.float64,
                            device=self.device)
