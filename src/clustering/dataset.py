import glob
import os
from collections import namedtuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

Picks = namedtuple('Picks', ['x', 'y', 'catalog'])


def parse_synthetic_picks(picks: pd.DataFrame, stations: pd.DataFrame):

    picks['time'] = pd.to_datetime(picks['time'], unit='ns').values.astype(int)
    picks = picks.join(stations.set_index('id'), on='station')
    picks = picks.drop(columns=['longitude', 'latitude', 'altitude'])
    picks = picks.sort_values(by='time')

    station_index_mapping = pd.Series(stations.index, index=stations['id'])
    picks['station_id'] = picks['station'].map(station_index_mapping)

    if 'event' in picks.columns:
        labels = picks['event']
        picks = picks.drop(columns=['event'])
    else:
        labels = None

    return picks, labels


class PhasePicksDataset(Dataset):
    def __init__(self,
                 root_dir,
                 stations_file,
                 file_mask,
                 input_fn=parse_synthetic_picks,
                 catalog_mask=None,
                 transform=None,
                 target_transform=None,
                 station_transform=None,
                 catalog_transform=None):

        self.root_dir = root_dir
        self._stations = pd.read_csv(
            os.path.join(self.root_dir, stations_file))
        self.files = sorted(glob.glob(os.path.join(self.root_dir, file_mask)))

        self.input_fn = input_fn

        self.catalog_files = None
        if catalog_mask:
            self.catalog_files = sorted(glob.glob(
                os.path.join(self.root_dir, catalog_mask)))

        self.transform = transform
        self.target_transform = target_transform
        self.station_transform = station_transform
        self.catalog_transform = catalog_transform

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

        samples = pd.read_csv(file)
        samples, labels = self.input_fn(samples, self.stations)

        catalog = None
        if self.catalog_files:
            catalog = pd.read_csv(self.catalog_files[idx], index_col=0)
            catalog = self.convert_datetime(catalog)

        if self.transform:
            samples = self.transform(samples)

        if self.target_transform:
            labels = self.target_transform(labels)

        if self.catalog_transform:
            catalog = self.catalog_transform(catalog)

        return Picks(x=samples, y=labels, catalog=catalog)

    @staticmethod
    def convert_datetime(df):
        df['time'] = pd.to_datetime(df['time'], unit='ns').values.astype(int)
        return df


class GaMMAPickFormat:
    def __init__(self):
        pass

    def __call__(self, sample):
        sample = sample.rename(columns={'station': 'id',
                                        'phase': 'type',
                                        'amplitude': 'amp',
                                        'time': 'timestamp'})
        sample = sample[sample.columns.intersection(
            ['id', 'timestamp', 'type', 'amp'])]
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


class ColumnsTransform:
    def __init__(self, drop_cols=[], cat_cols=[]):
        self.drop_cols = drop_cols
        self.cat_cols = cat_cols

    def __call__(self, sample):
        for col in self.drop_cols:
            sample = sample.drop(columns=col)
        for col in self.cat_cols:
            sample[col] = sample[col].astype('category').cat.codes
        return sample


class ReduceDatetime:
    def __init__(self, column='time'):
        self.column = column

    def __call__(self, sample):
        sample[self.column] = sample[self.column] - \
            pd.to_datetime(sample[self.column].min(), unit='ns').value
        return sample


class SortDatetime:
    def __init__(self):
        pass

    def __call__(self, sample: pd.DataFrame, label: pd.Series | None = None):
        print(sample)
        # sort sample and label by time
        sample = sample.sort_values(by='time')
        label = label.reindex(sample.index)
        print(sample)
        return (sample, label) if label is not None else sample


class ScaleTransform:
    def __init__(self, columns=[]):
        self.columns = columns
        self.scaler = MinMaxScaler()

    def __call__(self, sample: pd.DataFrame) -> pd.DataFrame:
        sample[self.columns] = self.scaler.fit_transform(sample[self.columns])
        return sample
