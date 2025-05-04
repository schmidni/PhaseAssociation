# %% Imports and Configuration
import itertools

import tqdm

from src.dataset import (PhasePicksDataset, SeisBenchPickFormat,
                         SeisBenchStationFormat)
from src.metrics import ClusterStatistics
from src.runners import run_harpa

config = {
    'x(km)': (-0.5, 0.5),
    'y(km)': (-0.5, 0.5),
    'z(km)': (-0.5, 0.5),
    'vel': {'P': 5.4, 'S': 3.1},
    'P_phase': True,
    'S_phase': True,
    'min_peak_pre_event': 4,
    'min_peak_pre_event_s': 0,
    'min_peak_pre_event_p': 0,
    # 'dbscan_eps': 0.1,
    # 'dbscan_min_samples': 4,
    # 'lr': 0.01,
    'lr_decay': 0.05,
    # 'wasserstein_p': 2,
    'max_time_residual':  0.01,
    'epochs_before_decay': 10000,
    'epochs_after_decay': 10000,
}

statistics = ClusterStatistics()


ds = PhasePicksDataset(
    root_dir='../../data/test_2',
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv',
    transform=SeisBenchPickFormat(),
    station_transform=SeisBenchStationFormat()
)


for sample in tqdm.tqdm(itertools.islice(ds, 0, 2)):
    cat, labels_pred = run_harpa(sample.x, ds.stations, config)
    y = sample.y.where(sample.y.map(sample.y.value_counts())
                       > config['min_peak_pre_event'], -1)
    statistics.add(y.to_numpy(), labels_pred)

print(f'HARPA ARI: {statistics.ari()}, Accuray: {statistics.accuracy()}, '
      f'Precision: {statistics.precision()}, Recall: {statistics.recall()}')
