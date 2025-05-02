# %% Imports and Configuration

import tqdm

from src.dataset import GaMMAPickFormat, GaMMAStationFormat, PhasePicksDataset
from src.metrics import ClusterStatistics
from src.runners import run_harpa

config = {
    "x(km)": (-1, 1),
    "y(km)": (-1, 1),
    "z(km)": (-1, 1),
    "vel": {"P": 5.4, "S": 3.1},
    "P_phase": True,
    "S_phase": True,
    "min_peak_pre_event": 5,
    "min_peak_pre_event_s": 0,
    "min_peak_pre_event_p": 0,
}

# %% Run GaMMA
statistics = ClusterStatistics()


ds = PhasePicksDataset(
    root_dir='../../data/test',
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv',
    transform=GaMMAPickFormat(),
    station_transform=GaMMAStationFormat()
)


for sample in tqdm.tqdm([ds[0]]):
    cat, labels_pred = run_harpa(sample.x, ds.stations, config)

    statistics.add(sample.y.to_numpy(),
                   labels_pred)

    print(f"HARPA ARI: {statistics.ari()}, Accuray: {statistics.accuracy()}, ")

print(f"HARPA ARI: {statistics.ari()}, Accuray: {statistics.accuracy()}, "
      f"Precision: {statistics.precision()}, Recall: {statistics.recall()}")
