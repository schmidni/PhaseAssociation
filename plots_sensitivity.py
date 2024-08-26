# %%
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from create_synthetic_data import create_synthetic_data
from src.clustering.dataset import (GaMMAPickFormat, GaMMAStationFormat,
                                    PhasePicksDataset)
from src.clustering.models import run_gamma
from src.clustering.utils import ClusterStatistics, plot_arrivals
from src.synthetics.create_associations import inventory_to_stations

# %%
out_dir = Path('data/times')

stations = inventory_to_stations('stations/station_cords_blab_VALTER.csv')

events = np.array([4])
n_catalogs = 10
duration = 0.1

startdate = pd.to_datetime(datetime.now())
for i, ev in enumerate(events):

    event_times = np.linspace(0, duration, ev+2)
    event_times = event_times[1:-1]
    print(event_times)
    create_synthetic_data(out_dir,
                          n_catalogs,
                          ev,
                          ev,
                          duration,
                          stations,
                          add_noise=True,
                          noise_factor=1,
                          event_times=event_times,
                          startdate=startdate,
                          fixed_mag=-2)

dataset = PhasePicksDataset(
    root_dir=out_dir,
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv',
    transform=GaMMAPickFormat(),
    station_transform=GaMMAStationFormat()
)

# %%
config = {
    "ncpu": 4,
    "dims": ['x(km)', 'y(km)', 'z(km)'],  # needs to be *(km), column names
    "use_amplitude": False,
    "vel": {"p": 5.5, "s": 2.7},
    "method": "BGMM",
    # "n_init": 3,
    "oversample_factor": 5,  # factor on the number of initial clusters
    "z(km)": (-1, 1),
    "covariance_prior": [1e-5, 1e-10],  # time, amplitude
    "bfgs_bounds": (    # bounds in km
        (-1, 1),        # x
        (-1, 1),        # y
        (-1, 1),        # depth
        (None, None),   # t
    ),
    "use_dbscan": True,
    "dbscan_eps": 0.01,  # seconds
    "dbscan_min_samples": 10,
    "min_picks_per_eq": 8,
    "max_sigma11": 0.01,
    "max_sigma22": 2,
    # "max_sigma12": 2.0
}

statistics = ClusterStatistics()

for sample in tqdm(dataset):

    cat_gmma, labels_pred = run_gamma(sample.x, dataset.stations, config)

    statistics.add(sample.y.to_numpy(),
                   labels_pred
                   )

    associations = sample.x.copy().join(dataset.stations.set_index('id'), on='id')
    associations['dx'] = PhasePicksDataset.get_distance(
        associations, ['x(km)', 'y(km)', 'z(km)'])*1000
    associations['time'] = pd.to_datetime(
        associations['timestamp'], unit='ns').values.astype(int)

    associations['time'] = (pd.to_datetime(
        associations['timestamp'], unit='ns') - startdate).values.astype(int)/1e6
    sample.catalog['time'] = (pd.to_datetime(
        sample.catalog['time'], unit='ns') - startdate).values.astype(int)/1e6
    cat_gmma['time'] = (pd.to_datetime(
        cat_gmma['time'], unit='ns') - startdate).values.astype(int)/1e6

    plot_arrivals(associations[['dx', 'time']],
                  sample.catalog[['dx', 'time']],
                  cat_gmma[['dx', 'time']],
                  sample.y.to_numpy(),
                  labels_pred)


print(f"GaMMA ARI: {statistics.ari()}, Accuray: {statistics.accuracy()}, "
      f"Precision: {statistics.precision()}, Recall: {statistics.recall()}")
# print(f"GaMMA event precision: {statistics.event_precision()}, "
#       f"GaMMA event recall: {statistics.event_recall()}")
# %%
