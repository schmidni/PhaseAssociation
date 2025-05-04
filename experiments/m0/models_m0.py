# %% Imports and Configuration
import itertools
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset import (PhasePicksDataset, SeisBenchPickFormat,
                         SeisBenchStationFormat)
from src.runners import run_gamma

config = {
    "ncpu": multiprocessing.cpu_count(),
    "dims": ['x(km)', 'y(km)', 'z(km)'],  # needs to be *(km), column names
    "use_amplitude": False,
    "vel": {"p": 5.4, "s": 3.1},
    "method": "BGMM",
    "oversample_factor": 10,  # factor on the number of initial clusters
    "z(km)": (-0.3, 0.3),
    "covariance_prior": [1e-5, 1e-2],  # time, amplitude
    "bfgs_bounds": (    # bounds in km
        (-0.3, 0.3),        # x
        (-0.3, 0.3),        # y
        (-0.3, 0.3),        # depth
        (None, None),   # t
    ),
    "use_dbscan": True,
    "dbscan_eps": 0.007,  # seconds
    "dbscan_min_samples": 5,

    "min_picks_per_eq": 8,
    "max_sigma11": 0.01,
    "max_sigma22": 2.0,
    # "max_sigma12": 2.0
}

# %% Run GaMMA

ds = PhasePicksDataset(
    root_dir='data/m0',
    stations_file='stations.csv',
    file_mask='m0_10s_formatted_high.csv',
    catalog_mask='catalog.csv',
    transform=SeisBenchPickFormat(),
    station_transform=SeisBenchStationFormat()
)
sample = ds[0]


sizes = sample.x.groupby('id').size()
diff = sample.x['timestamp'].max() - sample.x['timestamp'].min()
diff = diff.microseconds/1e6
plt.bar(sizes.index, 1/(diff/sizes.values))
plt.xticks(rotation=90)

# %%
cat_gmma, labels_pred = run_gamma(sample.x, ds.stations, config)

# %% Plot Results
associations = sample.x.copy().join(ds.stations.set_index('id'), on='id')
associations['dx'] = PhasePicksDataset.get_distance(
    associations, ['x(km)', 'y(km)', 'z(km)'])*1000
associations['time'] = pd.to_datetime(
    associations['timestamp'], unit='ns').values.astype(int)


color_iter = itertools.cycle(
    ["navy", "c", "cornflowerblue", "gold", "orange", "green",
     "lime", "red", "purple", "blue", "pink", "brown", "gray",
     "magenta", "cyan", "olive", "maroon", "darkslategray", "darkkhaki"])


def plot_arrivals(arrivals, cat, cat_pred, labels_pred):
    fig, ax = plt.subplots(1, sharex=True, figsize=(14, 10))

    for idx in range(len(np.unique(labels_pred))):
        ax.scatter(arrivals.loc[labels_pred == idx, 'time'],
                   arrivals.loc[labels_pred == idx, 'dx'],
                   color=color_iter.__next__(), s=100
                   )

    ax.scatter(arrivals.loc[labels_pred == -1, 'time'],
               arrivals.loc[labels_pred == -1, 'dx'],
               color='black', s=100)
    ax.scatter(cat_pred['time'][cat_pred['dx'] < 350],
               cat_pred['dx'][cat_pred['dx'] < 350],
               color='darkorange', marker='x', s=80)

    # truth
    ax.scatter(cat['time'], cat['dx'],
               color='red', marker='x', s=120)

    return (fig, ax)


plot_arrivals(associations[['dx', 'time']],
              sample.catalog[['dx', 'time']],
              cat_gmma[['dx', 'time']],
              labels_pred)

# %%


def plot_all(arrivals):
    fig, ax = plt.subplots(1, sharex=True, figsize=(14, 10))

    ax.scatter(arrivals['time'], arrivals['dx'], color='black', s=100)

    return (fig, ax)


plot_all(associations[['dx', 'time']])

# # %%
