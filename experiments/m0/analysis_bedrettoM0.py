# %% Imports and Configuration
import glob
import itertools
import multiprocessing
import os

import pandas as pd
from matplotlib import pyplot as plt

from src.dataset import PhasePicksDataset, SeisBenchStationFormat
from src.runners import run_gamma

# %% preprocess
catalog_files = sorted(
    glob.glob(os.path.join('data/bedretto', '*Ss0.30.csv')))

for file in catalog_files:
    data = pd.read_csv(file)
    data = data.rename(columns={'trace_id': 'station',
                                'onset_time': 'time',
                                })
    data['station'] = data['station'].str[3:8]
    data.to_csv(file.replace('.csv', '_processed.csv'))


# %%
config = {
    "ncpu": multiprocessing.cpu_count()-1,
    "dims": ['x(km)', 'y(km)', 'z(km)'],  # needs to be *(km), column names
    "use_amplitude": False,
    "vel": {"p": 5.4, "s": 3.1},
    "method": "BGMM",
    "oversample_factor": 10,  # factor on the number of initial clusters
    "z(km)": (-1, 1),
    "covariance_prior": [1e-4, 1e-2],  # time, amplitude
    "bfgs_bounds": (    # bounds in km
        (-1, 1),        # x
        (-1, 1),        # y
        (-1, 1),        # depth
        (None, None),   # t
    ),
    "use_dbscan": True,
    "dbscan_eps": 0.01,  # seconds
    "dbscan_min_samples": 5,

    "min_picks_per_eq": 5,
    "max_sigma11": 0.01,
    "max_sigma22": 2.0,
    # "max_sigma12": 2.0
}


class GaMMAPickFormatBedretto:
    def __init__(self):
        pass

    def __call__(self, sample):
        sample = sample.rename(columns={'station': 'id',
                                        'phase': 'type',
                                        # 'amplitude': 'amp',
                                        'time': 'timestamp'
                                        })
        sample = sample[['id',
                         'timestamp',
                         'type',
                        #  'amp'
                         ]]
        sample['timestamp'] = pd.to_datetime(sample['timestamp'], unit='ns')
        sample['prob'] = 1
        return sample


ds = PhasePicksDataset(
    root_dir='data/bedretto',
    stations_file='stations.csv',
    file_mask='*processed.csv',
    transform=GaMMAPickFormatBedretto(),
    station_transform=SeisBenchStationFormat()
)

# %%
print(len(ds[1].x))
# %%
cat_gmma, labels_pred = run_gamma(ds[1].x, ds.stations, config)


# %%
print(cat_gmma)


# %%
arrivals = ds[1].x.copy()
starttime = arrivals['timestamp'].min().as_unit('ns')

stations = ds.stations.copy()
stations = stations.rename(
    columns={'x(km)': 'x', 'y(km)': 'y', 'z(km)': 'z'})
stations[['x', 'y', 'z']] = stations[['x', 'y', 'z']]*1e3

arrivals = arrivals.rename(columns={'timestamp': 'time', 'id': 'station'})
arrivals['time'] = arrivals['time'] - starttime
arrivals['time'] = arrivals['time'].dt.total_seconds()*1e6
arrivals = arrivals.join(stations.set_index('id'), on='station')
arrivals = pd.concat([arrivals, pd.Series(labels_pred, name='label')], axis=1)

events = cat_gmma.copy()
events = events.rename(
    columns={'x(km)': 'x', 'y(km)': 'y', 'z(km)': 'z'})
events[['x', 'y', 'z']] = events[['x', 'y', 'z']]*1e3
events['time'] = pd.to_datetime(events['time'], unit='ns') - starttime
events['time'] = events['time'].dt.total_seconds()*1e6

# %%
plot_axis = 'y'
fig, ax = plt.subplots(figsize=(16, 12))
plt.xlabel('time [ms]', fontsize=20)
plt.ylabel(f'Station {plot_axis}-coordinate [m]', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(arrivals[plot_axis].min()-10, arrivals[plot_axis].max()+10)

color_iter = itertools.cycle(
    ["black", "navy", "c", "cornflowerblue", "gold", "orange", "green",
     "lime", "red", "purple", "blue", "pink", "brown", "gray",
     "magenta", "cyan", "olive", "maroon", "darkslategray", "darkkhaki"])

for group, df in arrivals.groupby('label'):
    color = color_iter.__next__()
    ax.scatter(df['time'], df[plot_axis],
               marker='o', color=color)
    if not group == -1:
        ax.scatter(events['time'][group], events[plot_axis][group],
                   marker='x', color=color, s=200, linewidths=8)
