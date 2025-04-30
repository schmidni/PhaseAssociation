# %%
import multiprocessing
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from create_synthetic_data import create_synthetic_data
from src.dataset import GaMMAPickFormat, GaMMAStationFormat, PhasePicksDataset
from src.metrics import ClusterStatistics
from src.plotting.arrivals import plot_arrivals
from src.runners import run_gamma
from src.synthetics.create_associations import inventory_to_stations

# %%
stations = inventory_to_stations('stations/station_cords_blab_VALTER.csv')
out_dir = Path('data/sensitivity')

config = {
    "ncpu": 1,
    "dims": ['x(km)', 'y(km)', 'z(km)'],  # needs to be *(km), column names
    "use_amplitude": True,
    "vel": {"p": 5.5, "s": 2.7},
    "method": "BGMM",
    "oversample_factor": 10,  # factor on the number of initial clusters
    "z(km)": (-1, 1),
    "covariance_prior": [1e-5, 1e-2],  # time, amplitude
    "bfgs_bounds": (    # bounds in km
        (-1, 1),        # x
        (-1, 1),        # y
        (-1, 1),        # depth
        (None, None),   # t
    ),
    "use_dbscan": True,
    "dbscan_eps": 0.01,  # seconds
    "dbscan_min_samples": 5,

    "min_picks_per_eq": 8,
    "max_sigma11": 0.01,
    "max_sigma22": 2.0,
    # "max_sigma12": 2.0
}
# %%
events_per_second = np.arange(0, 121, 5)[1:]

events = 4
n_catalogs = 3
fixed_times = True

plot = False
add_noise = True
noise_factor = 1
startdate = pd.to_datetime(datetime.now())

data_config = {}
datasets = {}
stats = {}
last_sample = {}

for j, eps in enumerate(events_per_second):
    # if fixed times is true, add some spacing at start and end
    # and therefore increase the duration
    duration = 1/eps * (events + int(fixed_times))

    data_config[j] = {
        "duration": duration,
        "events": events,
        "ticklabel": eps
    }

for key, value in data_config.items():
    event_times = np.linspace(0, value['duration'], events+2)[1:-1]
    create_synthetic_data(out_dir / Path(str(key)),
                          n_catalogs,
                          value['events'],
                          value['events'],
                          value['duration'],
                          stations,
                          add_noise_picks=add_noise,
                          noise_factor=noise_factor,
                          event_times=event_times if fixed_times else None,
                          startdate=startdate,
                          )

    datasets[key] = PhasePicksDataset(
        root_dir=out_dir / Path(str(key)),
        stations_file='stations.csv',
        file_mask='arrivals_*.csv',
        catalog_mask='catalog_*.csv',
        transform=GaMMAPickFormat(),
        station_transform=GaMMAStationFormat()
    )
    stats[key] = ClusterStatistics()


# %%
for key, ds in datasets.items():
    print(f"Running dataset {key}...")
    start = perf_counter()
    inp = [(sample.x, ds.stations, config) for sample in ds]
    n_threads = multiprocessing.cpu_count()-1
    with Pool(n_threads) as pool:
        results = pool.starmap(run_gamma, inp)

    for sample, (cat_gmma, labels_pred) in zip(ds, results):
        stats[key].add(sample.y.to_numpy(),
                       labels_pred
                       )

    print(f"GaMMA ARI: {stats[key].ari()}, "
          f"Accuray: {stats[key].accuracy()}, "
          f"Precision: {stats[key].precision()}, "
          f"Recall: {stats[key].recall()}, "
          f"Duration: {round(perf_counter()-start, 1)}s")

    last_sample[key] = (cat_gmma.copy(), labels_pred.copy(), sample)

# %%
linestyle = {'linestyle': '--',
             'marker': '^',
             'markersize': 15}
ticksize = 20
fontsize = 25

ari = np.array([stats[key].ari() for key in data_config.keys()])
precision = np.array([stats[key].precision() for key in data_config.keys()])
recall = np.array([stats[key].recall() for key in data_config.keys()])

ticklabels = [data_config[key]['ticklabel'] for key in data_config.keys()]

plt.figure(figsize=(16, 12))
plt.plot(ticklabels, ari, color='b', label='ARI', **linestyle)
plt.plot(ticklabels, precision, color='r', label='Precision', **linestyle)
plt.plot(ticklabels, recall, color='c', label='Recall', **linestyle)

plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.xlabel('Events Per Second', fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.ylim(0.5, 1.025)
plt.savefig('sensitivity.png')

# %%
if plot:
    for key, entry in last_sample.items():
        associations = entry[2].x.copy().join(
            ds.stations.set_index('id'), on='id')
        # associations['dx'] = PhasePicksDataset.get_distance(
        #     associations, ['x(km)', 'y(km)', 'z(km)'])*1000
        associations['dx'] = associations['y(km)']*1000
        associations['time'] = pd.to_datetime(
            associations['timestamp'], unit='ns').values.astype(int)
        associations['time'] = (
            pd.to_datetime(associations['timestamp'], unit='ns') - startdate
        ).values.astype(int) / 1e6

        entry[2].catalog['time'] = (
            pd.to_datetime(entry[2].catalog['time'], unit='ns') - startdate
        ).values.astype(int)/1e6
        entry[2].catalog['dx'] = entry[2].catalog['n']

        entry[0]['time'] = (pd.to_datetime(
            entry[0]['time'], unit='ns') - startdate).values.astype(int)/1e6
        entry[0]['dx'] = entry[0]['y(km)']*1000

        fig, ax = plot_arrivals(associations[['dx', 'time']],
                                entry[2].catalog[['dx', 'time']],
                                entry[0][['dx', 'time']],
                                entry[2].y.to_numpy(),
                                entry[1])

        fig.supxlabel('time [ms]', fontsize=fontsize)
        fig.supylabel('y-coordinate [m]', fontsize=fontsize)
        ax[0].tick_params(labelsize=ticksize)
        ax[1].tick_params(labelsize=ticksize)


# %%
