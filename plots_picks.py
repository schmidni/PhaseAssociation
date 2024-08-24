# %%
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from create_synthetic_data import create_synthetic_data
from src.clustering.dataset import (GaMMAPickFormat, GaMMAStationFormat,
                                    PhasePicksDataset)
from src.synthetics.create_associations import inventory_to_stations

# %%
out_dir = Path('data/times')
n_catalogs = 1
events = 1
duration = 0.015
stations = inventory_to_stations('stations/station_cords_blab_VALTER.csv')

events = np.array([2, 5])


startdate = datetime.now()
for i, e in enumerate(events):

    event_times = np.linspace(0, duration, e+2)
    event_times = event_times[1:-1]

    create_synthetic_data(out_dir,
                          1,
                          e,
                          e,
                          duration,
                          stations,
                          add_noise=False,
                          noise_factor=1,
                          idx=i,
                          event_times=event_times,
                          startdate=startdate,
                          del_folder=False)

dataset = PhasePicksDataset(
    root_dir=out_dir,
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv',
    transform=GaMMAPickFormat(),
    station_transform=GaMMAStationFormat()
)


# %%
labelsize = 30
ticksize = 26
markersize = 175
legendsize = 30

for i in range(len(events)):

    data = dataset[i]

    event = data.catalog[['n', 'time']].copy()
    event['time'] = pd.to_datetime(event['time'], unit='ns') - startdate
    event['time'] = event['time'].dt.total_seconds()*1e3

    stations = dataset.stations
    stations = stations.rename(
        columns={'x(km)': 'x', 'y(km)': 'y', 'z(km)': 'z'})
    stations[['x', 'y', 'z']] = stations[['x', 'y', 'z']]*1e3
    arrivals = data.x

    arrivals = arrivals.rename(columns={'timestamp': 'time', 'id': 'station'})
    arrivals['time'] = arrivals['time'] - startdate
    arrivals['time'] = arrivals['time'].dt.total_seconds()*1e3
    arrivals = arrivals.join(stations.set_index('id'), on='station')
    arrivals = pd.concat([arrivals, data.y], axis=1)

    fig, ax = plt.subplots(figsize=(16, 12))
    plt.xlabel('time [ms]', fontsize=labelsize)
    plt.ylabel('Station y-coordinate [m]', fontsize=labelsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.ylim(-110, 0)
    plt.xlim(0, (duration+0.015)*1e3)

    def format_func(x, pos):
        return np.round(x, 2)

    # Apply the formatter to the x-axis
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))

    colors = ['blue', 'green', 'red', 'purple', 'orange',
              'brown', 'pink', 'gray', 'olive', 'black']

    for group, df in arrivals.groupby('event'):
        e = event.iloc[group]
        ax.scatter(df['time'], df['y'], marker='o',
                   color=colors[group], s=markersize, label='event')
        ax.scatter(e['time'], e['n'], marker='x',
                   color='red', s=markersize+10, linewidth=4, label='arrival')

    plt.legend(['Arrival', 'Event'], loc='upper left',
               prop={'size': legendsize})
    fig.show()

# %%
