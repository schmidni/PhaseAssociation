# %%
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.synthetics.create_associations import (create_associations,
                                                inventory_to_stations)
from src.synthetics.create_synthetic_catalog import create_synthetic_catalog

startdate = datetime.now()


def create_synthetic_data(out_dir: Path,
                          n_catalogs: int,
                          min_events: int,
                          max_events: int,
                          duration: int,
                          stations: pd.DataFrame,
                          add_noise: bool = True,
                          noise_factor: float = 1):

    center = np.array(
        [stations['e'].mean(), stations['n'].mean(), stations['u'].mean()])

    v_p = 5500.  # m/s
    v_s = 2700.  # m/s

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations.to_csv(f'{out_dir}/stations.csv', index=False)

    print("Creating synthetic catalogs...")
    for i in tqdm.tqdm(range(n_catalogs)):
        n = np.random.randint(min_events, max_events+1)  # random int
        catalog = create_synthetic_catalog(
            n, duration, *center, startdate=startdate)
        arrivals = create_associations(catalog, stations, v_p, v_s, 60,
                                       duration, startdate=startdate,
                                       add_noise=False)
        arrivals2 = create_associations(catalog, stations, v_p, v_s, 60,
                                        duration, startdate=startdate,
                                        add_noise=True,
                                        noise_factor=1)
        arrivals3 = create_associations(catalog, stations, v_p, v_s, 60,
                                        duration, startdate=startdate,
                                        add_noise=True,
                                        noise_factor=3)
        arrivals4 = create_associations(catalog, stations, v_p, v_s, 60,
                                        duration, startdate=startdate,
                                        add_noise=True,
                                        noise_factor=6)
        arrivals.to_csv(f'{out_dir}/arrivals_{i}_1.csv', index=False)
        arrivals2.to_csv(f'{out_dir}/arrivals_{i}_2.csv', index=False)
        arrivals3.to_csv(f'{out_dir}/arrivals_{i}_3.csv', index=False)
        arrivals4.to_csv(f'{out_dir}/arrivals_{i}_4.csv', index=False)
        catalog.to_csv(f'{out_dir}/catalog_{i}.csv', index=True)


# %%
stations = inventory_to_stations('stations/station_cords_blab_VALTER.csv')
out_dir = Path('data/raw/associations')
min_events = 5
max_events = 5
duration = 0.0001
n_catalogs = 1

create_synthetic_data(out_dir,
                      n_catalogs,
                      min_events,
                      max_events,
                      duration,
                      stations)

# %%
labelsize = 24
ticksize = 22
markersize = 150

stations = pd.read_csv('stations/station_cords_blab_VALTER.csv')
stations.rename(columns={'station_code': 'id'}, inplace=True)
stations = stations[['id', 'x', 'y', 'z']]

arrivals = pd.read_csv(
    'data/raw/associations/arrivals_0_4.csv', parse_dates=['time'])
arrivals['time'] = arrivals['time'] - startdate
arrivals['time'] = arrivals['time'].dt.microseconds

arrivals = arrivals.join(stations.set_index('id'), on='station')

fig, ax = plt.subplots(figsize=(16, 12))
plt.xlabel('time [μs]', fontsize=labelsize)
plt.ylabel('Station y-coordinate [m]', fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.ylim(-120, 0)
plt.xlim(0, 150)


def format_func(x, pos):
    return np.round(x, 2)


# Apply the formatter to the x-axis
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))

colors = ['blue', 'green', 'red', 'purple', 'orange',
          'brown', 'pink', 'gray', 'olive', 'black']
for group, df in arrivals.groupby('event'):
    ax.scatter(df['time'], df['y'], marker='o',
               color=colors[group], s=markersize)

fig.show()

# %%
