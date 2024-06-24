# %% Imports
import shutil
from pathlib import Path

import numpy as np

from src.create_associations import create_associations, inventory_to_stations
from src.create_synthetic_catalog import create_synthetic_catalog

# %%
# stations = inventory_to_stations('stations/station_cords_blab_VALTER.csv')
# stations.to_csv('stations.csv', index=False)

# # %%
# n_events = 100
# duration = 300
# e0 = stations['e'].mean()
# n0 = stations['n'].mean()
# u0 = stations['u'].mean()

# # %%
# catalog = create_synthetic_catalog(n_events, duration, e0, n0, u0)
# catalog.to_csv('synthetic_catalog.csv')

# # %%
# v_p = 5500.  # m/s
# v_s = 2700.  # m/s
# associations = create_associations(catalog, stations, v_p, v_s, 60)
# associations.to_csv('associations.csv', index=False)

# %%
stations = inventory_to_stations('stations/station_cords_blab_VALTER.csv')
e0 = stations['e'].mean()
n0 = stations['n'].mean()
u0 = stations['u'].mean()

duration = 30

v_p = 5500.  # m/s
v_s = 2700.  # m/s

out_dir = Path('data/raw')
shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

for i in range(10000):
    # random integer number beween 1 and 100
    n_events = np.random.randint(1, 15)
    catalog = create_synthetic_catalog(n_events, duration, e0, n0, u0)
    associations = create_associations(catalog, stations, v_p, v_s, 60)
    arrivals = associations.join(stations.set_index('id'), on='station')
    arrivals = arrivals.drop(columns=['longitude', 'latitude', 'altitude'])
    arrivals.to_csv(f'{out_dir}/arrivals_{i}.csv', index=False)

# %%
