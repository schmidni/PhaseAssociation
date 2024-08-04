import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from src.synthetics.create_associations import (create_associations,
                                                inventory_to_stations)
from src.synthetics.create_synthetic_catalog import create_synthetic_catalog


def create_synthetic_data(out_dir: Path,
                          n_catalogs: int,
                          min_events: int,
                          max_events: int,
                          duration: int,
                          stations: pd.DataFrame,):

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
        catalog = create_synthetic_catalog(n, duration, *center)
        arrivals = create_associations(catalog, stations, v_p, v_s, 60)
        arrivals.to_csv(f'{out_dir}/arrivals_{i}.csv', index=False)
        catalog.to_csv(f'{out_dir}/catalog_{i}.csv', index=True)


if __name__ == '__main__':
    stations = inventory_to_stations('stations/station_cords_blab_VALTER.csv')
    out_dir = Path('data/raw')
    min_events = 15
    max_events = 15
    duration = 15
    n_catalogs = 10

    create_synthetic_data(out_dir,
                          n_catalogs,
                          min_events,
                          max_events,
                          duration,
                          stations)
