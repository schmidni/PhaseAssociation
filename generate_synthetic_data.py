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
                          n_events: int,
                          n_events_fixed: bool,
                          duration: int,
                          stations: pd.DataFrame,
                          min_events: int = 1):

    center = np.array(
        [stations['e'].mean(), stations['n'].mean(), stations['u'].mean()])

    v_p = 5500.  # m/s
    v_s = 2700.  # m/s

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations.to_csv(f'{out_dir}/stations.csv', index=False)

    print("Creating synthetic catalogs...")
    for i in tqdm.tqdm(range(n_catalogs)):
        n = n_events
        if not n_events_fixed:
            # random integer number
            n = np.random.randint(min_events, n_events)
        catalog = create_synthetic_catalog(n, duration, *center)
        associations = create_associations(catalog, stations, v_p, v_s, 60)
        arrivals = associations.join(stations.set_index('id'), on='station')
        arrivals = arrivals.drop(columns=['longitude', 'latitude', 'altitude'])
        arrivals.to_csv(f'{out_dir}/arrivals_{i}.csv', index=False)
        catalog.to_csv(f'{out_dir}/catalog_{i}.csv', index=True)


if __name__ == '__main__':

    stations = inventory_to_stations('stations/station_cords_blab_VALTER.csv')
    out_dir = Path('data/raw')
    min_events = 15
    n_events = 20
    n_events_fixed = False
    duration = 15
    n_catalogs = 100

    create_synthetic_data(out_dir,
                          n_catalogs,
                          n_events,
                          n_events_fixed,
                          duration,
                          stations,
                          min_events=min_events)
