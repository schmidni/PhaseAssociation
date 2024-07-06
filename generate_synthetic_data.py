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
                          max_n_events: int,
                          duration: int,
                          stations: pd.DataFrame):

    center = np.array(
        [stations['e'].mean(), stations['n'].mean(), stations['u'].mean()])

    v_p = 5500.  # m/s
    v_s = 2700.  # m/s

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations.to_csv(f'{out_dir}/stations.csv', index=False)

    print("Creating synthetic catalogs...")
    for i in tqdm.tqdm(range(n_catalogs)):
        # random integer number beween 1 and 100
        n_events = np.random.randint(1, max_n_events)
        catalog = create_synthetic_catalog(n_events, duration, *center)
        associations = create_associations(catalog, stations, v_p, v_s, 60)
        arrivals = associations.join(stations.set_index('id'), on='station')
        arrivals = arrivals.drop(columns=['longitude', 'latitude', 'altitude'])
        arrivals.to_csv(f'{out_dir}/arrivals_{i}.csv', index=False)


if __name__ == '__main__':

    stations = inventory_to_stations('stations/station_cords_blab_VALTER.csv')
    out_dir = Path('data/raw')
    max_n_events = 15
    duration = 30
    n_catalogs = 1

    create_synthetic_data(out_dir, n_catalogs,
                          max_n_events, duration, stations)
