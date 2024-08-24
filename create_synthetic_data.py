import shutil
from datetime import datetime
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
                          stations: pd.DataFrame,
                          add_noise: bool = True,
                          noise_factor: float = 1,
                          idx: int = 1,
                          startdate: datetime = datetime.now(),
                          event_times: np.ndarray | None = None,
                          del_folder: bool = True):

    center = np.array(
        [stations['e'].mean(), stations['n'].mean(), stations['u'].mean()])

    v_p = 5500.  # m/s
    v_s = 2700.  # m/s

    if del_folder:
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations.to_csv(f'{out_dir}/stations.csv', index=False)

    print("Creating synthetic catalogs...")
    for i in tqdm.tqdm(range(n_catalogs)):
        n = np.random.randint(min_events, max_events+1)  # random int
        catalog = create_synthetic_catalog(
            n, duration, *center, startdate=startdate, event_times=event_times)
        arrivals = create_associations(catalog, stations, v_p, v_s, 60,
                                       duration, startdate=startdate,
                                       add_noise=add_noise,
                                       noise_factor=noise_factor)

        arrivals.to_csv(f'{out_dir}/arrivals_{(i+1)*idx}.csv', index=False)
        catalog.to_csv(f'{out_dir}/catalog_{(i+1)*idx}.csv', index=True)


if __name__ == '__main__':
    stations = inventory_to_stations('stations/station_cords_blab_VALTER.csv')
    out_dir = Path('data/raw')
    min_events = 190
    max_events = 210
    duration = 20
    n_catalogs = 10
    add_noise = True
    noise_factor = 1

    create_synthetic_data(out_dir,
                          n_catalogs,
                          min_events,
                          max_events,
                          duration,
                          stations,
                          add_noise=add_noise,
                          noise_factor=noise_factor)
