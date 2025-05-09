import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from src.synthetics.create_associations import create_associations
from src.synthetics.create_synthetic_catalog import create_synthetic_catalog


def create_synthetic_data(out_dir: Path,
                          n_catalogs: int,
                          min_events: int,
                          max_events: int,
                          duration: int,
                          stations: pd.DataFrame,
                          startdate: datetime = datetime.now(),
                          fixed_mag: float | None = None,
                          add_noise_picks: bool = True,
                          pc_noise_picks: float = 0.1,
                          overwrite: bool = False,
                          max_magnitude: float = 1.0,
                          noise_tt: float = 0.1,
                          noise_gmv: float = 0.05,
                          id: int | str = '') -> None:

    center = np.array(
        [stations['e'].mean(), stations['n'].mean(), stations['u'].mean()])

    v_p = 5400.  # m/s
    v_s = 3100.  # m/s
    percentile = 75  # percentile of gmvs to keep

    if overwrite:
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations.to_csv(f'{out_dir}/stations.csv', index=False)

    print("Creating synthetic catalogs...")
    for i in tqdm.tqdm(range(n_catalogs)):

        n_events = np.random.randint(min_events, max_events+1)  # random int

        catalog = create_synthetic_catalog(n_events,
                                           duration,
                                           *center,
                                           startdate=startdate,
                                           fixed_mag=fixed_mag,
                                           max_magnitude=max_magnitude)

        arrivals = create_associations(catalog,
                                       stations,
                                       v_p,
                                       v_s,
                                       percentile,
                                       startdate=startdate,
                                       add_noise_picks=add_noise_picks,
                                       noise_tt=noise_tt,
                                       noise_gmv=noise_gmv,
                                       pc_noise_picks=pc_noise_picks)
        ev = int(np.ceil((min_events + max_events) / 2))

        arrivals.to_csv(
            f'{out_dir}/arrivals_{duration}_{ev}_{id}_{i}.csv', index=False)
        catalog.to_csv(
            f'{out_dir}/catalog_{duration}_{ev}_{id}_{i}.csv', index=True)
