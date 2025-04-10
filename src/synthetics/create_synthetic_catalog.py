from datetime import datetime

import numpy as np
import pandas as pd

from src.synthetics.generate_poisson import generate_poisson_events
from src.synthetics.seismicity_samples_Dieterich94 import (
    get_rectangular_slippatch_from_FM, get_seismicity_sample_from_Dieterich94)
from src.synthetics.simulate_magnitudes import simulate_magnitudes


def create_synthetic_catalog(n_events: int,
                             duration: float,
                             e: float,
                             n: float,
                             u: float,
                             startdate: datetime = datetime.now(),
                             fixed_mag: float | None = None
                             ) -> pd.DataFrame:
    if fixed_mag is None:
        mags = simulate_magnitudes(n_events, 1.6*np.log(10), -3.5)
    else:
        mags = np.full(n_events, fixed_mag)

    # Specify reference focal mechanism
    stk0 = 45
    dip0 = 40
    mag = np.max(mags)  # Source size, magnitude
    stressdrop = 1e6  # [Pa]

    finsrc = get_rectangular_slippatch_from_FM(
        e, n, u, stk0, dip0, mag, stressdrop)

    # Create synthetic set of FMs
    cradius = finsrc['length']  # Crack radius, defines cluster size
    dx = cradius / 20  # Stdev of scatter around perfect plane

    catalog = get_seismicity_sample_from_Dieterich94(
        n_events, e, n, u, stk0, dip0, cradius, dx)

    keep = ['e', 'n', 'u']
    catalog = pd.DataFrame({k: catalog[k] for k in keep})

    rate = duration/n_events

    catalog['time'] = generate_poisson_events(rate, n_events)
    catalog.time = pd.to_timedelta(catalog.time, 's')
    catalog.time = catalog.time + startdate

    catalog['magnitude'] = mags

    return catalog
