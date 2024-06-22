from datetime import datetime

import matplotlib.pyplot as plt  # noqa
import numpy as np
import pandas as pd
from src.generate_poisson import generate_poisson_events, plot_poisson  # noqa
from src.seismicity_samples_Dieterich94 import (
    get_rectangular_slippatch_from_FM, get_seismicity_sample_from_Dieterich94)
from src.simulate_magnitudes import simulate_magnitudes


def create_synthetic_catalog(n_events: int,
                             duration: float,
                             e: float,
                             n: float,
                             u: float):

    # Specify reference focal mechanism
    stk0 = 45
    dip0 = 40
    mag = 1  # Source size, magnitude
    stressdrop = 1e6  # [Pa]

    finsrc = get_rectangular_slippatch_from_FM(
        e, n, u, stk0, dip0, mag, stressdrop)

    # Create synthetic set of FMs
    cradius = finsrc['length']  # Crack radius, defines cluster size
    dx = cradius / 20  # Stdev of scatter around perfect plane

    plot_stressdrop = False
    catalog = get_seismicity_sample_from_Dieterich94(
        n_events, e, n, u, stk0, dip0, cradius, dx, plot_stressdrop)

    keep = ['e', 'n', 'u']
    catalog = pd.DataFrame({k: catalog[k] for k in keep})

    rate = duration/n_events
    event_times = generate_poisson_events(rate, n_events)
    catalog['time'] = event_times
    catalog.time = pd.to_timedelta(catalog.time, 's')
    catalog.time = catalog.time + datetime.now()

    # plot_poisson(event_times, rate, time_duration)

    mags = simulate_magnitudes(n_events, 1.6*np.log(10), -3.5)
    catalog['magnitude'] = mags

    # print(max(mags))
    # plt.hist(mags)
    # plt.show()

    return catalog
