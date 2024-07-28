# %%
import numpy as np
import pandas as pd
import tqdm

from src.clustering.utils import (ClusterStatistics, gamma_preprocess,
                                  load_data, plot_arrivals)
from src.gamma.utils import association

config = {
    "ncpu": 4,
    "dims": ['x(km)', 'y(km)', 'z(km)'],  # needs to be *(km), column names
    "use_amplitude": True,
    "vel": {"p": 5.5, "s": 2.7},
    "method": "BGMM",
    "oversample_factor": 5,  # factor on the number of initial clusters
    "z(km)": (-0.1, 0.1),
    "covariance_prior": [5e-3, 2.0],  # time, amplitude
    "bfgs_bounds": (    # bounds in km
        (-1, 1),        # x
        (-1, 1),        # y
        (-1, 1),        # depth
        (None, None),   # t
    ),
    "use_dbscan": False,
    "dbscan_eps": 5,  # seconds
    "dbscan_min_samples": 3,

    "min_picks_per_eq": 3,
    "max_sigma11": 4.0,
    "max_sigma22": 2.0,
    "max_sigma12": 2.0
}

# %%
statistics = ClusterStatistics()
for i in tqdm.tqdm(range(2)):
    arrivals, catalog, stations, reference_time = load_data(i)
    arr_input, stn_input = gamma_preprocess(arrivals, stations)

    cat_gmma, assoc_gmma = association(
        arr_input, stn_input, config, method=config["method"])

    cat_gmma = pd.DataFrame(cat_gmma)

    cat_gmma['time'] = pd.to_datetime(cat_gmma['time'])
    cat_gmma['dt'] = (cat_gmma['time'] -
                      reference_time).dt.total_seconds() * 1000
    cat_gmma['dx'] = np.sqrt((cat_gmma['x(km)']*1000)**2 +
                             (cat_gmma['y(km)']*1000)**2 +
                             (cat_gmma['z(km)']*1000)**2)

    assoc_gmma = \
        pd.DataFrame(assoc_gmma,
                     columns=["pick_index", "event_index", "gamma_score"]) \
        .set_index('pick_index')

    arrivals = arrivals.join(assoc_gmma)
    arrivals = arrivals.fillna(-1)

    labels = arrivals['event'].to_numpy()
    labels_pred = arrivals['event_index'].to_numpy()

    statistics.add(labels,
                   labels_pred,
                   len(arrivals['event'].unique()),
                   len(arrivals['event_index'].unique()))

print(f"GaMMA ARI: {statistics.ari()}, Accuray: {statistics.accuracy()}, "
      f"Precision: {statistics.precision()}, Recall: {statistics.recall()}")
print(f"GaMMA discovered {statistics.perc_eq()}% of the events correctly.")

# %%
plot_arrivals(arrivals[['dx', 'dt']],
              catalog[['dx', 'dt']],
              cat_gmma[['dx', 'dt']],
              labels, labels_pred)

# %%
