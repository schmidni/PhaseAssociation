# %% Imports and Configuration
import itertools
import multiprocessing

import tqdm

from src.dataset import (PhasePicksDataset, SeisBenchPickFormat,
                         SeisBenchStationFormat)
from src.metrics import ClusterStatistics  # , plot_arrivals
from src.runners import run_gamma

config = {
    "ncpu": multiprocessing.cpu_count()-1,
    "dims": ['x(km)', 'y(km)', 'z(km)'],  # needs to be *(km), column names
    "use_amplitude": True,
    "vel": {"p": 5.5, "s": 2.7},
    "method": "BGMM",
    "oversample_factor": 10,  # factor on the number of initial clusters
    "z(km)": (-1, 1),
    "covariance_prior": [1e-5, 1e-2],  # time, amplitude
    "bfgs_bounds": (    # bounds in km
        (-1, 1),        # x
        (-1, 1),        # y
        (-1, 1),        # depth
        (None, None),   # t
    ),
    "use_dbscan": True,
    "dbscan_eps": 0.05,  # seconds
    "dbscan_min_samples": 5,

    "min_picks_per_eq": 8,
    "max_sigma11": 0.01,
    "max_sigma22": 2.0,
    # "max_sigma12": 2.0
}


# %% Run GaMMA
statistics = ClusterStatistics()


ds = PhasePicksDataset(
    root_dir='../../data/test',
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv',
    transform=SeisBenchPickFormat(),
    station_transform=SeisBenchStationFormat()
)


for sample in tqdm.tqdm(itertools.islice(ds, 5)):
    cat_gmma, labels_pred = run_gamma(sample.x, ds.stations, config)

    statistics.add(sample.y.to_numpy(),
                   labels_pred)

    print(f"GaMMA ARI: {statistics.ari()}, Accuray: {statistics.accuracy()}, ")

print(f"GaMMA ARI: {statistics.ari()}, Accuray: {statistics.accuracy()}, "
      f"Precision: {statistics.precision()}, Recall: {statistics.recall()}")


# %% Plot Results
# associations = sample.x.copy().join(ds.stations.set_index('id'), on='id')
# associations['dx'] = PhasePicksDataset.get_distance(
#     associations, ['x(km)', 'y(km)', 'z(km)'])*1000
# associations['time'] = pd.to_datetime(
#     associations['timestamp'], unit='ns').values.astype(int)

# plot_arrivals(associations[['dx', 'time']],
#               sample.catalog[['dx', 'time']],
#               cat_gmma[['dx', 'time']],
#               sample.y.to_numpy(), labels_pred)
