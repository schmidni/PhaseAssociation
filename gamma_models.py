# %%
import pandas as pd
import tqdm

from src.clustering.dataset import (GaMMAPickFormat, GaMMAStationFormat,
                                    PhasePicksDataset)
from src.clustering.utils import ClusterStatistics, plot_arrivals
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
    "use_dbscan": True,
    "dbscan_eps": 0.01,  # seconds
    "dbscan_min_samples": 3,

    "min_picks_per_eq": 2,
    "max_sigma11": 4.0,
    "max_sigma22": 2.0,
    "max_sigma12": 2.0
}

# %%
statistics = ClusterStatistics()

ds = PhasePicksDataset(
    root_dir='data/raw',
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv',
    transform=GaMMAPickFormat(),
    station_transform=GaMMAStationFormat()
)

for sample in tqdm.tqdm(ds):
    cat_gmma, assoc_gmma = association(
        sample.x, ds.stations, config, method=config["method"])

    cat_gmma = pd.DataFrame(cat_gmma)
    assoc_gmma = \
        pd.DataFrame(assoc_gmma,
                     columns=["pick_index", "event_index", "gamma_score"]) \
        .set_index('pick_index')

    assoc_gmma = sample.x.join(assoc_gmma)
    assoc_gmma = assoc_gmma.fillna(-1)

    labels = sample.y.to_numpy()
    labels_pred = assoc_gmma['event_index'].to_numpy()

    statistics.add(labels,
                   labels_pred,
                   len(sample.y.unique()),
                   len(assoc_gmma['event_index'].unique()))


print(f"GaMMA ARI: {statistics.ari()}, Accuray: {statistics.accuracy()}, "
      f"Precision: {statistics.precision()}, Recall: {statistics.recall()}")
print(f"GaMMA discovered {statistics.perc_eq()}% of the events correctly.")

# %%
# Plot Results
cat_gmma['time'] = pd.to_datetime(
    cat_gmma['time'], unit='ns').values.astype(int)
cat_gmma['dx'] = PhasePicksDataset.get_distance(
    cat_gmma, ['x(km)', 'y(km)', 'z(km)'])*1000

catalog_real = sample.catalog.copy()
catalog_real['dx'] = PhasePicksDataset.get_distance(
    catalog_real)

associations = assoc_gmma.join(ds.stations.set_index('id'), on='id')
associations['dx'] = PhasePicksDataset.get_distance(
    associations, ['x(km)', 'y(km)', 'z(km)'])*1000

associations.rename(columns={'timestamp': 'time'}, inplace=True)
associations['time'] = pd.to_datetime(
    associations['time'], unit='ns').values.astype(int)

plot_arrivals(associations[['dx', 'time']],
              catalog_real[['dx', 'time']],
              cat_gmma[['dx', 'time']],
              labels, labels_pred)

# %%
