# %% Imports and Configuration
import itertools

import pyocto
import tqdm

from src.dataset import (PhasePicksDataset, SeisBenchPickFormat,
                         SeisBenchStationFormat)
from src.metrics import ClusterStatistics  # , plot_arrivals
from src.runners import run_pyocto

velocity_model = pyocto.VelocityModel0D(
    p_velocity=5.5,
    s_velocity=2.7,
    tolerance=0.01,
)

associator = pyocto.OctoAssociator(
    xlim=(0, -0.25),
    ylim=(0, -0.15),
    zlim=(0, 0.25),

    time_before=0.1,  # 300,

    velocity_model=velocity_model,

    n_picks=8,
    n_p_and_s_picks=4,
    n_p_picks=4,
    n_s_picks=4,

    min_node_size=1e-1,  # 10.0, min node size for association
    min_node_size_location=1,  # 1.5, min node size for location

    pick_match_tolerance=0.1,  # 1.5, max diff predicted/observed time

    min_interevent_time=1e-10,  # 3.0, min time between events

    exponential_edt=False,  # leading to better locations
    edt_pick_std=1.0,  # 1.0,

    max_pick_overlap=4,  # 4, max number of picks shared between events

    refinement_iterations=3,  # 3,
    time_slicing=0.001,  # 1200.0,
    node_log_interval=0,  # 0, # logging interval in seconds

    location_split_depth=6,  # 6,
    location_split_return=4,  # 4,
    min_pick_fraction=0.25
)

# %% Run PyOcto
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
    events, labels_pred = run_pyocto(sample.x, ds.stations, associator)

    statistics.add(sample.y.to_numpy(),
                   labels_pred)

print(f"PyOcto ARI: {statistics.ari()}, Accuray: {statistics.accuracy()}, "
      f"Precision: {statistics.precision()}, Recall: {statistics.recall()}")

# %% Plot Results
# associations = sample.x.copy().join(ds.stations.set_index('id'), on='id')
# associations['dx'] = PhasePicksDataset.get_distance(
#     associations, ['x(km)', 'y(km)', 'z(km)'])*1000
# associations['time'] = pd.to_datetime(
#     associations['timestamp'], unit='ns').values.astype(int)

# plot_arrivals(associations[['dx', 'time']],
#               sample.catalog[['dx', 'time']],
#               events[['dx', 'time']],
#               sample.y.to_numpy(), labels_pred)

# %%
