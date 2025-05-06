# %% Imports and Configuration
import glob
import multiprocessing
import os
from copy import deepcopy

import pandas as pd
import pyocto

from src import run_phassoc
from src.dataset import (PhasePicksDataset, SeisBenchPickFormat,
                         SeisBenchStationFormat)
from src.plotting.arrivals import plot_arrivals
from src.plotting.embeddings import plot_embeddings
from src.runners import run_gamma, run_harpa, run_pyocto

# %% preprocess
catalog_files = sorted(
    glob.glob(os.path.join('../../data/m0', '*Ss0.30.csv')))

for file in catalog_files:
    data = pd.read_csv(file)
    data = data.rename(columns={'trace_id': 'station',
                                'onset_time': 'time'})
    data['station'] = data['station'].str[3:8]
    data.to_csv(file.replace('.csv', '_processed.csv'))

ds = PhasePicksDataset(
    root_dir='../../data/m0',
    stations_file='stations.csv',
    file_mask='*processed.csv',
    transform=SeisBenchPickFormat(),
    station_transform=SeisBenchStationFormat()
)

SAMPLE = deepcopy(ds[1].x)

# %%
# Filter defective stations from the dataset
length_sample = len(SAMPLE)
sizes = SAMPLE.groupby('id').size()
# plt.bar(sizes.index, sizes.values)
# plt.xticks(rotation=90)
# plt.show()
filter = ['V0506', 'V0510', 'V0702', 'V0812']
SAMPLE = SAMPLE[~SAMPLE['id'].isin(filter)].reset_index(drop=True)
print(
    f'Filtered {length_sample-len(SAMPLE)} picks from {length_sample} picks')

# %%
config_gamma = {
    "ncpu": multiprocessing.cpu_count()-1,
    "dims": ['x(km)', 'y(km)', 'z(km)'],  # needs to be *(km), column names
    "use_amplitude": False,
    "vel": {"p": 5.4, "s": 3.1},
    "method": "BGMM",
    "oversample_factor": 10,  # factor on the number of initial clusters
    "z(km)": (-1, 1),
    "covariance_prior": [1e-4, 1e-2],  # time, amplitude
    "bfgs_bounds": (    # bounds in km
        (-1, 1),        # x
        (-1, 1),        # y
        (-1, 1),        # depth
        (None, None),   # t
    ),
    "use_dbscan": True,
    "dbscan_eps": 0.01,  # seconds
    "dbscan_min_samples": 5,

    "min_picks_per_eq": 6,
    "max_sigma11": 0.01,
    "max_sigma22": 2.0,
    # "max_sigma12": 2.0
}
cat_gamma, labels_gamma = run_gamma(SAMPLE, ds.stations, config_gamma)

# %%
plot_arrivals(SAMPLE, labels_gamma, ds.stations, cat_gamma, title='GaMMA')


# %%
config_harpa = {
    'x(km)': (-0.5, 0.5),
    'y(km)': (-0.5, 0.5),
    'z(km)': (-0.5, 0.5),
    'vel': {'P': 5.4, 'S': 3.1},
    'P_phase': True,
    'S_phase': True,
    'min_peak_pre_event': 6,
    'min_peak_pre_event_s': 0,
    'min_peak_pre_event_p': 0,
    'lr_decay': 0.05,
    'max_time_residual':  0.01,
    'epochs_before_decay': 10000,
    'epochs_after_decay': 10000,
}

cat_harpa, labels_harpa = run_harpa(
    SAMPLE, ds.stations, config_harpa, verbose=True)

# %%
plot_arrivals(SAMPLE, labels_harpa, ds.stations, cat_harpa, title='HARPA')

# %%
config = {
    'dbscan_eps': 0.1,
    'vel': {
        'P': 5.4,
        'S': 3.1
    },
    'model': '../../models/model_noamp',
    'min_picks_per_event': 6
}

_, labels_phassoc, embeddings = run_phassoc(
    SAMPLE, ds.stations, config, verbose=True)

# %%
plot_arrivals(SAMPLE, labels_phassoc['labels'], ds.stations, title='PhAssoc')
plot_embeddings(embeddings, labels_phassoc['labels'], method='pca')

# %%
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

cat_pyocto, labels_pyocto = run_pyocto(SAMPLE, ds.stations, associator)


# %%
plot_arrivals(SAMPLE, labels_pyocto, ds.stations, cat_pyocto, title='PyOcto')

# %%
