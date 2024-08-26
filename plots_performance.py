# %%
from pathlib import Path

import numpy as np
import pyocto
from matplotlib import pyplot as plt
from tqdm import tqdm

from create_synthetic_data import create_synthetic_data
from src.clustering.dataset import (GaMMAPickFormat, GaMMAStationFormat,
                                    PhasePicksDataset)
from src.clustering.models import run_gamma, run_pyocto
from src.clustering.utils import ClusterStatistics
from src.synthetics.create_associations import inventory_to_stations

# %%
# generate 100 easy, 100 medium, 100 hard samples
stations = inventory_to_stations('stations/station_cords_blab_VALTER.csv')
out_dir = Path('data/performance')

velocity_model = pyocto.VelocityModel0D(
    p_velocity=5.5,
    s_velocity=2.7,
    tolerance=0.1,
)

config_pyocto = pyocto.OctoAssociator(
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

config_gamma = {
    "ncpu": 4,
    "dims": ['x(km)', 'y(km)', 'z(km)'],  # needs to be *(km), column names
    "use_amplitude": False,
    "vel": {"p": 5.5, "s": 2.7},
    "method": "BGMM",
    "oversample_factor": 5,  # factor on the number of initial clusters
    "z(km)": (-1, 1),
    "covariance_prior": [1e-5, 5],  # time, amplitude
    "bfgs_bounds": (    # bounds in km
        (-1, 1),        # x
        (-1, 1),        # y
        (-1, 1),        # depth
        (None, None),   # t
    ),
    "use_dbscan": True,
    "dbscan_eps": 0.01,  # seconds
    "dbscan_min_samples": 5,

    "min_picks_per_eq": 8,
    "max_sigma11": 0.01,
    "max_sigma22": 2.0,
    # "max_sigma12": 2.0
}

data_config = {
    "easy": {
        "min_events": 4,
        "max_events": 6,
        "add_noise": True,
        "noise_factor": 1,
        "label": "1"
    },
    "medium": {
        "min_events": 45,
        "max_events": 55,
        "add_noise": True,
        "noise_factor": 1,
        "label": "10"
    },
    "hard": {
        "min_events": 240,
        "max_events": 260,
        "add_noise": True,
        "noise_factor": 1,
        "label": "50"
    },
    "veryhard": {
        "min_events": 450,
        "max_events": 550,
        "add_noise": True,
        "noise_factor": 1,
        "label": "100"
    }
}

datasets = {}
stat_gamma = {}
stat_pyocto = {}

duration = 5
n_catalogs = 10

for key, value in data_config.items():
    if True:
        create_synthetic_data(out_dir / Path(key),
                              n_catalogs,
                              value['min_events'],
                              value['max_events'],
                              duration,
                              stations,
                              add_noise=value['add_noise'],
                              noise_factor=value['noise_factor'])

    datasets[key] = PhasePicksDataset(
        root_dir=out_dir / Path(key),
        stations_file='stations.csv',
        file_mask='arrivals_*.csv',
        catalog_mask='catalog_*.csv',
        transform=GaMMAPickFormat(),
        station_transform=GaMMAStationFormat()
    )
    stat_gamma[key] = ClusterStatistics()
    stat_pyocto[key] = ClusterStatistics()


# %%
for key, ds in datasets.items():
    # if not key == 'medium':
    #     continue
    print(f"Running {key} dataset")
    for sample in tqdm(ds):
        cat_gmma, labels_pred = run_gamma(sample.x, ds.stations, config_gamma)
        stat_gamma[key].add(sample.y.to_numpy(),
                            labels_pred,
                            sample.catalog,
                            cat_gmma)

        events, labels_pred = run_pyocto(sample.x, ds.stations, config_pyocto)
        stat_pyocto[key].add(sample.y.to_numpy(),
                             labels_pred,
                             sample.catalog,
                             events)

    print(f"GaMMA {key} ARI: {stat_gamma[key].ari()}, "
          f"Accuray: {stat_gamma[key].accuracy()}, "
          f"Precision: {stat_gamma[key].precision()}, "
          f"Recall: {stat_gamma[key].recall()}")
    print(f"GaMMA event precision {key}: {stat_gamma[key].event_precision()}, "
          f"GaMMA event recall {key}: {stat_gamma[key].event_recall()}")

    print(f"PyOcto {key} ARI: {stat_pyocto[key].ari()}, "
          f"Accuray: {stat_pyocto[key].accuracy()}, "
          f"Precision: {stat_pyocto[key].precision()}, "
          f"Recall: {stat_pyocto[key].recall()}")
    print(f"PyOcto event precision {key}: "
          f"{stat_pyocto[key].event_precision()}, "
          f"PyOcto event recall {key}: {stat_pyocto[key].event_recall()}")

# %%
GaMMA_ari = np.array([stat_gamma[key].ari() for key in data_config.keys()])
GaMMA_accuracy = np.array([stat_gamma[key].accuracy()
                          for key in data_config.keys()])
GaMMA_precision = np.array([stat_gamma[key].precision()
                           for key in data_config.keys()])
GaMMA_recall = np.array([stat_gamma[key].recall()
                        for key in data_config.keys()])

PyOcto_ari = np.array([stat_pyocto[key].ari() for key in data_config.keys()])
PyOcto_accuracy = np.array([stat_pyocto[key].accuracy()
                           for key in data_config.keys()])
PyOcto_precision = np.array([stat_pyocto[key].precision()
                            for key in data_config.keys()])
PyOcto_recall = np.array([stat_pyocto[key].recall()
                         for key in data_config.keys()])

lables = [d['label'] for d in data_config.values()]

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].plot(GaMMA_ari, label='GaMMA')
ax[0, 0].plot(PyOcto_ari, label='PyOcto')
ax[0, 0].set_title('ARI')
ax[0, 0].set_xticks(range(len(data_config.keys())))
ax[0, 0].set_xticklabels(lables)
ax[0, 0].set_xlabel('⌀ events / second')
ax[0, 0].set_ylim([0, 1])
ax[0, 0].legend()

ax[0, 1].plot(GaMMA_accuracy, label='GaMMA')
ax[0, 1].plot(PyOcto_accuracy, label='PyOcto')
ax[0, 1].set_title('Accuracy')
ax[0, 1].set_xticks(range(len(data_config.keys())))
ax[0, 1].set_xticklabels(lables)
ax[0, 1].set_xlabel('⌀ events / second')
ax[0, 1].set_ylim([0, 1])
ax[0, 1].legend()

ax[1, 0].plot(GaMMA_precision, label='GaMMA')
ax[1, 0].plot(PyOcto_precision, label='PyOcto')
ax[1, 0].set_title('Precision')
ax[1, 0].set_xticks(range(len(data_config.keys())))
ax[1, 0].set_xticklabels(lables)
ax[1, 0].set_xlabel('⌀ events / second')
ax[1, 0].set_ylim([0, 1])
ax[1, 0].legend()

ax[1, 1].plot(GaMMA_recall, label='GaMMA')
ax[1, 1].plot(PyOcto_recall, label='PyOcto')
ax[1, 1].set_title('Recall')
ax[1, 1].set_xticks(range(len(data_config.keys())))
ax[1, 1].set_xticklabels(lables)
ax[1, 1].set_xlabel('⌀ events / second')
ax[1, 1].set_ylim([0, 1])
ax[1, 1].legend()

plt.show()

# %%
