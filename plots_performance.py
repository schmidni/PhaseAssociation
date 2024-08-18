# %%
from pathlib import Path

import numpy as np
import pyocto
from tqdm import tqdm

from create_synthetic_data import create_synthetic_data
from src.clustering.dataset import (GaMMAPickFormat, GaMMAStationFormat,
                                    PhasePicksDataset)
from src.clustering.models import run_gamma, run_pyocto
from src.clustering.utils import ClusterStatistics
from src.synthetics.create_associations import inventory_to_stations

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

    time_before=1,  # 300,

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
    time_slicing=5,  # 1200.0,
    node_log_interval=0,  # 0,

    location_split_depth=6,  # 6,
    location_split_return=4,  # 4,
    min_pick_fraction=0.25
)

config_gamma = {
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

data_config = {
    "easy": {
        "min_events": 3,
        "max_events": 6,
        "add_noise": True,
        "percent_noise": 0.1
    },
    "medium": {
        "min_events": 20,
        "max_events": 30,
        "add_noise": True,
        "percent_noise": 0.2
    },
    "hard": {
        "min_events": 40,
        "max_events": 60,
        "add_noise": True,
        "percent_noise": 0.3
    }
}

datasets = {}
stat_gamma = {}
stat_pyocto = {}

duration = 10
n_catalogs = 100

for key, value in data_config.items():
    if False:
        create_synthetic_data(out_dir / Path(key),
                              n_catalogs,
                              value['min_events'],
                              value['max_events'],
                              duration,
                              stations,
                              add_noise=value['add_noise'],
                              percent_noise=value['percent_noise'])

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
                            len(sample.y.unique())-1,
                            len(np.unique(labels_pred)-1))

        events, labels_pred = run_pyocto(sample.x, ds.stations, config_pyocto)
        stat_pyocto[key].add(sample.y.to_numpy(),
                             labels_pred,
                             len(sample.y.unique())-1,
                             len(np.unique(labels_pred)-1))

    print(f"GaMMA {key} ARI: {stat_gamma[key].ari()}, "
          f"Accuray: {stat_gamma[key].accuracy()}, "
          f"Precision: {stat_gamma[key].precision()}, "
          f"Recall: {stat_gamma[key].recall()}")
    print(f"GaMMA {key} discovered {stat_gamma[key].perc_eq()}% "
          "of the events correctly.")
    print(f"PyOcto {key} ARI: {stat_pyocto[key].ari()}, "
          f"Accuray: {stat_pyocto[key].accuracy()}, "
          f"Precision: {stat_pyocto[key].precision()}, "
          f"Recall: {stat_pyocto[key].recall()}")
    print(f"PyOcto {key} discovered {stat_pyocto[key].perc_eq()}% "
          "of the events correctly.")
