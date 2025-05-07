import multiprocessing
import warnings

import pyocto
import tqdm

from src import run_phassoc
from src.metrics import ClusterStatistics
from src.runners import run_gamma, run_harpa, run_pyocto


def evaluate_padme(ds):

    config = {
        'dbscan_eps': 1000,
        'dbscan_min_samples': 4,
        'vel': {
            'P': 5.4,
            'S': 3.1
        },
        'model': '../../models/model_33k_2',
        'min_picks_per_event': 4
    }

    statistics = ClusterStatistics()

    for sample in tqdm.tqdm(ds):
        _, picks, embeddings = run_phassoc(
            sample.x, ds.stations, config, verbose=True)
        # y = sample.y.where(sample.y.map(sample.y.value_counts())
        #                    > config['min_picks_per_event'], -1)

        # statistics.add(y.to_numpy(), picks['labels'].to_numpy())
        statistics.add_embedding(embeddings, sample.y.to_numpy())
    return statistics


def evaluate_gamma(ds):
    config = {
        "ncpu": multiprocessing.cpu_count()-1,
        "dims": ['x(km)', 'y(km)', 'z(km)'],  # needs to be *(km), column names
        "use_amplitude": True,
        "vel": {"p": 5.4, "s": 3.1},
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
        "dbscan_eps": 0.01,  # seconds
        "dbscan_min_samples": 5,

        "min_picks_per_eq": 8,
        "max_sigma11": 0.01,
        "max_sigma22": 2.0,
        # "max_sigma12": 2.0
    }

    statistics = ClusterStatistics()

    for sample in tqdm.tqdm(ds):
        cat_gmma, labels_pred = run_gamma(sample.x, ds.stations, config)

        statistics.add(sample.y.to_numpy(),
                       labels_pred)

    return statistics


def evaluate_harpa(ds):
    statistics = ClusterStatistics()
    config = {
        'x(km)': (-0.5, 0.5),
        'y(km)': (-0.5, 0.5),
        'z(km)': (-0.5, 0.5),
        'vel': {'P': 5.4, 'S': 3.1},
        'P_phase': True,
        'S_phase': True,
        'min_peak_pre_event': 4,
        'min_peak_pre_event_s': 0,
        'min_peak_pre_event_p': 0,
        # 'dbscan_eps': 0.1,
        # 'dbscan_min_samples': 4,
        # 'lr': 0.01,
        'lr_decay': 0.05,
        # 'wasserstein_p': 2,
        'max_time_residual':  0.1,
        'epochs_before_decay': 10000,
        'epochs_after_decay': 10000,
    }

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    for sample in tqdm.tqdm(ds):
        cat_harpa, labels_harpa = run_harpa(
            sample.x, ds.stations, config, verbose=True)
        y = sample.y.where(sample.y.map(sample.y.value_counts())
                           > config['min_peak_pre_event'], -1)
        statistics.add(y.to_numpy(), labels_harpa)

    return statistics


def evaluate_pyocto(ds):
    statistics = ClusterStatistics()
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

    for sample in tqdm.tqdm(ds):
        events, labels_pred = run_pyocto(sample.x, ds.stations, associator)

        statistics.add(sample.y.to_numpy(),
                       labels_pred)

    return statistics
