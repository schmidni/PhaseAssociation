import numpy as np
import pandas as pd

from src.dataset import PhasePicksDataset
from src.gamma.utils import association


def run_pyocto(picks, stations, associator):
    events, associations = associator.associate_gamma(picks, stations)

    labels_pred = np.full(len(picks), -1)
    labels_pred[associations['pick_idx']] = associations['event_idx']

    events['z'] = 0.1
    events['dx'] = PhasePicksDataset.get_distance(
        events, ['x', 'y', 'z'])*1000
    events['time'] = events['time'].astype(int)*1e9

    return events, labels_pred


def run_gamma(picks, stations, config):
    events, associations = association(
        picks, stations, config, method=config["method"])

    events = pd.DataFrame(events)

    if len(associations) == 0:
        return events, np.full(len(picks), -1)

    events['time'] = pd.to_datetime(
        events['time'], unit='ns').values.astype(int)
    events['dx'] = PhasePicksDataset.get_distance(
        events, ['x(km)', 'y(km)', 'z(km)'])*1000

    # association columns "pick_index", "event_index", "gamma_score"
    associations = np.array([*associations])[:, :2].astype(int)
    labels_pred = np.full(len(picks), -1)
    labels_pred[associations[:, 0]] = associations[:, 1]

    return events, labels_pred
