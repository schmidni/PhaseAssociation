import numpy as np
import pandas as pd
from harpa import association as harpa_association

from src.gamma.utils import association as gamma_association


def run_pyocto(picks, stations, associator):
    events, associations = associator.associate_gamma(picks, stations)

    labels_pred = np.full(len(picks), -1)
    labels_pred[associations['pick_idx']] = associations['event_idx']

    events['z'] = 0.1
    events['time'] = events['time'].astype(int)*1e9

    events = events.rename(columns={'x': 'x(km)',
                                    'y': 'y(km)',
                                    'z': 'z(km)'})
    events['event_index'] = events.index

    return events, labels_pred


def run_gamma(picks, stations, config):
    events, associations = gamma_association(
        picks, stations, config, method=config["method"])

    events = pd.DataFrame(events)

    if len(associations) == 0 or len(events) == 0:
        return events, np.full(len(picks), -1)

    events['time'] = pd.to_datetime(
        events['time'], unit='ns').values.astype(int)

    # association columns "pick_index", "event_index", "gamma_score"
    associations = np.array([*associations])[:, :2].astype(int)
    labels_pred = np.full(len(picks), -1)
    labels_pred[associations[:, 0]] = associations[:, 1]

    return events, labels_pred


def run_harpa(picks, stations, config, verbose=False):
    pick_df_out, catalog_df = harpa_association(
        picks, stations, config, verbose=verbose)

    catalog_df['time'] = catalog_df['time'].map(
        lambda x: x.datetime)

    return catalog_df, pick_df_out['event_index'].to_numpy()
