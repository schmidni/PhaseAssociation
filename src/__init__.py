import copy
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from src.models import PhasePickTransformer


def estimate_eps(stations, vp, sigma=3.0):
    X = stations[['x(km)', 'y(km)', 'z(km)']].values
    D = np.sqrt(
        ((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=-1))
    Tcsr = minimum_spanning_tree(D).toarray()

    eps = np.max(Tcsr) / vp * 1.5

    return eps


def DBSCAN_cluster(picks, stations, config):

    if type(picks['timestamp'].iloc[0]) is str:
        picks.loc[:, 'timestamp'] = picks['timestamp'].apply(
            lambda x: datetime.fromisoformat(x))
    t = (
        picks['timestamp']
        .apply(lambda x: x.tz_convert('UTC').timestamp()
               if x.tzinfo is not None else x.tz_localize('UTC').timestamp())
        .to_numpy()
    )

    timestamp0 = np.min(t)
    t = t - timestamp0
    data = t[:, np.newaxis]

    meta = stations.merge(picks['id'], how='right',
                          on='id', validate='one_to_many')

    locs = meta[['x(km)', 'y(km)', 'z(km)']].to_numpy()

    if 'dbscan_eps' in config:
        eps = config['dbscan_eps']
    else:
        eps = estimate_eps(stations, config['vel']['P'])

    if 'dbscan_min_samples' in config:
        dbscan_min_samples = config['dbscan_min_samples']
    else:
        dbscan_min_samples = 3

    vel = config['vel']

    db = DBSCAN(eps=eps, min_samples=dbscan_min_samples).fit(
        np.hstack([data[:, 0:1], locs[:, :2] / np.average(vel['P'])]),
    )

    labels = db.labels_
    unique_labels = set(labels)
    unique_labels = unique_labels.difference([-1])
    picks['dbs'] = labels

    picks['dbs'] = picks['dbs'].replace(-1,
                                        pd.NA).astype('Int64').ffill().bfill()

    return picks, unique_labels


def run_phassoc(picks, stations, model_path, min_samples):

    picks = picks.join(stations.set_index('id'), on='id')
    station_index_mapping = pd.Series(stations.index, index=stations['id'])
    picks['id_index'] = picks['id'].map(station_index_mapping)
    picks = picks.drop(
        columns=['prob', 'id'])
    picks['timestamp'] = picks['timestamp'].values.astype(int)
    picks['timestamp'] = picks['timestamp'] - picks['timestamp'].min()
    picks['type'] = picks['type'].astype('category').cat.codes
    picks = picks[['x(km)', 'y(km)', 'z(km)', 'timestamp',
                   'type', 'amp', 'id_index']]

    scaler = MinMaxScaler()
    for col in ['x(km)', 'y(km)', 'z(km)', 'timestamp', 'amp']:
        picks[col] = scaler.fit_transform(picks[col].to_numpy().reshape(-1, 1))

    st = torch.tensor(picks['id_index'].to_numpy()).to(
        dtype=torch.long).unsqueeze(0)
    xs = torch.tensor(picks.drop(columns='id_index').to_numpy()).to(
        dtype=torch.float32).unsqueeze(0)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    model = PhasePickTransformer(
        input_dim=6, num_stations=len(stations),
        embed_dim=128, num_heads=4, num_layers=2,
        max_picks=2000).to(device)

    model.load_state_dict(torch.load(model_path,
                                     weights_only=True,
                                     map_location=device))

    with torch.no_grad():
        model.eval()
        embeddings = model(xs.to(device), st.to(device))

    eps = estimate_eps_with_elbow(
        embeddings.squeeze(), min_samples=min_samples, plot=False)

    embeddings = embeddings.cpu().squeeze().detach().numpy()
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)

    picks['labels'] = db.labels_

    return None, picks


def associate_phassoc(picks, station_df, config, verbose=False):
    picks = copy.deepcopy(picks)

    picks, unique_labels = DBSCAN_cluster(picks, station_df, config)
    if verbose:
        print(f'Associating {len(picks)} picks separated into '
              f'{len(unique_labels)} slides.')

    pick_df_list = []
    for slice_index in range(len(unique_labels)):
        _, pick_df = \
            run_phassoc(picks[picks['dbs'] == slice_index],
                        station_df,
                        config['model'],
                        config['min_picks_per_event'])
        pick_df_list.append(pick_df)
    pick_df = reindex_picks(pick_df_list)

    if verbose:
        print(f'Associated {len(np.unique(pick_df["labels"]))-1} '
              'unique events.')

    return pick_df, None


def reindex_picks(pick_df_list):
    result = []
    label_offset = 0

    for df in pick_df_list:
        df_copy = df.copy()
        mask = df_copy['labels'] >= 0  # identify valid (non-noise) labels

        # Shift valid labels
        df_copy.loc[mask, 'labels'] += label_offset

        # Update label_offset to ensure future labels are unique
        if not df_copy[mask].empty:
            label_offset = df_copy[mask]['labels'].max() + 1

        result.append(df_copy)

    return pd.concat(result, ignore_index=True)


def estimate_eps_with_elbow(X, min_samples, plot=True):
    # Step 1: Compute distances to k-th nearest neighbor
    dist_matrix = torch.cdist(X, X)
    sorted_dists, _ = torch.sort(dist_matrix, dim=1)
    kth_distances = sorted_dists[:, min_samples-1]  # distance to k-th neighbor
    kth_distances_np = kth_distances.cpu().numpy()
    sorted_kd = np.sort(kth_distances_np)

    # Step 2: Compute line from first to last point
    n_points = len(sorted_kd)
    all_coords = np.vstack((np.arange(n_points), sorted_kd)).T
    start, end = all_coords[0], all_coords[-1]

    # Vector between start and end
    line_vec = end - start
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    # Vector from start to each point
    vec_from_start = all_coords - start
    # Project onto the line (perpendicular distance)
    scalar_proj = np.dot(vec_from_start, line_vec_norm)
    proj = np.outer(scalar_proj, line_vec_norm)
    vec_to_line = vec_from_start - proj
    dist_to_line = np.linalg.norm(vec_to_line, axis=1)

    # Step 3: Find the index with the max distance — the elbow
    elbow_index = np.argmax(dist_to_line)
    eps_estimate = sorted_kd[elbow_index]

    return eps_estimate
