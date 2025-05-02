import numpy as np
from scipy.stats import kendalltau
from torchvision import transforms

from src.dataset import PhasePicksDataset, ReduceDatetime

ds = PhasePicksDataset(
    root_dir='data/test',
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv',
    transform=transforms.Compose([
        ReduceDatetime(),
    ])
)


def compute_cf(picks_df, eq_ids, eq_times_df):
    # Filter to only P-phase picks
    p_picks_df = picks_df[picks_df['type'] == 'P'].copy()

    # Merge in earthquake IDs and origin times
    p_picks_df['earthquake_id'] = eq_ids
    p_picks_df['origin_time'] = p_picks_df['earthquake_id'].map(
        eq_times_df['time'])
    p_picks_df['origin_time'] = p_picks_df['origin_time'] - \
        p_picks_df['origin_time'].min()

    taus = []

    p_picks_df = p_picks_df[p_picks_df['earthquake_id'] != -1]
    # Compute Kendall's τ for each station
    for _, group in p_picks_df.groupby('id_index'):

        if len(group) < 2:
            print('skip')
            continue  # Need at least 2 picks to compare ranks

        arrival_times = group['timestamp'].values
        origin_times = group['origin_time'].values

        tau, _ = kendalltau(origin_times, arrival_times)

        taus.append(tau)

    # Average τ over all stations and calculate CF
    avg_tau = np.mean(taus) if taus else 0

    cf = 1 - max(avg_tau, 0)

    return cf


cfs = []
for d in ds:
    cfs.append(compute_cf(d.x, d.y, d.catalog))

print('CF:', np.mean(cfs))
