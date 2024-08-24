from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

from src.synthetics.butler_vanaswegen import Butler_VanAswegen_1993


def inventory_to_stations(path: str):
    stations = pd.read_csv(path)
    stations.rename(columns={'station_code': 'id', 'elevation': 'altitude',
                             'x': 'e', 'y': 'n', 'z': 'u'}, inplace=True)
    stations.drop(
        columns=['network_code', 'location_code', 'chanel_code'], inplace=True)

    return stations


def create_associations(catalog: pd.DataFrame,
                        stations: pd.DataFrame,
                        v_p: float,
                        v_s: float,
                        percentile: float,
                        startdate: datetime = datetime.now(),
                        add_noise: bool = True,
                        noise_factor: float = 1,
                        noise_tt: float = 0.05,
                        noise_gmv: float = 0.1,
                        pc_noise_picks: float = 0.2
                        ) -> pd.DataFrame:
    """
    Create a synthetic dataset of phase picks for a given catalog of events.

    Args:
        catalog:        pandas dataframe with columns
                        ['time', 'magnitude', 'e', 'n', 'u']
        stations:       pandas dataframe with columns
                        ['id', 'e', 'n', 'u']
        v_p:            velocity of P-waves in m/s
        v_s:            velocity of S-waves in m/s
        percentile:     percentile of gmvs to keep
        duration:       duration of the synthetic dataset in seconds
        startdate:      startdate of the synthetic dataset
        add_noise:      add noise to the dataset
        noise_factor:   factor of noise to add, is applied on top of the
                        specific noise factors.
        noise_tt:       noise factor for travel times
        noise_gmv:      noise factor for gmvs
        pc_noise_picks: percentage of noise picks

    Returns:
        arrivals:       pandas dataframe with columns
                        ['event', 'station', 'time', 'phase', 'amplitude']
    """

    distances = distance_matrix(
        catalog[['e', 'n', 'u']], stations[['e', 'n', 'u']])
    event_times = np.array(catalog.time.values)

    # arrival times
    tt_p = distances/v_p * 1e9
    tt_s = distances/v_s * 1e9

    if add_noise:  # add noise to travel times
        tt_p += np.random.normal(
            0, noise_factor*tt_p*noise_tt, tt_p.shape)
        tt_s += np.random.normal(
            0, noise_factor*tt_s*noise_tt, tt_s.shape)

    # calculate arrival times
    at_p = tt_p.astype('timedelta64[ns]') + event_times[:, np.newaxis]
    at_s = tt_s.astype('timedelta64[ns]') + event_times[:, np.newaxis]

    def butler_proxy(distances):
        bva = np.vectorize(Butler_VanAswegen_1993)
        return bva(catalog['magnitude'], distances)[0]
    gmvs = np.apply_along_axis(butler_proxy, 0, distances)

    if add_noise:  # add noise to gmvs
        noise_gmv = np.random.normal(
            0, gmvs*noise_factor*noise_gmv, gmvs.shape)
        gmvs += np.clip(noise_gmv, -0.99*gmvs, 3*gmvs)
        gmvs = np.maximum(gmvs, 1e-6)

    # calculate a cutoff and discard picks below the cutoff
    cutoff = np.percentile(gmvs, percentile)
    detection_mask = gmvs > cutoff
    detection_mask_np = detection_mask.ravel()
    event_indices, station_indices = np.unravel_index(
        np.arange(at_p.size), at_p.shape)

    # assemble the picks to pandas dataframe
    phase_p = np.full_like(event_indices, 'P', dtype='U1')
    phase_s = np.full_like(event_indices, 'S', dtype='U1')

    at_p_np = np.array(
        list(zip(event_indices, station_indices,
             at_p.ravel(), phase_p, gmvs.ravel())),
        dtype=[('event', 'i4'),
               ('station', 'i4'),
               ('time', 'datetime64[ns]'),
               ('phase', 'U1'),
               ('amplitude', 'f8')])[detection_mask_np]
    at_p_np = pd.DataFrame(at_p_np)

    at_s_np = np.array(
        list(zip(event_indices, station_indices,
             at_s.ravel(), phase_s, gmvs.ravel())),
        dtype=[('event', 'i4'),
               ('station', 'i4'),
               ('time', 'datetime64[ns]'),
               ('phase', 'U1'),
               ('amplitude', 'f8')])[detection_mask_np]
    at_s_np = pd.DataFrame(at_s_np)

    if add_noise:  # add false picks to dataset
        n_noise = int(len(at_p_np)*pc_noise_picks*noise_factor)

        # sample stations with replacement
        stations_noise = stations['id'].sample(
            n_noise, replace=True).index.to_numpy()

        # true label of noise is -1
        events_noise = np.full(n_noise, -1)

        # sample random times uniformly distributed
        actual_duration = at_s_np['time'].max() - startdate
        actual_duration = actual_duration.total_seconds() * 1e9

        times_noise = \
            pd.to_timedelta(
                np.random.uniform(0, actual_duration, n_noise), 'ns') \
            + pd.to_datetime(startdate)

        # sample random phases
        phases_noise = np.random.choice(['P', 'S'], n_noise)

        # sample random amplitudes
        amplitudes_noise = at_s_np['amplitude'] \
            .sample(n_noise, replace=True).to_numpy()
        amplitudes_noise = np.random.normal(
            amplitudes_noise, 0.3*amplitudes_noise*noise_factor)
        amplitudes_noise = np.maximum(amplitudes_noise, 1e-6)

        # assemble the noise picks to pandas dataframe
        noise_np = np.array(
            list(zip(events_noise, stations_noise, times_noise,
                 phases_noise, amplitudes_noise)),
            dtype=[('event', 'i4'),
                   ('station', 'i4'),
                   ('time', 'datetime64[ns]'),
                   ('phase', 'U1'),
                   ('amplitude', 'f8')])
        noise = pd.DataFrame(noise_np)
    else:
        noise = pd.DataFrame()

    # concatenate all picks into one dataset
    arrivals = pd.concat([at_p_np, at_s_np, noise], ignore_index=True)
    arrivals['station'] = arrivals['station'].map(stations['id'])

    # shuffle the dataset
    arrivals = arrivals.sample(frac=1).reset_index(drop=True)

    return arrivals
