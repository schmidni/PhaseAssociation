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
                        duration: int,
                        startdate: datetime = datetime.now(),
                        add_noise: bool = False,
                        percent_noise: float = 0.1
                        ) -> pd.DataFrame:
    distances = distance_matrix(
        catalog[['e', 'n', 'u']], stations[['e', 'n', 'u']])

    event_times = np.array(catalog.time.values)

    # arrival times
    at_p = (distances/v_p *
            1e6).astype('timedelta64[ns]')
    at_s = (distances/v_s *
            1e6).astype('timedelta64[ns]')

    if add_noise:
        # add +/- 0-x% noise to arrival times
        noise_at = np.random.uniform(-percent_noise, percent_noise, at_p.shape)
        at_p += at_p*noise_at
        noise_at = np.random.uniform(-percent_noise, percent_noise, at_s.shape)
        at_s += at_s*noise_at

    at_p = at_p + event_times[:, np.newaxis]
    at_s = at_s + event_times[:, np.newaxis]

    def butler_proxy(distances):
        bva = np.vectorize(Butler_VanAswegen_1993)
        return bva(catalog['magnitude'], distances)[0]
    gmvs = np.apply_along_axis(butler_proxy, 0, distances)

    if add_noise:
        # add noise to gmvs
        noise_gmv = np.random.normal(0, percent_noise*5, gmvs.shape)
        gmvs += gmvs*np.clip(noise_gmv, -1, 5)

    cutoff = np.percentile(gmvs, percentile)
    detection_mask = gmvs > cutoff
    detection_mask_np = detection_mask.ravel()
    event_indices, station_indices = np.unravel_index(
        np.arange(at_p.size), at_p.shape)

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

    if add_noise:
        # add x% noise picks
        n_noise = int(len(at_p_np)*2*percent_noise)

        # add x% noise picks
        stations_noise = stations['id'].sample(
            n_noise, replace=True).index.to_numpy()
        events_noise = np.full(n_noise, -1)

        times_noise = \
            pd.to_timedelta(np.random.uniform(0, duration, n_noise), 's') \
            + pd.to_datetime(startdate)

        phases_noise = np.random.choice(['P', 'S'], n_noise)
        amplitudes_noise = at_s_np['amplitude'] \
            .sample(n_noise, replace=True).to_numpy()
        amplitudes_noise += np.clip(
            np.random.normal(0, percent_noise*5, n_noise), -0.4, 1) \
            * amplitudes_noise

        noise_np = np.array(
            list(zip(events_noise, stations_noise, times_noise,
                 phases_noise, amplitudes_noise)),
            dtype=[('event', 'i4'),
                   ('station', 'i4'),
                   ('time', 'datetime64[ns]'),
                   ('phase', 'U1'),
                   ('amplitude', 'f8')])
        noise = pd.DataFrame(noise_np)

    arrivals = pd.concat([at_p_np, at_s_np, noise], ignore_index=True)

    arrivals['station'] = arrivals['station'].map(stations['id'])
    arrivals = arrivals.sample(frac=1).reset_index(drop=True)

    return arrivals
