import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from src.butler_vanaswegen import Butler_VanAswegen_1993


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
                        percentile: float):
    distances = distance_matrix(
        catalog[['e', 'n', 'u']], stations[['e', 'n', 'u']])

    event_times = np.array(catalog.time.values)

    # arrival times
    at_p = (distances/v_p *
            1e6).astype('timedelta64[ns]') + event_times[:, np.newaxis]
    at_s = (distances/v_s *
            1e6).astype('timedelta64[ns]') + event_times[:, np.newaxis]

    def butler_proxy(distances):
        bva = np.vectorize(Butler_VanAswegen_1993)
        return bva(catalog['magnitude'], distances)[0]
    gmvs = np.apply_along_axis(butler_proxy, 0, distances)

    cutoff = np.percentile(gmvs, percentile)
    detection_mask = gmvs > cutoff
    detection_mask_np = detection_mask.ravel()
    event_indices, station_indices = np.unravel_index(
        np.arange(at_p.size), at_p.shape)

    phase_p = np.full_like(event_indices, 'P', dtype='U1')
    phase_s = np.full_like(event_indices, 'S', dtype='U1')

    at_p_np = np.array(
        list(zip(event_indices, station_indices, at_p.ravel(), phase_p)),
        dtype=[('event', 'i4'),
               ('station', 'i4'),
               ('time', 'datetime64[ns]'),
               ('phase', 'U1')])[detection_mask_np]
    at_p_np = pd.DataFrame(at_p_np)

    at_s_np = np.array(
        list(zip(event_indices, station_indices, at_s.ravel(), phase_s)),
        dtype=[('event', 'i4'),
               ('station', 'i4'),
               ('time', 'datetime64[ns]'),
               ('phase', 'U1')])[detection_mask_np]
    at_s_np = pd.DataFrame(at_s_np)

    arrivals = pd.concat([at_p_np, at_s_np], ignore_index=True)

    arrivals['station'] = arrivals['station'].map(stations['id'])

    return arrivals
