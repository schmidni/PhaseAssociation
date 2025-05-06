# %%
import multiprocessing
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import seaborn as sns
import seisbench.models as sbm
import torch
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from pyproj import CRS, Transformer
from tqdm import tqdm

sns.set_theme(font_scale=1.2)
sns.set_style("ticks")

# %%
# Projections
wgs84 = CRS.from_epsg(4326)
local_crs = CRS.from_epsg(9155)  # SIRGAS-Chile 2016 / UTM zone 19S
transformer = Transformer.from_crs(wgs84, local_crs)

client = Client("GFZ")

t0 = UTCDateTime("2014/05/01 00:00:00")
t1 = t0 + 12 * 60 * 60
# t1 = t0 + 24 * 60 * 60   # Full day, requires more memory
stream = client.get_waveforms(network="CX",
                              station="*",
                              location="*",
                              channel="HH?",
                              starttime=t0,
                              endtime=t1)

inv = client.get_stations(network="CX",
                          station="*",
                          location="*",
                          channel="HH?",
                          starttime=t0,
                          endtime=t1)

# %%
picker = sbm.PhaseNet.from_pretrained("instance")

if torch.cuda.is_available():
    picker.cuda()

# We tuned the thresholds a bit - Feel free to play around with these values
picks = picker.classify(stream, batch_size=256,
                        P_threshold=0.075, S_threshold=0.1).picks

Counter([p.phase for p in picks])  # Output number of P and S picks


# %%
pick_df = []
for p in picks:
    pick_df.append({
        "id": p.trace_id,
        "timestamp": p.peak_time.datetime,
        "prob": p.peak_value,
        "type": p.phase.lower()
    })
pick_df = pd.DataFrame(pick_df)

station_df = []
for station in inv[0]:
    station_df.append({
        "id": f"CX.{station.code}.",
        "longitude": station.longitude,
        "latitude": station.latitude,
        "elevation(m)": station.elevation
    })
station_df = pd.DataFrame(station_df)

station_df["x(km)"] = station_df.apply(lambda x: transformer.transform(
    x["latitude"], x["longitude"])[0] / 1e3, axis=1)
station_df["y(km)"] = station_df.apply(lambda x: transformer.transform(
    x["latitude"], x["longitude"])[1] / 1e3, axis=1)
station_df["z(km)"] = - station_df["elevation(m)"] / 1e3

northing = {station: y for station, y in zip(
    station_df["id"], station_df["y(km)"])}
station_dict = {station: (x, y) for station, x, y in zip(
    station_df["id"], station_df["x(km)"], station_df["y(km)"])}
pick_df.sort_values("timestamp")

# %%
config_gamma = {
    "ncpu": multiprocessing.cpu_count()-1,
    "dims": ['x(km)', 'y(km)', 'z(km)'],  # needs to be *(km), column names
    "use_amplitude": False,
    "vel": {"p": 7.0, "s": 7.0 / 1.75},
    "method": "BGMM",
    "oversample_factor": 4,  # factor on the number of initial clusters
    "z(km)": (0, 150),
    "covariance_prior": [1e-4, 1e-2],  # time, amplitude
    "bfgs_bounds": (    # bounds in km
        (249, 601),        # x
        (7199, 8001),        # y
        (0, 151),        # depth
        (None, None),   # t
    ),
    "use_dbscan": True,
    "dbscan_eps": 25,  # seconds
    "dbscan_min_samples": 3,

    "min_picks_per_eq": 5,
    "max_sigma11": 2.0,
    "max_sigma22": 1.0,
    "max_sigma12": 1.0
}
