import sys
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.dates import DateFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# Load CSV data
# picks_df = pd.read_csv('/Users/ltian/Project/dl-microquake/results/2024-08-29T184610---2024-08-29T184650/V*_Sp0.15_Ss0.30.csv')
picks_df = pd.read_csv(
    '2024-04-30T040909-T040910_V__Sp0.15_Ss0.30.csv')
station_coords_df = pd.read_csv('station_cords_blab_VALTER.csv')
reference_df = pd.read_csv('sc_fdsn_cat_M0a.csv')

# Define thresholds for plotting
p_threshold_low = 0
s_threshold_low = 0

# Function to safely convert strings to UTCDateTime, handling errors


def safe_convert_to_utc(time_str):
    try:
        return UTCDateTime(time_str)
    except Exception:
        return pd.NaT


def create_client(url):
    parsed = urlparse(url)
    base_url = parsed._replace(
        netloc=f"{parsed.hostname}:{parsed.port}").geturl()
    if parsed.username and parsed.password:
        print(f"Connecting to {base_url} with credentials", file=sys.stderr)
        return Client(base_url=base_url, user=parsed.username, password=parsed.password)
    else:
        print(f"Connecting to {base_url}", file=sys.stderr)
        return Client(base_url=base_url)


# Convert peak_time and reference times to UTCDateTime objects
picks_df['onset_time'] = picks_df['onset_time'].apply(safe_convert_to_utc)
reference_df['reference_time'] = reference_df['time'].apply(
    safe_convert_to_utc)

# Drop rows where conversion was unsuccessful
picks_df.dropna(subset=['onset_time'], inplace=True)
reference_df.dropna(subset=['reference_time'], inplace=True)

loc_main = [-100.707280261442000, -57.234644129406700, -
            82.839036000000100]  # x, y, z, M0a

# remove picks from certain bad stations
# picks_df = picks_df[~picks_df['trace_id'].str.contains('V0506')] # example of M0b
# picks_df = picks_df[~picks_df['trace_id'].str.contains('V0305')] # example M0a

# Define your custom time window
custom_start_time = UTCDateTime("2024-04-30T04:09:09")  # M0a
# custom_start_time = UTCDateTime("2024-08-29T18:46:10") # M0b
dur = 0.2

while custom_start_time < UTCDateTime("2024-04-30T04:09:10"):
    # while custom_start_time < UTCDateTime("2024-08-29T18:46:50"):
    custom_end_time = custom_start_time + dur

    # Filter picks_df and reference_df to include only data within the custom time window
    picks_df_cut = picks_df[
        (picks_df['onset_time'] >= custom_start_time) & (picks_df['onset_time'] <= custom_end_time)].copy()
    reference_df_cut = reference_df[
        (reference_df['reference_time'] >= custom_start_time) & (reference_df['reference_time'] <= custom_end_time)].copy()

    # Convert onset_time to pandas datetime format before storing in the DataFrame
    picks_df_cut['peak_datetime'] = picks_df_cut['onset_time'].apply(
        lambda x: pd.to_datetime(str(x), utc=True))
    reference_df_cut['reference_datetime'] = reference_df_cut['reference_time'].apply(
        lambda x: pd.to_datetime(str(x), utc=True))
    # picks_df_cut.loc[:, 'peak_datetime'] = picks_df_cut['onset_time'].apply(lambda x: x.datetime)

    # Construct trace_id in station_coords_df for mapping
    station_coords_df['trace_id'] = station_coords_df['network_code'] + '.' + \
        station_coords_df['station_code'] + \
        '..'+station_coords_df['chanel_code']
    station_coords_df['dist'] = ((station_coords_df['x'] - loc_main[0])**2 + (
        station_coords_df['y'] - loc_main[1])**2 + (station_coords_df['z'] - loc_main[2])**2)**0.5

    # Create a mapping from trace_id to elevation
    # trace_id_to_elevation = pd.Series(station_coords_df.elevation.values,index=station_coords_df.trace_id).to_dict()
    # Create a mapping from trace_id to distance from the main shock
    trace_id_to_dist = pd.Series(
        station_coords_df.dist.values, index=station_coords_df.trace_id).to_dict()

    # Sort trace_ids based on their corresponding elevation
    # sorted_trace_ids = sorted(picks_df['trace_id'].unique(), key=lambda x: trace_id_to_elevation.get(x, 0))
    # Sort trace_ids based on their corresponding distance
    sorted_trace_ids = sorted(
        picks_df_cut['trace_id'].unique(), key=lambda x: trace_id_to_dist.get(x, 0))

    # Assign station index based on sorted trace_ids
    station_indices = {trace_id: index for index,
                       trace_id in enumerate(sorted_trace_ids)}
    picks_df_cut.loc[:, 'station_index'] = picks_df_cut['trace_id'].map(
        station_indices)

    # Create a mapping from trace_id to station_name
    trace_id_to_station_name = pd.Series(
        station_coords_df.station_code.values, index=station_coords_df.trace_id).to_dict()

    # Map trace_id to elevation in picks_df
    # picks_df['elevation'] = picks_df['trace_id'].map(trace_id_to_elevation)
    # Map trace_id to distance in picks_df
    picks_df_cut.loc[:, 'dist'] = picks_df_cut['trace_id'].map(
        trace_id_to_dist)

    # Filter picks within distance
    picks_df_cut = picks_df_cut[picks_df_cut['dist'] <= 42]

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 7), dpi=200)

    # Normalize peak values for color mapping
    norm_p = Normalize(vmin=0, vmax=1)
    norm_s = Normalize(vmin=0, vmax=1)

    # Filter picks for P and S based on thresholds
    # p_picks_filtered = picks_df[(picks_df['phase'] == 'P') & (picks_df['peak_value'] >= p_threshold_low) & (picks_df['peak_value'] <= 0.98)]
    # s_picks_filtered = picks_df[(picks_df['phase'] == 'S') & (picks_df['peak_value'] >= s_threshold_low) & (picks_df['peak_value'] <= 0.98)]
    # select picks within 50m
    p_picks_filtered = picks_df_cut[(picks_df_cut['phase'] == 'P')]
    s_picks_filtered = picks_df_cut[(picks_df_cut['phase'] == 'S')]

    # Scatter plot for P and S picks with elevation as y-values
    # sc_p = ax.scatter(p_picks_filtered['peak_datetime'], p_picks_filtered['elevation'],
    sc_p = ax.scatter(p_picks_filtered['peak_datetime'], p_picks_filtered['dist'],
                      # c=p_picks_filtered['peak_value'],
                      cmap='Blues', norm=norm_p, label='P-wave Picks', alpha=0.8)
    # sc_s = ax.scatter(s_picks_filtered['peak_datetime'], s_picks_filtered['elevation'],
    sc_s = ax.scatter(s_picks_filtered['peak_datetime'], s_picks_filtered['dist'],
                      # c=s_picks_filtered['peak_value'],
                      cmap='Reds', norm=norm_s, label='S-wave Picks', alpha=0.4)

    # Combo with time seris plot with waveform (optional)
    # Load waveform
    client = create_client(
        "http://user:BedrettoLab2017!@bedretto-experiment.ethz.ch:8081")
    st = client.get_waveforms(
        "8R", "V0305", "*", "JDD", custom_start_time, custom_end_time)
    # Process waveform
    trace = st[0]  # Assuming only one trace is returned
    time_array = np.linspace(custom_start_time.timestamp,
                             custom_end_time.timestamp, trace.stats.npts)
    # Convert to pandas datetime
    time_array = pd.to_datetime(time_array, unit='s', utc=True)
    waveform = trace.data
    waveform = (waveform - np.min(waveform)) / \
        (np.max(waveform) - np.min(waveform))  # Normalize to [0,1]
    waveform = waveform * 10  # Scale for visibility
    # Overlay waveform on scatter plot
    ax.plot(time_array, waveform + 5, color='black', alpha=0.7,
            linewidth=1)  # Shift waveform vertically

    # Axis labels and title
    ax.set_xlabel('Time')
    # ax.set_ylabel('Elevation (m)')
    ax.set_ylabel('Distance from the main shock (m)')

    # Get unique elevations and corresponding labels
    # unique_elevations = picks_df['elevation'].unique()
    unique_distances = picks_df_cut['dist'].unique()
    # unique_labels = [f"{trace_id_to_elevation[trace_id]:.2f}({trace_id_to_station_name[trace_id]})" for trace_id in picks_df['trace_id'].unique()]
    unique_labels = [
        f"{trace_id_to_dist[trace_id]:.2f}({trace_id_to_station_name[trace_id]})" for trace_id in picks_df_cut['trace_id'].unique()]
    # Set y-ticks and their labels
    # ax.set_yticks(unique_elevations)
    ax.set_yticks(unique_distances)
    ax.set_yticklabels(unique_labels, fontsize=8)
    for _, row in reference_df_cut.iterrows():
        ax.axvline(x=row['reference_datetime'],
                   color='green', linestyle='--', alpha=0.5)

    # ax.set_title('P and S Picks by Station Elevation on '+str(custom_start_time.date)+' by QuakePhase and SeisComp')
    ax.set_title('P and S Picks by Station Distance to the mainshock on ' +
                 str(custom_start_time.date)+' by DL Picker and SeisComp')
    ax.legend()

    # Add colorbars for P and S picks (if any)
    # cbar_p = plt.colorbar(ScalarMappable(norm=norm_p, cmap='Blues'), ax=ax)
    # cbar_p.set_label('P Pick Value')
    # divider = make_axes_locatable(ax)
    # cax_s = divider.append_axes("right", size="2%", pad=0.1)
    # cbar_s = plt.colorbar(ScalarMappable(norm=norm_s, cmap='Reds'), cax=cax_s)
    # cbar_s.set_label('S Pick Value')

    plt.tight_layout()
    # plt.savefig(f'/Users/ltian/Bedretto/figs/multi_dl_picks/dl_{dur}s_picks_by_station_dist_{custom_start_time}.png')
    plt.show()

    custom_start_time += dur
