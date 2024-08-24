# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.spatial import distance_matrix

from src.synthetics.seismicity_samples_Dieterich94 import (
    get_rectangular_slippatch_from_FM, get_seismicity_sample_from_Dieterich94,
    plot_rectangular_slippatch, set_bounding_box)

# matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')
ticksize = 24
labelsize = 26
markersize = 200

# %%
# FMD Plot ###################################################################
catalog_fmd = pd.read_csv('plots/data/fmd/catalog_0.csv')


# Calculate histogram data
counts, bins = np.histogram(catalog_fmd['magnitude'], bins=30)

# Use the midpoints of the bins for the x-axis
bin_centers = (bins[:-1] + bins[1:]) / 2

# Plot the scatter plot
plt.figure(figsize=(16, 12))
plt.scatter(bin_centers, counts, marker='^',
            edgecolors='blue', facecolors='none', s=markersize)
plt.yscale('log')
# y scale minimum and maximum values
plt.ylim(1, 1e5)
plt.xlabel('Magnitude', fontsize=labelsize)
plt.ylabel('Number of Events', fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.show()

# Rate Plot ##################################################################
catalog_rate = pd.read_csv(
    'plots/data/rate/catalog_0.csv', parse_dates=['time'], index_col=0)
catalog_rate = catalog_rate.set_index('time', drop=True)

events_per_second = catalog_rate.resample('1s').size()

# Step 2: Plot the resampled data
plt.figure(figsize=(16, 12))
events_per_second.plot(kind='bar', width=1, color='blue')
plt.xlabel('Time [min]', fontsize=labelsize)
plt.ylabel('Number of Events', fontsize=labelsize)


def format_func(x, pos):
    total_seconds = int(x)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f'{minutes}:{seconds:02}'


# Apply the formatter to the x-axis
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))

# Adjust ticks to show every 30 seconds
tick_locs = range(0, len(events_per_second), 30)
plt.xticks(ticks=tick_locs, rotation=0)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.show()


# Plot 3D hypocentre distribution & finite slip patch ########################

stations = pd.read_csv('stations/station_cords_blab_VALTER.csv')

stations.rename(columns={'station_code': 'id'}, inplace=True)
stations_enu = stations[['id', 'x', 'y', 'z']]

# Specify reference focal mechanism
stk0 = 45
dip0 = 40

# Specify hypocentral coordinates, magnitude & stressdrop
e0 = stations['x'].mean()
n0 = stations['y'].mean()
u0 = stations['z'].mean()

mag = 1  # Source size
stressdrop = 1e6  # [Pa]

# Plot properties
view_angle = [-3, 90]
plotme = False

finsrc = get_rectangular_slippatch_from_FM(
    e0, n0, u0, stk0, dip0, mag, stressdrop)

# Create synthetic set of FMs
neq = 100  # Number of quakes
cradius = finsrc['length']  # Crack radius, defines cluster size
dx = cradius / 20  # Stdev of scatter around perfect plane

catalog = get_seismicity_sample_from_Dieterich94(
    neq, e0, n0, u0, stk0, dip0, cradius, dx, plotme)

# 3D hypocentre distribution & finite slip patch
fig = plt.figure(204, figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('E', fontsize=labelsize)
ax.set_ylabel('N', fontsize=labelsize)
ax.set_zlabel('Up', fontsize=labelsize)
ax.view_init(view_angle[0], view_angle[1])

# Plot rectangular slip patch
plot_rectangular_slippatch(ax, finsrc, 'm')

# Plot hypocentres
ax.scatter(catalog['e'], catalog['n'], catalog['u'],
           '.', color=[.2, .2, .2], s=markersize)
ax.scatter(stations_enu['x'], stations_enu['y'],
           stations_enu['z'], 'o', color='r', s=markersize)

XYZ = np.array([catalog['e'], catalog['n'], catalog['u']]).T

# Set plotting box
set_bounding_box(ax, XYZ, 0.4, 99.6)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
ax.tick_params(axis='z', labelsize=ticksize)
plt.show()

#
# %% Plot arrivals ###########################################################
stations = pd.read_csv('stations/station_cords_blab_VALTER.csv')
stations.rename(columns={'station_code': 'id'}, inplace=True)
stations = stations[['id', 'x', 'y', 'z']]

arrivals = pd.read_csv(
    'plots/data/arrivals/arrivals_0.csv', parse_dates=['time'])
arrivals['time'] = arrivals['time'] - arrivals['time'].min()
arrivals['time'] = arrivals['time'].dt.microseconds

arrivals = arrivals.join(stations.set_index('id'), on='station')

fig, ax = plt.subplots(figsize=(16, 12))
plt.xlabel('time [μs]', fontsize=labelsize)
plt.ylabel('Station y-coordinate [m]', fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)


def format_func(x, pos):
    return np.round(x, 2)


# Apply the formatter to the x-axis
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))

colors = ['blue', 'green', 'red', 'purple', 'orange',
          'brown', 'pink', 'gray', 'olive', 'black']
for group, df in arrivals.groupby('event'):
    ax.scatter(df['time'], df['y'], marker='o',
               color=colors[group], s=markersize)

fig.show()


# %% Plot Noise ##############################################################

stations = pd.read_csv('stations/station_cords_blab_VALTER.csv')
stations.rename(columns={'station_code': 'id'}, inplace=True)
stations = stations[['id', 'x', 'y', 'z']]

catalog = pd.read_csv(
    'plots/data/rate/catalog_0.csv', parse_dates=['time'], index_col=0)
catalog = catalog.set_index('time', drop=True)

distances = distance_matrix(
    catalog[['e', 'n', 'u']], stations[['x', 'y', 'z']])

tt_p = distances/5500 * 1e3

noise = np.random.normal(0, tt_p*0.1, tt_p.shape)

# plot a histogram of noise values

plt.figure(figsize=(16, 12))
plt.hist(noise.flatten(), bins=200, color='blue')
plt.xlabel('Noise [ms]', fontsize=labelsize)
plt.ylabel('Arrivals', fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.show()

# %%
