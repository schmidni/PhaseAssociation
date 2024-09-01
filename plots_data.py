# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.spatial import distance_matrix

from src.synthetics.butler_vanaswegen import Butler_VanAswegen_1993
from src.synthetics.seismicity_samples_Dieterich94 import (
    get_rectangular_slippatch_from_FM, get_seismicity_sample_from_Dieterich94,
    plot_rectangular_slippatch, set_bounding_box)

# matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')
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
            edgecolors='#0072B2', facecolors='none', s=markersize)
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
events_per_second.plot(kind='bar', width=1, color='#0072B2')
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

# %%
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
           stations_enu['z'], marker='v', color='r', s=markersize)

XYZ = np.array([catalog['e'], catalog['n'], catalog['u']]).T

# Set plotting box
set_bounding_box(ax, XYZ, 0.4, 99.6)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
ax.tick_params(axis='z', labelsize=ticksize)
plt.show()


# %% Plot Traveltime Noise ###################################################
stations = pd.read_csv('stations/station_cords_blab_VALTER.csv')
stations.rename(columns={'station_code': 'id'}, inplace=True)
stations = stations[['id', 'x', 'y', 'z']]

center = np.array(
    [[stations['x'].mean(), stations['y'].mean(), stations['z'].mean()]])

distances = distance_matrix(
    center, stations[['x', 'y', 'z']].to_numpy())[0]

distances = np.sort(distances)

tt_p = distances/5500
tt_s = distances/2700

noise_p = np.random.normal(0, tt_p*0.05, tt_p.shape)
noise_s = np.random.normal(0, tt_s*0.05, tt_s.shape)

tt_p_n = tt_p + noise_p
tt_s_n = tt_s + noise_s

plt.figure(figsize=(16, 12))
plt.scatter(distances, tt_p_n*1e3, color='#009E73', s=100, label='P-wave')
plt.scatter(distances, tt_s_n*1e3, color='#D55E00', s=100, label='S-wave')
plt.plot(distances, tt_p*1e3, color='#009E73')
plt.plot(distances, tt_s*1e3, color='#D55E00')
plt.xlabel('distance [m]', fontsize=labelsize)
plt.ylabel('traveltime [ms]', fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.legend(fontsize=22)
plt.show()


# %% Plot Amplitude Noise ###################################################
plt.figure(figsize=(16, 12))
magnitudes = [-4, -3, -2, -1]
colors = ['#D55E00', '#0072B2', '#CC79A7', '#009E73']
for i, mag in enumerate(magnitudes):
    gmvs = Butler_VanAswegen_1993(mag, distances)[0]

    er = np.random.normal(0, np.abs(np.log10(gmvs))*0.025, gmvs.shape)
    gmvs_n = gmvs * 10**er

    plt.scatter(distances, gmvs_n, marker='x', color=colors[i],
                s=125, label='noisy')
    plt.plot(distances, gmvs, color=colors[i], label='theoretical')
    plt.annotate(xy=(distances[-1], gmvs[-1]),
                 xytext=(-30, 20),
                 textcoords='offset points',
                 text=f'm={mag}',
                 fontsize=20,
                 annotation_clip=True,
                 color=colors[i])

plt.xlabel('distance [m]', fontsize=labelsize)
plt.ylabel('pgv [m/s]', fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.yscale('log')
plt.xscale('log')
plt.legend(['noisy', 'theoretical'], fontsize=22)
plt.show()

# %%
