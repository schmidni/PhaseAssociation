import math

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def azidip2plane(e0, n0, u0, strike, dip, RS, RD):
    # Compute x/y/z coordinates of point on a plane with given strike and dip,
    # at an along-strike distance RS, and a along-dip distance RD.
    # Input units: angles in degrees, distances in meters

    dipdir = (strike - 90) * np.pi / 180
    dip = dip * np.pi / 180
    strike = strike * np.pi / 180

    # Formulation for E/N/U
    n = n0 + RS * np.cos(strike) + RD * np.cos(dip) * np.cos(dipdir)
    e = e0 + RS * np.sin(strike) + RD * np.cos(dip) * np.sin(dipdir)
    u = u0 + RD * np.sin(dip)

    return e, n, u


def sample_Dieterich94_stress_profile(
        neq, crack_radius, events_inside_crack):
    # Compute stress profile from Dieterich 1994, eq. 20 and randomly sample
    # from it, to simulate aftershock distributions as a function of distance
    # from the main shock hypocentre. Then distribute these distances on a 2D
    # plane, by randomly sampling an azimuth.
    # Men-Andrin Meier, 26/5/2023

    if events_inside_crack is None:
        events_inside_crack = False

    quakes = {}
    quakes['mname'] = 'Dieterich94'
    quakes['crad'] = crack_radius

    # 1. Compute stress profile
    # Equation 20 of Dieterich 1994, JGR: "Simplified representation of stress
    # change from slip on a planar surface in a homogeneous elastic medium"
    # Only defined for x>c
    dx = 0.01
    x = np.arange(0, 10*crack_radius + dx, dx)
    dtau0 = np.zeros_like(x)

    dtauEq = -2
    dtaumax = 5 * abs(dtauEq)

    # Equation only holds for x>c
    # Set stress inside crack to 0:
    #       no events will be sampled from within c
    # Set stress inside crack to taumax:
    #       numerous events sampled from within c
    if events_inside_crack:
        dtau0[x < crack_radius] = dtaumax
    else:
        dtau0[x < crack_radius] = 0

    # Equation 20
    dtau0[x > crack_radius] = -dtauEq * \
        (((1 - (crack_radius ** 3 /
         x[x > crack_radius] ** 3)) ** (-1 / 2)) - 1)

    dtau0[dtau0 > dtaumax] = dtaumax

    # 2. Use stress profile as pdf to sample random main shock / aftershock
    # distances
    # Compute cdf, and use uniformly random distribution [0, 1] to sample on
    # y-axis
    dtaui = np.cumsum(dtau0) * dx
    dtauin = dtaui / (np.sum(dtau0) * dx)

    frnd = np.random.rand(neq)
    ri = np.zeros(neq)
    for irnd in range(neq):
        idx = np.argmin(np.abs(dtauin - frnd[irnd]))
        ri[irnd] = x[idx]

    # Random position on 2D plane with uniformly random azimuth
    azi = 2 * np.pi * np.random.rand(neq)
    xi = ri * np.cos(azi)
    yi = ri * np.sin(azi)

    stress = {}
    stress['x'] = x
    stress['dtau'] = dtau0
    quakes['ri'] = ri
    quakes['xi'] = xi
    quakes['yi'] = yi

    return quakes, stress


def get_seismicity_sample_from_Dieterich94(
        neq, e0, n0, u0, azi, dip, crack_radius, dx):
    # Get samples of quakes from stress profile
    cat, stress = sample_Dieterich94_stress_profile(
        neq, crack_radius, 1)

    # Distribute samples along plane with particular dip and rake
    # Use x- and y-values from sampled seismicity as
    # along-strike and along-dip distances, respectively
    e, n, u = azidip2plane(e0, n0, u0, azi, dip, cat['xi'], cat['yi'])

    # Generated samples are on plane. Add random perturbation to
    # simulate off-plane seismicity
    cat['e'] = e + dx * np.random.randn(neq)
    cat['n'] = n + dx * np.random.randn(neq)
    cat['u'] = u + dx * np.random.randn(neq)

    cat['azi'] = azi
    cat['dip'] = dip
    cat['stress'] = stress

    cat['prop'] = {}
    cat['prop']['tstring'] = \
        'Seismicity sampled from Dieterich 1994 stress ' \
        f'profile, for crack with radius = {crack_radius}m'
    cat['prop']['name'] = 'Dieterich 1994 seismicity'

    return cat


def magnitude2moment(Mw):
    M0 = math.pow(10, (3/2 * (Mw + 6.0666)))  # [Nm]
    return M0


def get_rectangular_slippatch_from_FM(
        e0, n0, u0, strike, dip, mag, stressdrop):
    # Compute square-shaped finite source centered around hypocentre.

    # Rectangular source model used in Meier et al., 2014, JGR
    # estimate source patch dimension L of a square source with the relation
    # for strike-slip faults [Knopoff, 1958; Scholz, 2002]
    # where L is the fault dimension in meters.
    # estimate average slip u over the fault area L , use def of seismic moment
    # [Aki and Richards, 2002]

    # Use the following commands and script for plotting:
    # plot_rectangular_slippatch(src, e0, n0, u0, colour)

    # stressdrop = 1e6;  # [MPa]
    shearmod = 30e9  # [GPa]

    M0 = magnitude2moment(mag)
    L = (2 * M0 / (np.pi * stressdrop)) ** (1/3)  # [m]
    D = M0 / (shearmod * L ** 2)  # [m]

    # Change to radians
    dipdir = (strike - 90) * np.pi / 180
    dip = dip * np.pi / 180
    strike = strike * np.pi / 180

    # Compute 4 corners of square with centre at [e0, n0, u0]
    rs = np.array([-L/2, L/2, L/2, -L/2])
    rd = np.array([-L/2, -L/2, L/2, L/2])

    dx = rs * np.sin(strike) + rd * np.cos(dip) * np.sin(dipdir)
    dy = rs * np.cos(strike) + rd * np.cos(dip) * np.cos(dipdir)
    dz = rd * np.sin(dip)

    src = {}
    src['e4'] = e0 + dx
    src['n4'] = n0 + dy
    src['u4'] = u0 + dz

    # Make closed surface, for plotting with fill3.m
    src['e5'] = np.append(src['e4'], src['e4'][0])
    src['n5'] = np.append(src['n4'], src['n4'][0])
    src['u5'] = np.append(src['u4'], src['u4'][0])

    # Store everything in output structure
    src['e0'] = e0
    src['n0'] = n0
    src['u0'] = u0
    src['mag'] = mag
    src['stk'] = strike * 180 / np.pi
    src['dip'] = dip * 180 / np.pi
    src['dipdir'] = src['stk'] - 90
    src['length'] = L
    src['slip'] = D
    src['stressdrop'] = stressdrop
    src['shearmod'] = shearmod

    return src


def plot_rectangular_slippatch(ax, finsrc, colour=[0, 0, 1]):

    ax.plot(finsrc['e0'], finsrc['n0'],
            finsrc['u0'], 'ok', markerfacecolor='r')

    poly = Poly3DCollection(
        [list(zip(finsrc['e5'], finsrc['n5'], finsrc['u5']))],
        edgecolor=[.2, .2, .2], alpha=0.3, facecolor=colour)

    ax.add_collection3d(poly)


def get_bounding_box(XYZ, plo=0, pup=100):
    if XYZ.shape[1] < 2:
        raise ValueError('XYZ must be a at least 2D array')

    x = XYZ[:, 0]
    y = XYZ[:, 1]
    lim = {'x': [np.percentile(x, plo), np.percentile(x, pup)],
           'y': [np.percentile(y, plo), np.percentile(y, pup)]}

    lim['x'].sort()
    lim['y'].sort()

    if XYZ.shape[1] == 3:
        z = XYZ[:, 2]
        lim['z'] = [np.percentile(z, plo), np.percentile(z, pup)]
        lim['z'].sort()

    return lim


def set_bounding_box(ax, XYZ, plo=0, pup=100):
    """
    Find and set x/y/z limits to include all data points that lie between
    the <plo> and <pup> percentiles

    Examples: include all data points from 1st to 99th percentile
    2D: set_bounding_box([x.val,y.val], 1, 99)
    3D: set_bounding_box([x.val,y.val,z.val], 1, 99)
    """

    ndim = XYZ.shape[1]

    lim = get_bounding_box(XYZ, plo, pup)

    if ndim == 2:
        ax.set_xlim(lim['x'])
        ax.set_ylim(lim['y'])

    elif ndim == 3:
        ax.set_xlim(lim['x'])
        ax.set_ylim(lim['y'])
        ax.set_zlim(lim['z'])

    else:
        raise ValueError('Works only in 2D or 3D')

    lim['plo'] = plo
    lim['pup'] = pup

    return lim


def plot_3D(finsrc, catalog, stations_enu):

    view_angle = [-3, 90]
    # 3D hypocentre distribution & finite slip patch
    fig = plt.figure(204)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('E')
    ax.set_ylabel('N')
    ax.set_zlabel('Up')
    ax.view_init(view_angle[0], view_angle[1])

    # Plot rectangular slip patch
    plot_rectangular_slippatch(ax, finsrc, 'm')

    # Plot hypocentres
    ax.scatter(catalog['e'], catalog['n'],
               catalog['u'], '.', color=[.2, .2, .2])
    ax.scatter(stations_enu['x'], stations_enu['y'],
               stations_enu['z'], 'o', color='r')

    XYZ = np.array([catalog['e'], catalog['n'], catalog['u']]).T

    # Set plotting box
    set_bounding_box(ax, XYZ, 0.4, 99.6)

    plt.show()
