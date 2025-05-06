import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

color_iter = itertools.cycle(
    ["navy", "c", "cornflowerblue", "gold", "orange", "green",
     "lime", "red", "purple", "blue", "pink", "brown", "gray",
     "magenta", "cyan", "olive", "maroon", "darkslategray", "darkkhaki"])


def plot_arrivals_comparison(arrivals, cat, cat_pred, labels, labels_pred):
    fig, ax = plt.subplots(2, sharex=True, figsize=(14, 10))

    for idx in range(len(np.unique(labels_pred))):
        ax[1].scatter(arrivals.loc[labels_pred == idx, 'time'],
                      arrivals.loc[labels_pred == idx, 'dx'],
                      color=color_iter.__next__(), s=100
                      )

    ax[1].scatter(arrivals.loc[labels_pred == -1, 'time'],
                  arrivals.loc[labels_pred == -1, 'dx'],
                  color='black', s=100)
    ax[1].scatter(cat_pred['time'], cat_pred['dx'],
                  color='darkorange', marker='x', s=80)

    # truth
    if labels:
        for idx in range(len(np.unique(labels))):
            ax[0].scatter(arrivals.loc[labels == idx, 'time'],
                          arrivals.loc[labels == idx, 'dx'],
                          color=color_iter.__next__(), s=100
                          )
        if cat:
            ax[0].scatter(cat['time'], cat['dx'],
                          color='darkorange', marker='x', s=80)
        ax[0].scatter(arrivals.loc[labels == -1, 'time'],
                      arrivals.loc[labels == -1, 'dx'],
                      color='black', s=100)
    return (fig, ax)


def plot_arrivals(arrivals,
                  labels_pred,
                  stations,
                  events=None,
                  title=None,
                  xlim=[-0.05, 0.05,],
                  ylim=[-10, 10],
                  lim=True):

    arrivals = arrivals.copy()
    labels_pred = labels_pred.copy()
    stations = stations.copy()
    events = events.copy() if events is not None else None

    starttime = arrivals['timestamp'].min().as_unit('ns')

    stations[['x(km)', 'y(km)', 'z(km)']] = \
        stations[['x(km)', 'y(km)', 'z(km)']]*1e3

    arrivals = arrivals.rename(columns={'timestamp': 'time', 'id': 'station'})
    arrivals['time'] = arrivals['time'] - starttime
    arrivals['time'] = arrivals['time'].dt.total_seconds()
    arrivals = arrivals.join(stations.set_index('id'), on='station')
    arrivals = pd.concat([arrivals, pd.Series(labels_pred, name='label')],
                         axis=1)

    if events is not None:
        events[['x(km)', 'y(km)', 'z(km)']] = \
            events[['x(km)', 'y(km)', 'z(km)']]*1e3
        events['time'] = pd.to_datetime(events['time'], unit='ns') - starttime
        events['time'] = events['time'].dt.total_seconds()

    plot_axis = 'y(km)'
    fig, ax = plt.subplots(figsize=(16, 12))
    plt.suptitle(
        f'n_events = {len(np.unique(labels_pred))-1}', fontsize=25, y=0.02)
    if title is not None:
        plt.title(title, fontsize=25)
    plt.xlabel('time [ms]', fontsize=20)
    plt.ylabel(f'Station {plot_axis[0]}-coordinate [m]', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if lim:
        plt.xlim(arrivals['time'].min()+xlim[0],
                 arrivals['time'].max()+xlim[1])
        plt.ylim(arrivals[plot_axis].min()+ylim[0],
                 arrivals[plot_axis].max()+ylim[1])

    color_iter = itertools.cycle(
        ["c", "cornflowerblue", "gold", "orange", "green",
         "lime", "red", "purple", "blue", "pink", "brown", "gray",
         "magenta", "cyan", "olive", "maroon", "navy", "darkslategray",
         "darkkhaki"])

    for group, df in arrivals.groupby('label'):
        if group == -1:
            color = 'black'
        else:
            color = color_iter.__next__()

        if group != -1 and events is not None:
            ev = events[events['event_index'] == group]
            ax.scatter(ev['time'], ev[plot_axis],
                       marker='x', color=color, s=200, linewidths=8)

        ax.scatter(df['time'], df[plot_axis],
                   marker='o', color=color, s=100,)

    plt.show()
