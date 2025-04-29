import itertools

import matplotlib.pyplot as plt
import numpy as np

color_iter = itertools.cycle(
    ["navy", "c", "cornflowerblue", "gold", "orange", "green",
     "lime", "red", "purple", "blue", "pink", "brown", "gray",
     "magenta", "cyan", "olive", "maroon", "darkslategray", "darkkhaki"])


def plot_arrivals(arrivals, cat, cat_pred, labels, labels_pred):
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
