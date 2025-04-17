import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import pair_confusion_matrix as PCM


class ClusterStatistics:
    def __init__(self):
        self._ari = []
        self._pcm = []
        self._event_confusion = []

    def add(self, labels, labels_pred, cat_true=None, cat_pred=None):

        mask = (labels == -1) & (labels_pred == -1)

        labels = labels[~mask]
        labels[labels == -1] = np.arange(1e6, 1e6+len(labels[labels == -1]))

        labels_pred = labels_pred[~mask]
        labels_pred[labels_pred == -1] = \
            np.arange(2e6, 2e6+len(labels_pred[labels_pred == -1]))

        self._ari.append(ARI(labels, labels_pred))

        self._pcm.append(PCM(labels, labels_pred))

        if cat_true is not None and cat_pred is not None:
            self._event_confusion.append(self.detected(cat_true, cat_pred))

    def detected(self, cat_true, cat_pred):
        min = np.minimum(cat_true['time'].min(), cat_pred['time'].min())
        max = np.maximum(cat_true['time'].max(), cat_pred['time'].max())
        nbins = ((max - min) / 0.1e9).astype(int)
        count_true, _ = np.histogram(
            cat_true['time'], range=(min, max), bins=nbins)
        count_pred, _ = np.histogram(
            cat_pred['time'], range=(min, max), bins=nbins)

        total = np.sum(count_true)
        diff = count_true-count_pred
        fn = np.sum(np.where(diff > 0, diff, 0))
        fp = np.sum(np.where(diff < 0, -diff, 0))
        tp = np.minimum(count_true, count_pred).sum()
        tn = total - tp - fp - fn
        return np.array([[tp, fp], [fn, tn]])

    def event_confusion(self):
        return np.mean(self._event_confusion, axis=0)

    def event_precision(self):
        return np.round(self.event_confusion()[0, 0] /
                        (self.event_confusion()[0, 1]
                         + self.event_confusion()[0, 0]), 3)

    def event_recall(self):
        return np.round(self.event_confusion()[0, 0] /
                        (self.event_confusion()[1, 0]
                         + self.event_confusion()[0, 0]), 3)

    def ari(self):
        return np.round(np.mean(self._ari), 3)

    def pcm(self):
        return np.round(np.mean(self._pcm, axis=0), 3)

    def precision(self):
        return np.round(self.pcm()[1, 1] /
                        (self.pcm()[0, 1] + self.pcm()[1, 1]), 3)

    def recall(self):
        return np.round(self.pcm()[1, 1] /
                        (self.pcm()[1, 0] + self.pcm()[1, 1]), 3)

    def accuracy(self):
        return np.round((self.pcm()[0, 0] +
                         self.pcm()[1, 1]) / self.pcm().sum(), 3)

    def perc_eq(self):
        if not self._perc_eq:
            return None
        # TODO: Better metric
        return np.round(100*(1-np.mean(self._perc_eq)), 3)


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
