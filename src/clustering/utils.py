import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import pair_confusion_matrix as PCM


class ClusterStatistics:
    def __init__(self):
        self._ari = []
        self._pcm = []
        self._perc_eq = []

    def add(self, labels, labels_pred, n_true=None, n_pred=None):
        self._ari.append(ARI(labels, labels_pred))
        self._pcm.append(PCM(labels, labels_pred))
        if n_true and n_pred:
            self._perc_eq.append(abs(n_true-n_pred) / n_true)

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


def gamma_preprocess(arrivals, stations):
    stn = stations.copy()
    stn[['e', 'n', 'u']] = stn[['e', 'n', 'u']] / 1000
    stn['u'] = stn['u'] * -1
    stn = stn.rename(
        columns={'e': 'x(km)', 'n': 'y(km)', 'u': 'z(km)'})
    stn = stn[['id', 'x(km)', 'y(km)', 'z(km)']]

    arr = arrivals.copy()
    arr = arr.rename(columns={'station': 'id',
                              'time': 'timestamp',
                              'phase': 'type',
                              'amplitude': 'amp'})
    arr = arr[['id', 'timestamp', 'type', 'amp']]
    arr['prob'] = 1

    return arr, stn


def load_data(index):
    arrivals = pd.read_csv(f'data/raw/arrivals_{index}.csv')
    catalog = pd.read_csv(f'data/raw/catalog_{index}.csv', index_col=0)
    stations = pd.read_csv('data/raw/stations.csv')
    catalog['time'] = pd.to_datetime(catalog['time'])

    reference_time = catalog['time'].min().floor('min')

    catalog['dt'] = (catalog['time'] -
                     reference_time).dt.total_seconds() * 1000
    catalog['dx'] = np.sqrt(
        catalog['e']**2 + catalog['n']**2 + catalog['u']**2)

    arrivals['time'] = pd.to_datetime(arrivals['time'])
    arrivals['dt'] = (arrivals['time'] -
                      reference_time).dt.total_seconds() * 1000
    arrivals['dx'] = np.sqrt(
        arrivals['e']**2 + arrivals['n']**2 + arrivals['u']**2)

    return arrivals, catalog, stations, reference_time


def plot_arrivals(arrivals, cat, cat_pred, labels, labels_pred):
    fig, ax = plt.subplots(2, sharex=True, figsize=(7, 5))
    # fig.suptitle('Vertically stacked subplots')

    colors = ["darkkhaki", "c", "cornflowerblue", "gold", "green",
              "lime", "red", "purple", "blue", "pink", "brown", "gray",
              "magenta", "cyan", "olive", "maroon", "darkslategray"]

    # prediction
    for idx in range(len(np.unique(labels_pred))):
        ax[1].scatter(arrivals.loc[labels_pred == idx, 'dt'] / 1000,
                      arrivals.loc[labels_pred == idx, 'dx'],
                      color=colors[idx], s=80
                      )

    ax[1].scatter(arrivals.loc[labels_pred == -1, 'dt'] / 1000,
                  arrivals.loc[labels_pred == -1, 'dx'],
                  color='black', s=20)
    ax[1].scatter(cat_pred['dt'] / 1000, cat_pred['dx'],
                  color='darkorange', marker='x')

    # truth
    for idx in range(len(np.unique(labels))):
        ax[0].scatter(arrivals.loc[labels == idx, 'dt'] / 1000,
                      arrivals.loc[labels == idx, 'dx'],
                      color=colors[-idx+1], s=80
                      )
    ax[0].scatter(cat['dt'] / 1000, cat['dx'], color='darkorange', marker='x')
