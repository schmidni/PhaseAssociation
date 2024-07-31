# %%
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn import mixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.clustering.utils import ClusterStatistics, load_data

color_iter = itertools.cycle(
    ["navy", "c", "cornflowerblue", "gold", "orange", "green",
     "lime", "red", "purple", "blue", "pink", "brown", "black", "gray",
     "magenta", "cyan", "olive", "maroon", "darkslategray", "darkkhaki"])


def plot_results(X, Y_, means, covariances, x, y, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(
            zip(means, covariances, color_iter)):
        if covar.ndim < 2:
            covar = np.diag(covar)
        # use advanced indexing and broadcasting to select
        # the rows and columns corresponding to x and y
        covar = covar[np.ix_([x, y], [x, y])]
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, x], X[Y_ == i, y], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(
            (mean[x], mean[y]), v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# %%
index = 1
arrivals, catalog, stations, reference_time = load_data(index)

# %%
for i in range(len(np.unique(arrivals['event']))):
    plt.scatter(arrivals[arrivals['event'] == i]['dt'],
                arrivals[arrivals['event'] == i]['dx'],
                color=color_iter.__next__())
plt.scatter(catalog['dt'], catalog['dx'], color='darkorange', marker='x')
plt.show()

# %%
X = arrivals[['dt', 'dx']].to_numpy()

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
# X = standard_scaler.fit_transform(X)
# X = minmax_scaler.fit_transform(X)
# X[:,2] = minmax_scaler.fit_transform(X[:,2].reshape(-1, 1)).flatten()
# X[:,2] = standard_scaler.fit_transform(X[:,2].reshape(-1, 1)).flatten()
x_feat = 0
y_feat = 1
components = 20

gmm = mixture.GaussianMixture(
    n_components=components,
    covariance_type="full",
    max_iter=500)
gmm.fit(X)
gmm_pred = gmm.predict(X)

plot_results(X,
             gmm_pred,
             gmm.means_,
             gmm.covariances_,
             x_feat,
             y_feat,
             0,
             "Gaussian Mixture")

dpgmm = mixture.BayesianGaussianMixture(
    n_components=components,
    weight_concentration_prior=1/components,
    n_init=3,
    max_iter=500,
    covariance_type='full',
    weight_concentration_prior_type='dirichlet_process',
    covariance_prior=np.array([[1e-5, 0], [0, 1]])
)
dpgmm.fit(X)
dpgmm_pred = dpgmm.predict(X)

plot_results(X,
             dpgmm_pred,
             dpgmm.means_,
             dpgmm.covariances_,
             x_feat,
             y_feat,
             1,
             "Bayesian Gaussian Mixture with a Dirichlet process prior",
             )

plt.show()

labels = arrivals['event'].to_numpy()
gmm_metrics = ClusterStatistics()
gmm_metrics.add(labels, gmm_pred)
dpgmm_metrics = ClusterStatistics()
dpgmm_metrics.add(labels, dpgmm_pred)

print(f"GMM ARI: {gmm_metrics.ari()}, "
      f"Precision: {gmm_metrics.precision()}, "
      f"Recall: {gmm_metrics.recall()}")
print(f"DPGMM ARI: {dpgmm_metrics.ari()}, "
      f"Precision: {dpgmm_metrics.precision()}, "
      f"Recall: {dpgmm_metrics.recall()}")

# %%
components = 20

gmm_metrics = ClusterStatistics()
dpgmm_metrics = ClusterStatistics()

for index in range(100):
    arrivals = pd.read_csv(f'data/raw/arrivals_{index}.csv')
    catalog = pd.read_csv(f'data/raw/catalog_{index}.csv', index_col=0)
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

    X = arrivals[['dt', 'dx']].to_numpy()

    gmm.fit(X)
    gmm_pred = gmm.predict(X)
    dpgmm.fit(X)
    dpgmm_pred = dpgmm.predict(X)

    labels = arrivals['event'].to_numpy()

    gmm_metrics.add(labels, gmm_pred)
    dpgmm_metrics.add(labels, dpgmm_pred)


print(f"GMM ARI: {gmm_metrics.ari()}, "
      f"Accuracy: {gmm_metrics.accuracy()}, "
      f"Precision: {gmm_metrics.precision()}, "
      f"Recall: {gmm_metrics.recall()}")
print(f"DPGMM ARI: {dpgmm_metrics.ari()}, "
      f"Accuracy: {dpgmm_metrics.accuracy()}, "
      f"Precision: {dpgmm_metrics.precision()}, "
      f"Recall: {dpgmm_metrics.recall()}")
