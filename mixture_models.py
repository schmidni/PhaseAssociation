# %%
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn import mixture
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import pair_confusion_matrix as PCM
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# %%
index = 5

arrivals = pd.read_csv(f'data/raw/arrivals_{index}.csv')
catalog = pd.read_csv(f'data/raw/catalog_{index}.csv', index_col=0)
catalog['time'] = pd.to_datetime(catalog['time'])

reference_time = catalog['time'].min().floor('min')

catalog['dt'] = (catalog['time'] - reference_time).dt.total_seconds() * 1000
catalog['dx'] = np.sqrt(catalog['e']**2 + catalog['n']**2 + catalog['u']**2)

# %%
arrivals['time'] = pd.to_datetime(arrivals['time'])
arrivals['dt'] = (arrivals['time'] - reference_time).dt.total_seconds() * 1000
arrivals['dx'] = np.sqrt(
    arrivals['e']**2 + arrivals['n']**2 + arrivals['u']**2)


# %%
plt.scatter(arrivals['dt'], arrivals['dx'])
plt.scatter(catalog['dt'], catalog['dx'])
plt.show()

# %%
color_iter = itertools.cycle(
    ["navy", "c", "cornflowerblue", "gold", "darkorange", "green",
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
X = arrivals[['dt', 'dx', 'amplitude']].to_numpy()

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
# X = standard_scaler.fit_transform(X)
# X = minmax_scaler.fit_transform(X)
# X[:,2] = minmax_scaler.fit_transform(X[:,2].reshape(-1, 1)).flatten()
# X[:,2] = standard_scaler.fit_transform(X[:,2].reshape(-1, 1)).flatten()
x_feat = 0
y_feat = 1

real_n_components = len(catalog)
components = 20

gmm = mixture.GaussianMixture(
    # n_components=components,
    n_components=real_n_components,
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
    weight_concentration_prior_type='dirichlet_distribution',
    covariance_prior=np.array([[1e-5, 0, 0], [0, 1, 0], [0, 0, 1e-5]])
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

# %%
labels = arrivals['event'].to_numpy()


gmm_ARI = ARI(labels, gmm_pred)
dpgmm_ARI = ARI(labels, dpgmm_pred)

# TN = 0,0, FP = 0,1, FN = 1,0, TP = 1,1
# Precision = TP / (FP + TP), Recall = TP / (FN + TP)
gmm_PCM = PCM(labels, gmm_pred)
gmm_precision = gmm_PCM[1, 1] / (gmm_PCM[0, 1] + gmm_PCM[1, 1])
gmm_recall = gmm_PCM[1, 1] / (gmm_PCM[1, 0] + gmm_PCM[1, 1])

dpgmm_PCM = PCM(labels, dpgmm_pred)
dpgmm_precision = dpgmm_PCM[1, 1] / (dpgmm_PCM[0, 1] + dpgmm_PCM[1, 1])
dpgmm_recall = dpgmm_PCM[1, 1] / (dpgmm_PCM[1, 0] + dpgmm_PCM[1, 1])


print(f"GMM ARI: {gmm_ARI}, "
      f"Precision: {gmm_precision}, Recall: {gmm_recall}")
print(f"DPGMM ARI: {dpgmm_ARI}, "
      f"Precision: {dpgmm_precision}, Recall: {dpgmm_recall}")


# %%
components = 20

gmm_metrics = []
dpgmm_metrics = []

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

    gmm_ARI = ARI(labels, gmm_pred)
    dpgmm_ARI = ARI(labels, dpgmm_pred)

    # TN = 0,0, FP = 0,1, FN = 1,0, TP = 1,1
    # Precision = TP / (FP + TP), Recall = TP / (FN + TP)
    gmm_PCM = PCM(labels, gmm_pred)
    gmm_precision = gmm_PCM[1, 1] / (gmm_PCM[0, 1] + gmm_PCM[1, 1])
    gmm_recall = gmm_PCM[1, 1] / (gmm_PCM[1, 0] + gmm_PCM[1, 1])

    dpgmm_PCM = PCM(labels, dpgmm_pred)
    dpgmm_precision = dpgmm_PCM[1, 1] / (dpgmm_PCM[0, 1] + dpgmm_PCM[1, 1])
    dpgmm_recall = dpgmm_PCM[1, 1] / (dpgmm_PCM[1, 0] + dpgmm_PCM[1, 1])

    gmm_metrics.append({
        'ARI': ARI(labels, gmm_pred),
        'Precision': gmm_precision,
        'Recall': gmm_recall
    })
    dpgmm_metrics.append({
        'ARI': ARI(labels, dpgmm_pred),
        'Precision': dpgmm_precision,
        'Recall': dpgmm_recall
    })

gmm_avg_metrics = {
    'ARI': np.mean([m['ARI'] for m in gmm_metrics]),
    'Precision': np.mean([m['Precision'] for m in gmm_metrics]),
    'Recall': np.mean([m['Recall'] for m in gmm_metrics])
}

dpgmm_avg_metrics = {
    'ARI': np.mean([m['ARI'] for m in dpgmm_metrics]),
    'Precision': np.mean([m['Precision'] for m in dpgmm_metrics]),
    'Recall': np.mean([m['Recall'] for m in dpgmm_metrics])
}

print(f"GMM ARI: {gmm_avg_metrics['ARI']}, "
      f"Precision: {gmm_avg_metrics['Precision']}, "
      f"Recall: {gmm_avg_metrics['Recall']}")
print(f"DPGMM ARI: {dpgmm_avg_metrics['ARI']}, "
      f"Precision: {dpgmm_avg_metrics['Precision']}, "
      f"Recall: {dpgmm_avg_metrics['Recall']}")

# %%
