# %%
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn import mixture

from src.clustering.dataset import PhasePicksDataset
from src.clustering.utils import ClusterStatistics, color_iter, plot_clusters

run_samples = 5
use_columns = ['time', 'dx']
covariance_prior = np.diag([1e-5, 1])
plot_index = 1
plot_x_feat = 0
plot_y_feat = 1


dataset = PhasePicksDataset(
    root_dir='data/raw',
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv'
)

data = dataset[plot_index]
components = len(np.unique(data.y))
data.x['dx'] = PhasePicksDataset.get_distance(data.x)

# %%
for i in range(len(np.unique(data.y))):
    plt.scatter(data.x[data.y == i]['time'],
                data.x[data.y == i]['dx'],
                color=color_iter.__next__())
plt.scatter(data.catalog['time'], PhasePicksDataset.get_distance(data.catalog),
            color='darkorange', marker='x')
plt.show()

# %%
X = data.x[use_columns].to_numpy()

gmm = mixture.GaussianMixture(
    n_components=components,
    covariance_type="full",
    max_iter=500)
gmm.fit(X)
gmm_pred = gmm.predict(X)

plot_clusters(X,
              gmm_pred,
              gmm.means_,
              gmm.covariances_,
              plot_x_feat,
              plot_y_feat,
              0,
              "Gaussian Mixture")

dpgmm = mixture.BayesianGaussianMixture(
    n_components=components,
    weight_concentration_prior=1/components,
    n_init=1,
    max_iter=500,
    covariance_type='full',
    weight_concentration_prior_type='dirichlet_process',
    covariance_prior=covariance_prior
)
dpgmm.fit(X)
dpgmm_pred = dpgmm.predict(X)

plot_clusters(X,
              dpgmm_pred,
              dpgmm.means_,
              dpgmm.covariances_,
              plot_x_feat,
              plot_y_feat,
              1,
              "Bayesian Gaussian Mixture with a Dirichlet process prior",
              )

plt.show()

labels = data.y.to_numpy()
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
gmm_metrics = ClusterStatistics()
dpgmm_metrics = ClusterStatistics()

for i, data in enumerate(tqdm.tqdm(dataset)):
    X = data.x.copy()
    X['dx'] = PhasePicksDataset.get_distance(X)
    X = X[use_columns].to_numpy()

    gmm.fit(X)
    gmm_pred = gmm.predict(X)

    dpgmm.fit(X)
    dpgmm_pred = dpgmm.predict(X)

    labels = data.y.to_numpy()

    gmm_metrics.add(labels, gmm_pred)
    dpgmm_metrics.add(labels, dpgmm_pred)
    if i == 2:
        break

print(f"GMM ARI: {gmm_metrics.ari()}, "
      f"Accuracy: {gmm_metrics.accuracy()}, "
      f"Precision: {gmm_metrics.precision()}, "
      f"Recall: {gmm_metrics.recall()}")
print(f"DPGMM ARI: {dpgmm_metrics.ari()}, "
      f"Accuracy: {dpgmm_metrics.accuracy()}, "
      f"Precision: {dpgmm_metrics.precision()}, "
      f"Recall: {dpgmm_metrics.recall()}")

# %%
