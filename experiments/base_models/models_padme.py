# %%
import tqdm

from src import run_phassoc
from src.dataset import (PhasePicksDataset, SeisBenchPickFormat,
                         SeisBenchStationFormat)
from src.metrics import ClusterStatistics
from src.plotting.embeddings import plot_embeddings, plot_embeddings_events

config = {
    'dbscan_eps': 0.2,
    'dbscan_min_samples': 4,
    'vel': {
        'P': 5.4,
        'S': 3.1
    },
    'model': '../../models/model_33k_2',
    'min_picks_per_event': 4
}

statistics = ClusterStatistics()


ds = PhasePicksDataset(
    root_dir='../../data/test_30',
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv',
    transform=SeisBenchPickFormat(),
    station_transform=SeisBenchStationFormat()
)


for sample in tqdm.tqdm(ds):
    _, picks, embeddings = run_phassoc(
        sample.x, ds.stations, config, verbose=True)
    y = sample.y.copy().where(sample.y.map(sample.y.value_counts())
                              > config['min_picks_per_event'], -1)

    statistics.add(y.to_numpy(), picks['labels'].to_numpy())

    x = sample.x.copy()
    x['timestamp'] = x['timestamp'].values.astype('int64')
    x['type'] = x['type'].astype('category').cat.codes
    x = x.drop(columns=['id'])
    plot_embeddings(x.to_numpy(), sample.y.to_numpy(), method='tsne')
    plot_embeddings(embeddings, sample.y.to_numpy(), method='tsne')
    plot_embeddings_events(embeddings, sample.y, sample.catalog.to_numpy())


print(f'PhAssoc ARI: {statistics.ari()}, Accuray: {statistics.accuracy()}, '
      f'Precision: {statistics.precision()}, Recall: {statistics.recall()}')


# %%

# %%
