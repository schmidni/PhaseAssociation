# %%
import tqdm

from src import associate_phassoc
from src.dataset import (PhasePicksDataset, SeisBenchPickFormat,
                         SeisBenchStationFormat)
from src.metrics import ClusterStatistics

config = {
    'dbscan_eps': 0.2,
    'dbscan_min_samples': 4,
    'vel': {
        'P': 5.4,
        'S': 3.1
    },
    'model': '../../model/model_m1',
    'min_picks_per_event': 4
}

statistics = ClusterStatistics()


ds = PhasePicksDataset(
    root_dir='../../data/test_2',
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv',
    transform=SeisBenchPickFormat(),
    station_transform=SeisBenchStationFormat()
)


for sample in tqdm.tqdm(ds):
    picks, _ = associate_phassoc(sample.x, ds.stations, config, verbose=True)
    y = sample.y.where(sample.y.map(sample.y.value_counts())
                       > config['min_picks_per_event'], -1)

    statistics.add(y.to_numpy(), picks['labels'].to_numpy())

print(f'PhAssoc ARI: {statistics.ari()}, Accuray: {statistics.accuracy()}, '
      f'Precision: {statistics.precision()}, Recall: {statistics.recall()}')
