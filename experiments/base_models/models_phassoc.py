# %% Imports and Configuration
import itertools

import tqdm

from src import associate_phassoc
from src.dataset import (PhasePicksDataset, SeisBenchPickFormat,
                         SeisBenchStationFormat)
from src.metrics import ClusterStatistics
from src.runners import run_phassoc

# %% Run GaMMA
statistics = ClusterStatistics()


ds = PhasePicksDataset(
    root_dir='../../data/test_10',
    stations_file='stations.csv',
    file_mask='arrivals_*.csv',
    catalog_mask='catalog_*.csv',
    transform=SeisBenchPickFormat(),
    station_transform=SeisBenchStationFormat()
)

model = '../../model/model_m1'

for sample in tqdm.tqdm(itertools.islice(ds, len(ds))):
    _, embeddings = run_phassoc(sample.x, ds.stations, model)

    statistics.add_embedding(embeddings, sample.y.to_numpy())

print(f"PhAssoc ARI: {statistics.ari()}, Accuray: {statistics.accuracy()}, "
      f"Precision: {statistics.precision()}, Recall: {statistics.recall()}")
