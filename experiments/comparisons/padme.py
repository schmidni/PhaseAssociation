# %% Imports and Configuration
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import (PhasePicksDataset, SeisBenchPickFormat,
                         SeisBenchStationFormat)
from src.evaluate import evaluate_padme

# %%
folders = ['test_0.1_20',
           'test_0.5_8',
           'test_1_2',
           'test_5_1',
           'test_10_1',
           'test_25_0.5']


harpa = []
for file in folders:
    ds = PhasePicksDataset(
        root_dir=f'../../data/{file}',
        stations_file='stations.csv',
        file_mask='arrivals_*.csv',
        catalog_mask='catalog_*.csv',
        transform=SeisBenchPickFormat(),
        station_transform=SeisBenchStationFormat()
    )

    print('padme --------------------------------')
    ev_padme = evaluate_padme(ds)
    harpa.append((ev_padme.precision(), ev_padme.recall()))

# %%
label = [0.1, 0.5, 1, 5, 10, 25]

# bar plot showing precision and recall for each label


def plot_bar(data, label, title):
    fig, ax = plt.subplots()
    x = np.arange(len(label))
    width = 0.35
    ax.bar(x - width/2, [d[0] for d in data], width, label='Precision')
    ax.bar(x + width/2, [d[1] for d in data], width, label='Recall')
    ax.set_xticks(x)
    ax.set_xticklabels(label)
    ax.set_xlabel('Events per Second')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    plt.show()


plot_bar(harpa, label, 'PADME')

# %%
