# %% Imports and Configuration
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import (PhasePicksDataset, SeisBenchPickFormat,
                         SeisBenchStationFormat)
from src.evaluate import evaluate_gamma, evaluate_harpa, evaluate_pyocto

# %%
# noise
folders = ['test_10_0.1',
           'test_10_0.3',
           'test_10_0.5',
           'test_10_0.7',
           'test_10_0.9']
# folders = ['test_0.1_20',
#            'test_0.5_8',
#            'test_1_2',
#            'test_5_1',
#            'test_10_1',
#            'test_25_0.5']
# folders = ['test_2_0.05',
#            'test_2_0.1',
#            'test_2_0.15',
#            'test_2_0.2',
#            'test_2_0.25',
#            'test_2_0.3']


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

    print('harpa --------------------------------')
    ev_harpa = evaluate_harpa(ds)
    harpa.append((ev_harpa.precision(), ev_harpa.recall()))

pyocto = []
for file in folders:
    ds = PhasePicksDataset(
        root_dir=f'../../data/{file}',
        stations_file='stations.csv',
        file_mask='arrivals_*.csv',
        catalog_mask='catalog_*.csv',
        transform=SeisBenchPickFormat(),
        station_transform=SeisBenchStationFormat()
    )

    print('pyocto --------------------------------')
    ev_pyocto = evaluate_pyocto(ds)
    pyocto.append((ev_pyocto.precision(), ev_pyocto.recall()))

gamma = []
for file in folders:
    ds = PhasePicksDataset(
        root_dir=f'../../data/{file}',
        stations_file='stations.csv',
        file_mask='arrivals_*.csv',
        catalog_mask='catalog_*.csv',
        transform=SeisBenchPickFormat(),
        station_transform=SeisBenchStationFormat()
    )

    print('gamma --------------------------------')
    ev_gamma = evaluate_gamma(ds)
    gamma.append((ev_gamma.precision(), ev_gamma.recall()))

# %% Plotting

names = ['10%',
         '30%',
         '50%',
         '70%',
         '90%']
title = ' for Different Amount of Noise Picks'
xlabel = 'Amount of Noise Picks'


def plot_comparison(gamma, harpa, pyocto, idx):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(gamma))

    # Plotting the precision and recall for each method
    ax.bar(x - 0.2, [g[idx] for g in gamma], 0.2, label='Gamma', color='blue')
    ax.bar(x, [h[idx] for h in harpa], 0.2, label='Harpa', color='green')
    ax.bar(x + 0.2, [py[idx] for py in pyocto],
           0.2, label='Pyocto', color='red')

    value = ['Precision', 'Recall']

    # Adding labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 1)
    ax.set_ylabel(value[idx])
    ax.set_title(f'{value[idx]}{title}')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()

    plt.show()


plot_comparison(gamma, harpa, pyocto, 0)
plot_comparison(gamma, harpa, pyocto, 1)

# %%
