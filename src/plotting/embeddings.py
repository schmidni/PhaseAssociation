import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_embeddings_reduced(embeddings, labels, catalog, method='pca'):
    """
    Plots 2D PCA projection of embeddings, colored by label.

    Args:
        embeddings (Tensor or ndarray): shape (N, D)
        labels (Tensor or ndarray): shape (N,)
        title (str): title of the plot
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(catalog, torch.Tensor):
        catalog = catalog.detach().cpu().numpy()
    if not len(np.unique(labels)) - 1 == len(catalog.squeeze()):
        raise ValueError(
            "Number of unique labels does not match number of events.")

    # Reduce dimensionality with PCA
    if method == 'pca':
        reducer = PCA(n_components=2)

    # Reduce dimensionality with t-
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30,
                       max_iter=1000, init='random', random_state=42)

    reduced = reducer.fit_transform(embeddings)

    # Get unique labels
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    cmap = plt.get_cmap('tab20', num_classes)
    color_list = [cmap(i) for i in range(num_classes)]
    color_list[0] = 'black'
    label_to_color = {label: color_list[i]
                      for i, label in enumerate(unique_labels)}

    fig = plt.figure(figsize=(14, 20))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 4])

    ax1 = fig.add_subplot(gs[0])  # Embedding (2D)
    ax2 = fig.add_subplot(gs[1])  # Timeline (1D)
    ax3 = fig.add_subplot(gs[2], projection='3d')  # 3D location plot

    # Top: Embedding plot
    for label in unique_labels:
        idx = labels == label
        ax1.scatter(reduced[idx, 0], reduced[idx, 1],
                    c=[label_to_color[label]],
                    label=str(label), s=50, alpha=0.8)
    ax1.set_title(f'{method.upper()} of Seismic Pick Embeddings')
    ax1.set_xlabel('Dim 1' if method == 'tsne' else 'PC 1')
    ax1.set_ylabel('Dim 2' if method == 'tsne' else 'PC 2')
    ax1.grid(True)
    ax1.legend(title='Event Label', bbox_to_anchor=(
        1.05, 1), loc='upper left', fontsize='small')

    # Bottom: Timeline plot
    for i, row in enumerate(catalog.squeeze()):
        label = i
        time = row[3]
        magnitude = row[4]
        # fallback in case of mismatch
        color = label_to_color.get(label, 'gray')
        ax2.scatter(time, magnitude, color=color, s=100)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Magnitude')
    ax2.grid(True)
    ax2.set_title('Event Timeline')

    # Bottom 3rd plot: 3D spatial event location
    for i, row in enumerate(catalog.squeeze()):
        label = i
        e, n, u = row[0], row[1], row[2]
        color = label_to_color.get(label, 'gray')
        ax3.scatter(e, n, u, color=color, s=150)

    ax3.set_xlabel('Easting (m)')
    ax3.set_ylabel('Northing (m)')
    ax3.set_zlabel('Up (m)')
    ax3.set_title('3D Event Locations')

    plt.tight_layout()
    plt.show()
