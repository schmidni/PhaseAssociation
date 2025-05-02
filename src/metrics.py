import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import pair_confusion_matrix as PCM


class ClusterStatistics:
    def __init__(self):
        self._ari = []
        self._pcm = []
        self._event_confusion = []

    def add_embedding(self, embeddings, labels):
        # TODO: Clustering which deals with noise
        try:
            emb = embeddings.clone().cpu().numpy()
            lbl = labels.clone().cpu().numpy()
        except AttributeError:
            emb = embeddings
            lbl = labels

        num_clusters = len(np.unique(lbl))

        mask = lbl != -1

        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(
            emb[mask])

        self.add(lbl[mask], kmeans.labels_)

    def add(self, labels, labels_pred):

        mask = (labels == -1) & (labels_pred == -1)

        labels = labels[~mask]
        labels[labels == -1] = np.arange(1e6, 1e6+len(labels[labels == -1]))

        labels_pred = labels_pred[~mask]
        labels_pred[labels_pred == -1] = \
            np.arange(2e6, 2e6+len(labels_pred[labels_pred == -1]))

        self._ari.append(ARI(labels, labels_pred))

        self._pcm.append(PCM(labels, labels_pred))

    def event_confusion(self):
        return np.mean(self._event_confusion, axis=0)

    def event_precision(self):
        return np.round(self.event_confusion()[0, 0] /
                        (self.event_confusion()[0, 1]
                         + self.event_confusion()[0, 0]), 3)

    def event_recall(self):
        return np.round(self.event_confusion()[0, 0] /
                        (self.event_confusion()[1, 0]
                         + self.event_confusion()[0, 0]), 3)

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
