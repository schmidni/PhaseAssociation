import functools

from src.gnn.datasets import PhaseAssociationDataset, transform_knn_stations

transform = functools.partial(
    transform_knn_stations, label='event', include_s=True)

if __name__ == '__main__':
    dataset = PhaseAssociationDataset(
        'data', pre_transform=transform, force_reload=True)
