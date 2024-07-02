from src.gnn.datasets import PhaseAssociationDataset, transform_knn_stations

if __name__ == '__main__':
    dataset = PhaseAssociationDataset(
        'data', pre_transform=transform_knn_stations, force_reload=True)
