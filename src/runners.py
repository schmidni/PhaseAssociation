import numpy as np
import pandas as pd
import torch
from harpa import association as harpa_association
from sklearn.preprocessing import MinMaxScaler

from src.gamma.utils import association as gamma_association
from src.models import PhasePickTransformer


def run_pyocto(picks, stations, associator):
    events, associations = associator.associate_gamma(picks, stations)

    labels_pred = np.full(len(picks), -1)
    labels_pred[associations['pick_idx']] = associations['event_idx']

    events['z'] = 0.1
    events['time'] = events['time'].astype(int)*1e9

    return events, labels_pred


def run_gamma(picks, stations, config):
    events, associations = gamma_association(
        picks, stations, config, method=config["method"])

    events = pd.DataFrame(events)

    if len(associations) == 0:
        return events, np.full(len(picks), -1)

    events['time'] = pd.to_datetime(
        events['time'], unit='ns').values.astype(int)

    # association columns "pick_index", "event_index", "gamma_score"
    associations = np.array([*associations])[:, :2].astype(int)
    labels_pred = np.full(len(picks), -1)
    labels_pred[associations[:, 0]] = associations[:, 1]

    return events, labels_pred


def run_harpa(picks, stations, config):
    pick_df_out, catalog_df = harpa_association(
        picks, stations, config, verbose=True)

    return catalog_df, pick_df_out['event_index'].to_numpy()


def run_phassoc(picks, stations, model_path):

    picks = picks.join(stations.set_index('id'), on='id')
    station_index_mapping = pd.Series(stations.index, index=stations['id'])
    picks['id_index'] = picks['id'].map(station_index_mapping)
    picks = picks.drop(
        columns=['prob', 'id'])
    picks['timestamp'] = picks['timestamp'].values.astype(int)
    picks['timestamp'] = picks['timestamp'] - picks['timestamp'].min()
    picks['type'] = picks['type'].astype('category').cat.codes
    picks = picks[['x(km)', 'y(km)', 'z(km)', 'timestamp',
                   'type', 'amp', 'id_index']]

    scaler = MinMaxScaler()
    for col in ['x(km)', 'y(km)', 'z(km)', 'timestamp', 'amp']:
        picks[col] = scaler.fit_transform(picks[col].to_numpy().reshape(-1, 1))

    st = torch.tensor(picks['id_index'].to_numpy()).to(
        dtype=torch.long).unsqueeze(0)
    xs = torch.tensor(picks.drop(columns='id_index').to_numpy()).to(
        dtype=torch.float32).unsqueeze(0)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    model = PhasePickTransformer(
        input_dim=6, num_stations=len(stations),
        embed_dim=128, num_heads=4, num_layers=2,
        max_picks=2000).to(device)

    model.load_state_dict(torch.load(model_path,
                                     weights_only=True,
                                     map_location=device))

    with torch.no_grad():
        model.eval()
        embeddings = model(xs.to(device), st.to(device))
        embeddings = embeddings.cpu().squeeze().detach().numpy()

        return None, embeddings
