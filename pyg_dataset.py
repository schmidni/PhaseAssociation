# %%
import seaborn as sns
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.transforms import KNNGraph

# %%
stations = pd.read_csv('stations.csv')
# arrivals = pd.read_csv('arrivals.csv', parse_dates=['time'])

# %%
# arrivals.head()
# positions = torch.tensor(arrivals[['e', 'n', 'u']].values, dtype=torch.float)

# # %%
# time = arrivals['time'] - arrivals['time'].min()
# time = torch.tensor(time.astype('int64').values, dtype=torch.float64)

# # %%
# phase = torch.tensor(arrivals['phase'].replace(
#     {'P': '0', 'S': '1'}).astype('int32').values, dtype=torch.int32)

# features = torch.vstack((time, phase)).T


# %%
# data = Data()
# data.pos = positions
# data.x = features
# data.y = arrivals['event'].nunique()
# data = KNNGraph(k=3)(data)
# data.validate(raise_on_error=True)

# %%
def pre_transform(arrivals):
    positions = torch.tensor(
        arrivals[['e', 'n', 'u']].values, dtype=torch.float)

    time = arrivals['time'] - arrivals['time'].min()
    time = torch.tensor(time.astype('int64').values, dtype=torch.float64)

    phase = torch.tensor(arrivals['phase'].replace(
        {'P': '0', 'S': '1'}).astype('int32').values, dtype=torch.int32)

    features = torch.vstack((time, phase)).T

    data = Data()
    data.pos = positions
    data.x = features
    data.y = torch.tensor([[arrivals['event'].nunique()]], dtype=torch.float)
    data = KNNGraph(k=3)(data)
    data.validate(raise_on_error=True)

    return data

# %%


class PhaseAssociationDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 force_reload=False):
        super().__init__(root, transform, pre_transform,
                         pre_filter, force_reload=force_reload)
        print(self.processed_paths)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        all_files = os.listdir(self.raw_dir)
        all_files = [f for f in all_files if f.startswith('arrivals')]
        all_files = [f for f in all_files if f.endswith('.csv')]

        return all_files

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for raw_path in self.raw_paths:
            data = pd.read_csv(raw_path, parse_dates=['time'])
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


# %%
dataset = PhaseAssociationDataset(
    'data', pre_transform=pre_transform, force_reload=True)

# %%

embedding_size = 8


class GCN(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(2, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size*2, 1)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        # Scale the output of the sigmoid to the range [1, 100]
        # Sigmoid outputs (0,1), scale to (1,100)
        out = torch.sigmoid(out) * 99 + 1

        return out, hidden


model = GCN()

# %%

# Root mean squared error
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

data_size = len(dataset)
NUM_GRAPHS_PER_BATCH = 64
loader = DataLoader(dataset[:int(data_size * 0.8)],
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(dataset[int(data_size * 0.8):],
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)


def train():
    # Enumerate over the data
    for batch in loader:
        # Use GPU
        batch.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
        # Calculating the loss and gradients
        loss = loss_fn(pred, batch.y)
        loss.backward()
        # Update using the gradients
        optimizer.step()
    return loss, embedding


# %%
print("Starting training...")
losses = []
for epoch in range(500):
    loss, h = train()
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Train Loss {loss}")

# %%

losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]
loss_indices = [i for i, l in enumerate(losses_float)]
plt = sns.lineplot(losses_float)
plt

# %%
# Analyze the results for one batch
test_batch = next(iter(test_loader))
with torch.no_grad():
    test_batch.to(device)
    pred, embed = model(test_batch.x.float(),
                        test_batch.edge_index, test_batch.batch)
    df = pd.DataFrame()
    df["y_real"] = test_batch.y.tolist()
    df["y_pred"] = pred.tolist()

df["y_real"] = df["y_real"].apply(lambda row: row[0])
df["y_pred"] = df["y_pred"].apply(lambda row: row[0])
df

# %%
plt = sns.scatterplot(data=df, x="y_real", y="y_pred")
plt
# %%
