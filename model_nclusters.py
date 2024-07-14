# %%
from time import perf_counter

import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

from src.gnn.datasets import PhaseAssociationDataset

# %%
dataset = PhaseAssociationDataset('data', force_reload=False)

# %%

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GCN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 embedding_size=32,
                 dropout=0.5):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(in_channels, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        self.lin = Linear(embedding_size*2, embedding_size)
        self.dropout = torch.nn.Dropout(dropout)

        self.prelu = torch.nn.PReLU(device=device)

        # Output layer
        self.out = Linear(embedding_size, out_channels)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = self.prelu(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = self.prelu(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = self.prelu(hidden)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)

        hidden = F.relu(self.lin(hidden))
        hidden = self.dropout(hidden)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        # # Scale the output of the sigmoid to the range [1, 100]
        # # Sigmoid outputs (0,1), scale to (1,100)
        out = torch.sigmoid(out) * 14 + 1

        return out, hidden


model = GCN(dataset[0].num_features, 1,
            embedding_size=dataset[0].num_features*2, dropout=0.2)

# %%

# Root mean squared error
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=100, gamma=0.75)

model = model.to(device)

data_size = len(dataset)
NUM_GRAPHS_PER_BATCH = 128
loader = DataLoader(dataset[:int(data_size * 0.8)],
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(dataset[int(data_size * 0.8):],
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)


def train():
    model.train()
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
def test(loader):
    model.eval()
    predicted = torch.tensor([]).to(device)
    actual = torch.tensor([]).to(device)
    for data in loader:
        data.to(device)
        pred, embedding = model(data.x.float(), data.edge_index, data.batch)
        predicted = torch.cat((predicted, pred), 0)
        actual = torch.cat((actual, data.y), 0)

    predicted = predicted.cpu().detach().squeeze()
    actual = actual.cpu().detach().squeeze()
    class_metrics = {
        'sum': torch.sum(torch.abs(predicted - actual)).item(),
        'mean': torch.mean(torch.abs(predicted - actual)).item(),
        'accuracy': (torch.sum(
            torch.abs(predicted - actual) < 0.5) / len(actual)).item()
    }

    return class_metrics


# %%
print("Starting training...")
losses = []
time = perf_counter()
for epoch in range(1000):
    loss, h = train()
    scheduler.step()
    losses.append(loss)
    if epoch % 50 == 0:
        loss = sum(losses[-50:])/len(losses[-50:])
        print(f"Epoch {epoch} | Train Loss {loss} | "
              f"Time {(perf_counter() - time)/60} min")
        time = perf_counter()
        print('Train Scores: ', test(loader))
        print('Test Scores: ', test(test_loader))

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
