# %%
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from src.gnn.datasets import PhaseAssociationDataset

# %%
# Load data
dataset = PhaseAssociationDataset('data', force_reload=False)
n = len(dataset)

test_dataset = dataset[:int(n*0.2)]
train_dataset = dataset[int(n*0.2):]

test_loader = DataLoader(test_dataset, batch_size=1)
train_loader = DataLoader(train_dataset, batch_size=1)


# %%
class GNN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h


def nt_xent_loss(embeddings, labels, temperature=0.5):
    """
    Computes the NT-Xent loss for given embeddings and labels.

    Parameters:
    - embeddings: Tensor of shape (N, D) where N is the number of samples and D is the embedding dimension.
    - labels: Tensor of shape (N,) containing the labels for the samples.
    - temperature: The temperature parameter for scaling similarities.

    Returns:
    - loss: The computed NT-Xent loss.
    """
    # Normalize the embeddings
    embeddings = F.normalize(embeddings, dim=1)

    # Compute similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # Create label mask for positive pairs
    labels = labels.contiguous().view(-1, 1)
    label_mask = torch.eq(labels, labels.T).float()

    # Exclude self-similarity from the mask
    mask = torch.eye(labels.shape[0], device=labels.device)
    label_mask = label_mask - mask

    # Compute positive and negative similarities
    positives = sim_matrix * label_mask
    negatives = sim_matrix * (1 - label_mask - mask)

    # Compute log-softmax for negatives
    # exp_negatives = torch.exp(negatives)
    logsumexp_negatives = torch.logsumexp(negatives, dim=1, keepdim=True)

    # Compute positive similarities loss
    pos_loss = -positives.sum(dim=1) / temperature

    # Final loss
    loss = pos_loss + logsumexp_negatives.squeeze(1)
    return loss.mean()


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(dataset.num_features, 64, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# %%


def train(loader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for data in loader:
            data.to(device)
            optimizer.zero_grad()

            embeddings = model(data)

            loss = nt_xent_loss(embeddings, data.y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")


@torch.no_grad()
def test(loader):
    model.eval()
    ari = 0
    for data in loader:
        data.to(device)
        embeddings = model(data)

        # Cluster embeddings
        kmeans = KMeans(n_clusters=5, random_state=0).fit(embeddings.numpy())
        labels = kmeans.labels_

        ari += adjusted_rand_score(data.y.cpu(), labels)
    print(f"ARI: {ari / len(loader.dataset)}")


# %%
for i in range(10):
    train(train_loader)
    test(test_loader)

# %%
ari = 0
for data in train_loader:

    # Cluster embeddings
    kmeans = KMeans(n_clusters=5, random_state=0).fit(data.x.numpy())
    labels = kmeans.labels_

    ari += adjusted_rand_score(data.y.cpu(), labels)

print(f"ARI: {ari / len(train_loader.dataset)}")

# %%
