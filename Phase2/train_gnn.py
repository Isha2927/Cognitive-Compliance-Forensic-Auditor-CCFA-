import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm

print("Loading graph...")

# Load graph
data = torch.load("transaction_graph.pt", weights_only=False)

print("Graph loaded:", data)

if data.y is None:
    raise ValueError("Graph does not contain labels (data.y).")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

# -----------------------------
# GNN Model
# -----------------------------
class FraudGCN(torch.nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 1)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)

        return x  # NO sigmoid here


# Initialize model
model = FraudGCN(data.x.shape[1]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# Handle Class Imbalance
# -----------------------------
num_pos = (data.y == 1).sum().item()
num_neg = (data.y == 0).sum().item()

print("Fraud nodes:", num_pos)
print("Normal nodes:", num_neg)

pos_weight = torch.tensor([num_neg / num_pos]).to(device)

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# -----------------------------
# Training
# -----------------------------
epochs = 120

print("Training Started")

for epoch in tqdm(range(epochs)):

    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index).squeeze()

    loss = loss_fn(logits, data.y.float())

    loss.backward()
    optimizer.step()

    # Accuracy calculation
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        acc = (preds == data.y).sum().item() / len(data.y)

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")

print("Training Finished")

# Save model
torch.save(model.state_dict(), "fraud_gnn_model.pt")

print("Model saved as fraud_gnn_model.pt")