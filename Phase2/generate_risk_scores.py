import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd

print("Loading graph...")

data = torch.load("transaction_graph.pt", weights_only=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = data.to(device)

# -----------------------------
# Model Definition
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

        return x  # logits


print("Loading trained model...")

model = FraudGCN(data.x.shape[1]).to(device)

model.load_state_dict(torch.load("fraud_gnn_model.pt", map_location=device))

model.eval()

# -----------------------------
# Generate Risk Scores
# -----------------------------
print("Generating risk scores...")

with torch.no_grad():

    logits = model(data.x, data.edge_index)

    scores = torch.sigmoid(logits)

risk_scores = scores.squeeze().cpu().numpy()

# -----------------------------
# Save CSV
# -----------------------------
df = pd.DataFrame({
    "node_id": list(range(len(risk_scores))),
    "risk_score": risk_scores
})

df.to_csv("risk_scores.csv", index=False)

print("Risk scores saved to risk_scores.csv")