import pandas as pd
import torch
from torch_geometric.data import Data

print("Loading CSV files...")

nodes = pd.read_csv("Neo4j dataset/nodes.csv")
edges = pd.read_csv("Neo4j dataset/edges.csv")

# -----------------------------
# Create node index mapping
# -----------------------------
node_ids = nodes["a.id"].tolist()
id_map = {nid: i for i, nid in enumerate(node_ids)}

# -----------------------------
# Node features
# -----------------------------
x = torch.tensor(
    nodes[["a.pageRankScore", "a.communityId"]].values,
    dtype=torch.float
)

# -----------------------------
# Edge index
# -----------------------------
edge_list = []

for _, row in edges.iterrows():
    src = id_map.get(row["src"])
    dst = id_map.get(row["dst"])

    if src is not None and dst is not None:
        edge_list.append([src, dst])

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# -----------------------------
# Build node labels from fraud edges
# -----------------------------
labels = [0] * len(node_ids)

for _, row in edges.iterrows():

    if row["t.isFraud"] == 1:

        src = id_map.get(row["src"])
        dst = id_map.get(row["dst"])

        if src is not None:
            labels[src] = 1

        if dst is not None:
            labels[dst] = 1

y = torch.tensor(labels, dtype=torch.long)

# -----------------------------
# Create graph
# -----------------------------
data = Data(
    x=x,
    edge_index=edge_index,
    y=y
)

print("Graph created:")
print(data)

torch.save(data, "transaction_graph.pt")

print("transaction_graph.pt saved successfully")