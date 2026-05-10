import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

print("Loading CSV files...")

nodes = pd.read_csv("Neo4j dataset/nodes.csv")
edges = pd.read_csv("Neo4j dataset/edges.csv")

print("Nodes:", len(nodes))
print("Edges:", len(edges))

print("Mapping node IDs...")

# create mapping from node id -> index
node_id_map = {nid: i for i, nid in enumerate(nodes["a.id"])}

print("Building edge index...")

src = []
dst = []

for _, row in tqdm(edges.iterrows(), total=len(edges), desc="Processing edges"):

    if row["src"] in node_id_map and row["dst"] in node_id_map:
        src.append(node_id_map[row["src"]])
        dst.append(node_id_map[row["dst"]])

edge_index = torch.tensor([src, dst], dtype=torch.long)

print("Building node features...")

x = torch.tensor(
    nodes[["a.pageRankScore", "a.communityId"]].values,
    dtype=torch.float
)

print("Building labels...")

y = torch.tensor(edges["t.isFraud"].values, dtype=torch.long)

data = Data(x=x, edge_index=edge_index)

print("\nGraph Created Successfully")
print(data)

torch.save(data, "transaction_graph.pt")

print("Graph saved as transaction_graph.pt")