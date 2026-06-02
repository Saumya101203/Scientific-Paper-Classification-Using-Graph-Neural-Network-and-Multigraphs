import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WikiCS
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.utils import dropout_edge
import copy
from tqdm import tqdm

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_CHANNELS = 256
DROPOUT = 0.5
LR = 0.01
EPOCHS = 200
RUNS = 5

print(f"🚀 Running Wiki-CS Robustness Check on {DEVICE}...")

# 1. LOAD DATASET
dataset = WikiCS(root='./data/WikiCS')
data = dataset[0]
print(f"✅ Loaded Wiki-CS: {data.num_nodes} Nodes, {data.num_edges} Edges, {dataset.num_classes} Classes")

# 2. CONVERT TO HETERO (To match your Arxiv Architecture)
# We treat Wikipedia pages as 'paper' nodes
hetero_data = HeteroData()
hetero_data['paper'].x = data.x  # GloVe word embeddings
hetero_data['paper'].y = data.y
hetero_data['paper'].train_mask = data.train_mask[:, 0] # Use split 0
hetero_data['paper'].val_mask = data.val_mask[:, 0]
hetero_data['paper'].test_mask = data.test_mask # WikiCS provides stopping mask, we treat as test for simplicity

# --- AAAI-24: CREATE REVERSE EDGES ---
# Edge Type 1: Links To (Forward)
src, dst = data.edge_index
hetero_data['paper', 'links_to', 'paper'].edge_index = data.edge_index

# Edge Type 2: Linked By (Reverse)
# This allows the model to distinguish "Hubs" (Popular pages) from "Lists" (Pages with many links)
hetero_data['paper', 'linked_by', 'paper'].edge_index = torch.stack([dst, src], dim=0)

print("🔄 Generated AAAI-24 Reverse Edges: 'links_to' vs 'linked_by'")
hetero_data = hetero_data.to(DEVICE)

# 3. DEFINE MODEL (Your Exact Architecture)
class LocalSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='max')) # Max Agg
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='max'))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='max'))
        self.dropout = DROPOUT

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

# 4. TRAINING LOOP
results = []
print("\n--- Starting 5 Runs ---")

for run in range(RUNS):
    # Instantiate Weighted Hetero Model (Simulated via to_hetero)
    model = LocalSAGE(
        in_channels=dataset.num_features,
        out_channels=dataset.num_classes,
        hidden_channels=HIDDEN_CHANNELS
    ).to(DEVICE)
    
    # This automatically creates separate weights for 'links_to' vs 'linked_by'
    model = to_hetero(model, hetero_data.metadata(), aggr='mean').to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    
    best_val = 0
    best_test = 0
    
    pbar = tqdm(range(EPOCHS), desc=f"Run {run+1}")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        out = model(hetero_data.x_dict, hetero_data.edge_index_dict)['paper']
        loss = F.cross_entropy(out[hetero_data['paper'].train_mask], hetero_data['paper'].y[hetero_data['paper'].train_mask])
        loss.backward()
        optimizer.step()
        
        # Eval
        model.eval()
        with torch.no_grad():
            out = model(hetero_data.x_dict, hetero_data.edge_index_dict)['paper']
            pred = out.argmax(dim=-1)
            
            val_acc = (pred[hetero_data['paper'].val_mask] == hetero_data['paper'].y[hetero_data['paper'].val_mask]).float().mean().item()
            test_acc = (pred[hetero_data['paper'].test_mask] == hetero_data['paper'].y[hetero_data['paper'].test_mask]).float().mean().item()
            
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                
            pbar.set_postfix(Val=f"{best_val:.2%}", Test=f"{best_test:.2%}")
            
    results.append(best_test)
    print(f"   Run {run+1} Best Test Acc: {best_test:.4f}")

# 5. SUMMARY
results = torch.tensor(results)
print("\n" + "="*40)
print(f"🏆 Wiki-CS Robustness Results")
print(f"   Mean Test Acc: {results.mean().item():.4f} ± {results.std().item():.4f}")
print("="*40)