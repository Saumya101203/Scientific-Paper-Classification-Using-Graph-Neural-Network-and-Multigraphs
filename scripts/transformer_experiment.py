from pathlib import Path
import os
import yaml
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import to_hetero, TransformerConv
from torch_geometric.utils import homophily, dropout_edge, subgraph
from tqdm import tqdm
from ogb.nodeproppred import Evaluator

import utils

# =============================================================================
# --- EXPERIMENT CONFIGURATION ---
# =============================================================================
SUBSET_RATIO = 0.4  # Use 40% of the dataset
NUM_EPOCHS = 300
HEADS = 4
HIDDEN_CHANNELS = 128
DROPOUT = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# --- TRANSFORMER ARCHITECTURE ---
# =============================================================================
class LocalTransformer(torch.nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, hidden_channels, dropout, heads=4):
        super(LocalTransformer, self).__init__()
        self.convs = torch.nn.ModuleList()
        # Divide hidden_channels by heads to prevent parameter explosion
        head_dim = hidden_channels // heads 
        
        self.convs.append(TransformerConv(in_channels, head_dim, heads=heads, dropout=dropout, beta=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_channels, head_dim, heads=heads, dropout=dropout, beta=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
        # Final layer aggregates heads by averaging (concat=False) to match class output size
        self.convs.append(TransformerConv(hidden_channels, out_channels, heads=heads, concat=False, dropout=dropout, beta=True))
        self.dropout = dropout

    def forward(self, x, edge_index):
        # TransformerConv expects edge_index natively, not SparseTensors
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class WeightedHeteroWrapper(nn.Module):
    def __init__(self, base_model_cls, model_args, metadata, data, device):
        super().__init__()
        self.models = nn.ModuleDict()
        in_channels = model_args.get("in_channels")
        out_channels = model_args.get("out_channels")

        for edge_type in metadata[1]:
            model = base_model_cls(**model_args)
            self.models["__".join(edge_type)] = model
        
        self.agg_logits = nn.ParameterDict()
        for edge_type in metadata[1]:
            self.agg_logits["__".join(edge_type)] = nn.Parameter(
                torch.tensor(1.0 / len(metadata[1]), device=device), requires_grad=True
            )

        self.layer_norm = nn.LayerNorm(out_channels).to(device)
        self.skip_proj = nn.Linear(in_channels, out_channels).to(device)

    def forward(self, x_dict, edge_index_dict):
        out_embeddings = []
        out_logits = []
        x_paper = x_dict["paper"]
        
        for edge_type_key, model in self.models.items():
            edge_type = tuple(edge_type_key.split("__"))
            if edge_type not in edge_index_dict: continue 
            
            out = model(x_paper, edge_index_dict[edge_type]) 
            out_embeddings.append(out)
            out_logits.append(self.agg_logits[edge_type_key])
        
        if not out_embeddings:
            return {"paper": torch.zeros((x_paper.size(0), self.skip_proj.out_features), device=x_paper.device)}

        out_stack = torch.stack(out_embeddings, dim=0)
        logits_stack = torch.stack(out_logits, dim=0)
        learned_weights = F.softmax(logits_stack, dim=0)
        out_sum = torch.einsum('e,end->nd', learned_weights, out_stack) 
        out_combined = out_sum + self.skip_proj(x_dict["paper"]) 
        return {"paper": self.layer_norm(out_combined)}

# =============================================================================
# --- TRAINING UTILS ---
# =============================================================================
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)["paper"]
    loss = F.cross_entropy(out[data['paper'].train_idx], data["paper"].y[data['paper'].train_idx].squeeze())
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(model, data, idx):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)['paper']
    y_pred = out[idx].argmax(dim=-1, keepdim=True)
    evaluator = Evaluator(name="ogbn-arxiv")
    acc = evaluator.eval({"y_true": data["paper"].y[idx], "y_pred": y_pred})["acc"]
    return acc

# =============================================================================
# --- MAIN PIPELINE ---
# =============================================================================
if __name__ == "__main__":
    print(f"🚀 Initializing TransformerConv Experiment on: {DEVICE}")

    project_root: Path = utils.get_project_root()
    os.chdir(project_root)
    with open(str(project_root / "config/experiments_config.yaml")) as f: params = yaml.load(f, Loader=yaml.FullLoader)

    # 1. LOAD FULL DATA
    print("\n📦 Loading Full Dataset...")
    path_to_data = str(Path(params["data"]["graph_dataset"][params["dataset"]]))
    data = torch.load(path_to_data, weights_only=False)
    data = data[0] if isinstance(data, tuple) else data
    data = data.edge_type_subgraph(utils.edge_type_selection(params["edge_type_selection"][params["dataset"]]))

    # 2. INJECT 1524 BOSS LEVEL FEATURES (Must be done before subsetting)
    print("🧬 Fusing 1524-dim Lexical-Semantic Features...")
    upgraded_path = str(project_root / "data" / "ogbn_arxiv" / "processed" / "geometric_data_processed.pt")
    upgraded_data = torch.load(upgraded_path, weights_only=False)
    upgraded_data = upgraded_data[0] if isinstance(upgraded_data, tuple) else upgraded_data
    
    mixed_features = upgraded_data['paper'].x if (hasattr(upgraded_data, 'x_dict') and 'paper' in upgraded_data.x_dict) else upgraded_data.x
    tfidf_features = mixed_features[:, -500:] 
    
    simtg_features = torch.load(str(project_root / params["data"]["simtg_embs"][params["dataset"]])).type(torch.float32)
    data['paper'].x = torch.cat([simtg_features, tfidf_features], dim=1)
    
    # 3. CREATE 40% SUBSET
    print(f"\n✂️ Slicing {SUBSET_RATIO*100}% Subgraph...")
    original_num_nodes = data['paper'].num_nodes
    num_nodes_to_keep = int(original_num_nodes * SUBSET_RATIO)
    
    torch.manual_seed(42)
    subset_node_indices = torch.randperm(original_num_nodes)[:num_nodes_to_keep]
    
    data['paper'].x = data['paper'].x[subset_node_indices]
    data['paper'].y = data['paper'].y[subset_node_indices]
    
    data.edge_index_dict = {}
    for edge_type in data.edge_types:
        if 'adj_t' in data[edge_type]:
            row, col, _ = data[edge_type].adj_t.coo()
            edge_index = torch.stack([row, col], dim=0)
            new_edge_index, _ = subgraph(subset_node_indices, edge_index, relabel_nodes=True, num_nodes=original_num_nodes)
            data.edge_index_dict[edge_type] = new_edge_index

    # Remap Train/Val/Test
    num_subset_nodes = data['paper'].x.size(0)
    split_perm = torch.randperm(num_subset_nodes)
    data['paper'].train_idx = split_perm[:int(num_subset_nodes * 0.6)]
    data['paper'].val_idx = split_perm[int(num_subset_nodes * 0.6) : int(num_subset_nodes * 0.8)]
    data['paper'].test_idx = split_perm[int(num_subset_nodes * 0.8):]

    # 4. AAAI-24 REVERSE EDGES
    print("🔄 Generating Reverse Edges...")
    citation_edge = next((et for et in list(data.edge_index_dict.keys()) if 'cites' in et[1] or 'ref' in et[1]), None)
    if citation_edge:
        rev_edge = (citation_edge[0], 'cited_by', citation_edge[2])
        fwd_index = data.edge_index_dict[citation_edge]
        data.edge_index_dict[rev_edge] = torch.stack([fwd_index[1], fwd_index[0]], dim=0)

    data.to(DEVICE)

    # 5. INITIALIZE TRANSFORMER
    print("\n🧠 Initializing Graph Transformer...")
    out_channels = len(torch.unique(data["paper"].y))
    model_args = {"num_layers": 2, "in_channels": 1524, "out_channels": out_channels, "hidden_channels": HIDDEN_CHANNELS, "dropout": DROPOUT, "heads": HEADS}
    
    model = WeightedHeteroWrapper(LocalTransformer, model_args, data.metadata(), data, DEVICE).to(DEVICE)
    print(f"   -> No. Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    # 6. TRAIN LOOP
    print("\n🔥 Commencing Training...")
    best_acc = 0
    best_epoch = 0
    patience = 30  # Stop if no improvement for 30 epochs
    
    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train(model, data, optimizer)
        if epoch % 10 == 0:
            val_acc = test(model, data, data['paper'].val_idx)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                # Save the best weights
                torch.save(model.state_dict(), "models/best_transformer.pth")
                
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f} (Best: {best_acc:.4f} at Ep {best_epoch})")
            
            # Early Stopping Check
            if epoch - best_epoch >= patience:
                print(f"\n🛑 Early stopping triggered! No improvement since epoch {best_epoch}.")
                break
                
    # Load the best weights before testing
    model.load_state_dict(torch.load("models/best_transformer.pth"))
    test_acc = test(model, data, data['paper'].test_idx)
    print("="*50)
    print(f"🏆 FINAL TRANSFORMER TEST ACCURACY: {test_acc:.4f}")
    print("="*50)