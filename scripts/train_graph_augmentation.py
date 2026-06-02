# scripts/train_graph_aug_random.py

import torch
import yaml
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ogb.nodeproppred import Evaluator
import numpy as np

# --- Import from your project ---
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent)) 
import utils

# --- 1. Imports needed for our local model classes ---
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch.nn import ModuleList, BatchNorm1d
from torch_geometric.utils import homophily, dropout_edge

# --- 2. 🚀 FIXED SAGE CLASS ---
class LocalSAGE(torch.nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, hidden_channels, dropout):
        super(LocalSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='max'))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='max'))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='max'))
        self.dropout = dropout
    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x # Returns raw logits

# 3. 🚀 COPIED WeightedHeteroWrapper ---
class WeightedHeteroWrapper(nn.Module):
    def __init__(self, base_model_cls, model_args, metadata, data, device):
        super().__init__()
        self.models = nn.ModuleDict()
        in_channels = model_args.get("in_channels")
        out_channels = model_args.get("out_channels")
        for edge_type in metadata[1]:
            model = base_model_cls(**model_args)
            self.models["__".join(edge_type)] = model
        weights = {}
        total_homophily = 0
        print("Calculating homophily weights for aggregation...")
        y_cpu = data["paper"].y.to('cpu').squeeze()
        for edge_type in metadata[1]:
            adj_t = data.adj_t_dict[edge_type]
            row, col, _ = adj_t.coo()
            edge_index_cpu = torch.stack([row, col], dim=0).to('cpu')
            h = homophily(edge_index_cpu, y_cpu)
            weights[edge_type] = h
            total_homophily += h
            print(f"  Edge {edge_type}: {h:.4f}")
        self.agg_logits = nn.ParameterDict()
        for edge_type in metadata[1]:
            normalized_weight = (weights[edge_type] / total_homophily) \
                if total_homophily > 0 else (1.0 / len(metadata[1]))
            self.agg_logits["__".join(edge_type)] = nn.Parameter(
                torch.tensor(normalized_weight, device=device), requires_grad=True
            )
            print(f"  Normalized weight {edge_type}: {normalized_weight:.4f}")
        self.layer_norm = nn.LayerNorm(out_channels).to(device)
        self.skip_proj = nn.Linear(in_channels, out_channels).to(device)

    def forward(self, x_dict, adj_t_dict):
        out_embeddings = []
        out_logits = []
        x_paper = x_dict["paper"]
        for edge_type_key, model in self.models.items():
            edge_type = tuple(edge_type_key.split("__"))
            if edge_type not in adj_t_dict:
                continue
            out = model(x_paper, adj_t_dict[edge_type])
            out_embeddings.append(out)
            out_logits.append(self.agg_logits[edge_type_key])
        if not out_embeddings:
            return {"paper": torch.zeros((x_paper.size(0), self.skip_proj.out_features), device=x_paper.device)}
        out_stack = torch.stack(out_embeddings, dim=0)
        logits_stack = torch.stack(out_logits, dim=0)
        learned_weights = F.softmax(logits_stack, dim=0)
        out_sum = torch.einsum('e,end->nd', learned_weights, out_stack)
        out_combined = out_sum + self.skip_proj(x_dict["paper"])
        out_norm = self.layer_norm(out_combined)
        return {"paper": out_norm}

# 4. 🚀 COPIED get_model ---
def get_model(params, data, device, verbose=False):
    num_layers = params["model"]["num_layers"]
    in_channels = data["paper"].x.shape[1]
    out_channels = len(torch.unique(data["paper"].y))
    hidden_channels = params["model"]["hidden_channels"]
    dropout = params["model"]["dropout"]
    base_model_cls = None
    model_args = {
        "num_layers": num_layers,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "hidden_channels": hidden_channels,
        "dropout": dropout
    }
    if params["model"]["name"] == "SAGE":
        base_model_cls = LocalSAGE
    else:
        raise ValueError(f"This script only supports SAGE. Config has {params['model']['name']}")
    custom_aggr = params["model"].get("custom_aggr", "mean") 
    if custom_aggr == "homophily_weighted_mean":
        print("Using custom homophily-weighted aggregation.")
        model = WeightedHeteroWrapper(
            base_model_cls=base_model_cls, 
            model_args=model_args, 
            metadata=data.metadata(), 
            data=data, 
            device=device
        ).to(device)
    else:
        print(f"Using standard PyG to_hetero aggregation: {custom_aggr}")
        base_model = base_model_cls(**model_args).to(device)
        model = to_hetero(base_model, data.metadata(), aggr=custom_aggr).to(device)
    if verbose:
        print("No. parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

# 5. 🚀 COPIED train and test functions (train is MODIFIED)
def train(model, data, train_idx, optimizer, device):
    model.train()
    drop_prob = params["model"].get("dropedge_prob", 0.0)
    
    # --- ❗ KEY: We use the AUGMENTED adj_t_dict ---
    adj_t_dict = data.aug_adj_t_dict 
    
    if drop_prob > 0.0:
        adj_t_dict_dropped = {}
        for edge_type, adj_t in adj_t_dict.items(): # Use augmented dict
            
            # --- 🚀 FIX FOR TENSOR TYPE (handles both .coo and .sizes) ---
            if hasattr(adj_t, 'coo'):
                # It's a torch_sparse.SparseTensor (like 'author', 'venue')
                row, col, value = adj_t.coo()
                shape = adj_t.sizes() # Use .sizes()
            elif adj_t.is_sparse_csr:
                # It's our new torch.Tensor (sparse_csr) for 'references'
                adj_t_coo = adj_t.to_sparse_coo()
                row, col = adj_t_coo.indices()
                value = adj_t_coo.values()
                shape = adj_t.size() # Use .size()
            else:
                raise TypeError(f"Unknown sparse tensor type in adj_t_dict: {type(adj_t)}")
            # --- ---------------------------------------------------- ---

            edge_index = torch.stack([row, col], dim=0)
            edge_index_dropped, edge_mask = dropout_edge(
                edge_index, p=drop_prob,
                force_undirected=True, 
                training=True 
            )
            edge_attr_dropped = value[edge_mask] if value is not None else None
            
            if edge_index_dropped.numel() == 0:
                adj_t_dropped = torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.long, device=device),
                    torch.empty((0), device=device),
                    shape # <-- Use the correct shape
                ).to(device).to_sparse_csr()
            else:
                adj_t_dropped = torch.sparse_coo_tensor(
                    edge_index_dropped, 
                    edge_attr_dropped if edge_attr_dropped is not None else torch.ones(edge_index_dropped.size(1), device=edge_index_dropped.device),
                    shape # <-- Use the correct shape
                ).to(device).to_sparse_csr()
            adj_t_dict_dropped[edge_type] = adj_t_dropped
        adj_t_dict = adj_t_dict_dropped 

    optimizer.zero_grad()
    out = model(data.x_dict, adj_t_dict)["paper"]
    loss = F.cross_entropy(out[train_idx], data["paper"].y[train_idx].squeeze())
    loss.backward()
    optimizer.step()
    return float(loss), out

@torch.no_grad()
def test(model, data, idx, dataset, out=None):
    model.eval()
    if out is None:
        # Use the AUGMENTED graph for validation
        out = model(data.x_dict, data.aug_adj_t_dict)['paper']
        
    loss = F.cross_entropy(out[idx], data["paper"].y[idx].squeeze())
    
    # For the FINAL TEST, evaluate on the ORIGINAL graph
    if idx.equal(data["paper"].test_idx):
        # print("Running final test evaluation on ORIGINAL graph...")
        out_test = model(data.x_dict, data.adj_t_dict)['paper'] # Use original adj_t_dict
        y_pred = out_test[idx].argmax(dim=-1, keepdim=True)
    else:
        y_pred = out[idx].argmax(dim=-1, keepdim=True)
    
    if dataset == "ogbnarxiv":
        evaluator = Evaluator(name="ogbn-arxiv")
        acc = evaluator.eval({"y_true": data["paper"].y[idx], "y_pred": y_pred})["acc"]
    else:
        acc = int((y_pred.squeeze() == data["paper"].y[idx].squeeze()).sum()) / int(idx.shape[0])
    return acc, loss

# --- 6. 🚀 NEW Augmentation Function (Replaced K-NN with Random) ---
def augment_graph_with_random_anchors(data, hard_classes, train_idx, k_neighbors=5):
    """
    Augments the graph by adding synthetic edges for hard-to-classify nodes.
    Connects hard nodes to k RANDOM nodes from the training set
    that share the same class label.
    """
    print(f"\n--- Starting Graph Augmentation for {len(hard_classes)} classes ---")
    print(f"Finding {k_neighbors} random positive anchors for hard nodes...")
    
    # Get all training node features and labels
    train_idx_cpu = train_idx.cpu()
    train_labels = data["paper"].y[train_idx_cpu].squeeze().cpu().numpy()
    
    # --- Build a map of label -> list of global node indices ---
    print("Building label-to-node map...")
    label_to_indices_map = {}
    for i in range(len(train_labels)):
        label = train_labels[i]
        global_idx = train_idx_cpu[i].item()
        if label not in label_to_indices_map:
            label_to_indices_map[label] = []
        label_to_indices_map[label].append(global_idx)
    
    # Convert lists to numpy arrays for fast sampling
    for label in label_to_indices_map:
        label_to_indices_map[label] = np.array(label_to_indices_map[label])
    # -----------------------------------------------------------

    # Find which nodes in train_idx belong to the hard classes
    hard_node_mask = np.isin(train_labels, hard_classes)
    
    # Get the *global* node indices for these hard nodes
    hard_node_global_indices = train_idx_cpu[hard_node_mask]
    
    print(f"Found {len(hard_node_global_indices)} nodes from hard classes in the training set.")
    
    new_edges_src = []
    new_edges_dst = []
    
    for hard_node_global_idx in tqdm(hard_node_global_indices, desc="Creating synthetic edges"):
        hard_node_global_idx = hard_node_global_idx.item()
        hard_node_label = data["paper"].y[hard_node_global_idx].item()
        
        # Get all nodes that could be anchors
        positive_candidates = label_to_indices_map[hard_node_label]
        
        # We need at least k_neighbors + 1 candidates (to avoid sampling the node itself)
        if len(positive_candidates) < k_neighbors + 1:
            continue
            
        # Sample K neighbors
        anchor_indices = np.random.choice(positive_candidates, k_neighbors, replace=False)
        
        # Find neighbors that are NOT the node itself
        for anchor_idx in anchor_indices:
            if anchor_idx != hard_node_global_idx:
                # Add a new undirected edge
                new_edges_src.append(hard_node_global_idx)
                new_edges_dst.append(anchor_idx)
                new_edges_src.append(anchor_idx)
                new_edges_dst.append(hard_node_global_idx)

    print(f"Created {len(new_edges_src)} new synthetic edges.")
    if not new_edges_src:
        print("No new edges created. Returning original graph.")
        data.aug_adj_t_dict = data.adj_t_dict # No change
        return data

    # --- Add new edges to the 'references' subgraph ---
    ref_edge_type = ('paper', 'references', 'paper')
    
    orig_adj_t = data.adj_t_dict[ref_edge_type]
    orig_row, orig_col, _ = orig_adj_t.coo()
    
    new_edges_src = torch.tensor(new_edges_src, dtype=torch.long)
    new_edges_dst = torch.tensor(new_edges_dst, dtype=torch.long)
    
    all_row = torch.cat([orig_row, new_edges_src])
    all_col = torch.cat([orig_col, new_edges_dst])
    
    aug_adj_t = torch.sparse_coo_tensor(
        torch.stack([all_row, all_col], dim=0),
        torch.ones(all_row.size(0)), # Assume unweighted
        orig_adj_t.sizes()
    ).to(DEVICE).to_sparse_csr()
    
    # Create a *new* adj_t_dict for training
    data.aug_adj_t_dict = {key: val for key, val in data.adj_t_dict.items()}
    data.aug_adj_t_dict[ref_edge_type] = aug_adj_t # Overwrite 'references'
    
    print("Graph augmentation complete. New 'aug_adj_t_dict' created.")
    return data


# --- Main Execution ---
def main():
    global params, DEVICE # Make params and DEVICE global
    
    CONFIG_PATH = "config/experiments_config.yaml"
    
    HARD_CLASSES = [21, 11, 29, 17, 7] 
    K_NEIGHBORS = 5 # Number of random positive anchors
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Will run on: {DEVICE}.")
    print(f"Starting Graph Augmentation (Random Anchors) for classes: {HARD_CLASSES}")

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    
    with open(CONFIG_PATH) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    path_to_data = str(Path(params["data"]["graph_dataset"][params["dataset"]]))
    data = torch.load(path_to_data, weights_only=False)
    data = data.edge_type_subgraph(
        utils.edge_type_selection(params["edge_type_selection"][params["dataset"]])
    )
    
    if any(i in params["node_embs"] for i in ["simtg", "tape"]):
        path_embs = str(Path(params["data"][f"{params['node_embs']}_embs"][params["dataset"]]))
        data["paper"].x = torch.load(path_embs).type(torch.float32)
    print(f"Loaded data with {data['paper'].num_nodes} nodes.")
    data.to(DEVICE)
    
    train_idx = data["paper"].train_idx.to(DEVICE)
    val_idx = data["paper"].val_idx.to(DEVICE)
    test_idx = data["paper"].test_idx.to(DEVICE)
    
    # --- 8. 🚀 Run Augmentation ---
    data = augment_graph_with_random_anchors(data, HARD_CLASSES, train_idx, K_NEIGHBORS)

    # --- 9. Run Training Loop ---
    params["model"]["name"] = "SAGE"
    params["model"]["custom_aggr"] = "homophily_weighted_mean"
    params["model"]["dropedge_prob"] = 0.2
    
    all_runs_accs = []
    for run in range(params.get("runs", 1)): 
        model = get_model(params, data, DEVICE, verbose=True)
        
        optimizer = torch.optim.Adam(
            params=model.parameters(), 
            weight_decay=params["optimizer"]["weight_decay"], 
            lr=params["optimizer"]["lr"]
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

        best_acc = best_epoch = -1
        for epoch in tqdm(range(params["epochs"]), desc=f"Run {run:02d}"):
            train_loss, out = train(model, data, train_idx, optimizer, DEVICE)
            val_acc, val_loss = test(model, data, val_idx, params["dataset"], out=out)
            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
            elif epoch - best_epoch > params["early_stop_threshold"]:
                tqdm.write(f"Early stopped training at epoch {epoch:02d}.")
                break

        model_path = f"models/graph_aug_random_run{run}_best.pth" 
        torch.save(model.state_dict(), model_path)
        model.load_state_dict(torch.load(model_path))
        
        test_acc, test_loss = test(model, data, test_idx, params["dataset"])
        tqdm.write(f"Run {run:02d}: Best Epoch {best_epoch:02d}, Best Val Acc {best_acc:.4f}, Test Acc {test_acc:.4f}.")
        all_runs_accs.append([best_acc, test_acc])
        torch.cuda.empty_cache()

    all_runs_accs = torch.tensor(all_runs_accs)
    print("\n* ============================= GRAPH AUGMENTATION (RANDOM) RUN =============================")
    print(f"Best Test Acc: {all_runs_accs[:, 1].max().item()*100:.2f}.")

if __name__ == "__main__":
    main()