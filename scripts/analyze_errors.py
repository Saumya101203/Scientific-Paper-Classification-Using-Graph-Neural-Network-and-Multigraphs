# scripts/analyze_errors.py

import torch
import yaml
import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

# --- Import from your project ---
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent)) 
import utils
# We don't import models, we define LocalSAGE here

# --- 1. Imports needed for our local model classes ---
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch.nn import ModuleList, BatchNorm1d
from torch_geometric.utils import homophily


# --- 2. 🚀 NEW, FIXED SAGE CLASS (Copied from experiments.py) ---
class LocalSAGE(torch.nn.Module):
    """
    This is the corrected SAGE model.
    It uses 'aggr=max' and returns RAW LOGITS.
    """
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

# 3. 🚀 COPIED WeightedHeteroWrapper (Copied from experiments.py)
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

# 4. 🚀 COPIED and SIMPLIFIED get_model (Copied from experiments.py)
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
        base_model_cls = LocalSAGE # Use our local, fixed SAGE
    else:
        # We only care about SAGE for this analysis
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

@torch.no_grad()
def predict(model, data, idx):
    """
    Runs the model and returns predictions and true labels for a given index.
    """
    model.eval()
    
    # Get raw logits from our fixed model
    out = model(data.x_dict, data.adj_t_dict)['paper']
    
    # Get predictions
    y_pred = out[idx].argmax(dim=-1)
    y_true = data["paper"].y[idx].squeeze()
    
    return y_true.cpu().numpy(), y_pred.cpu().numpy()

def main():
    # --- 1. Configuration ---
    CONFIG_PATH = "config/experiments_config.yaml"
    
    # ❗ This should be the path to the model you just trained
    MODEL_PATH = "models/ogbnarxiv_SAGE_lp0.1_run0_best.pth" 
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Will run on: {DEVICE}.")

    # --- 2. Load Config and Data ---
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    
    with open(CONFIG_PATH) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    # IMPORTANT: Make sure your config has 'fos' commented out
    # so it matches the data the model was trained on.
    print(f"Using edge selection: {params['edge_type_selection'][params['dataset']]}")
    
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
    
    val_idx = data["paper"].val_idx.to(DEVICE)
    
    # --- 3. Load Model ---
    print(f"Loading model structure from config...")
    # Pass 'data' to get_model for homophily calculation
    model = get_model(params, data, DEVICE, verbose=True)
    
    print(f"Loading trained weights from: {MODEL_PATH}")
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")
    
    # --- 4. Run Predictions ---
    print("Generating predictions on validation set...")
    y_true, y_pred = predict(model, data, val_idx)
    
    # --- 5. Analyze and Report Errors ---
    print("\n--- Error Analysis Report ---")
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    
    print("\nOverall Accuracy: {:.2f}%".format(report['accuracy'] * 100))
    
    hardest_classes = report_df.drop(['accuracy', 'macro avg', 'weighted avg']) \
                             .sort_values(by='f1-score', ascending=True)
    
    print("\n--- Top 5 Hardest Classes (by F1-Score) ---")
    print(hardest_classes.head(5).to_markdown(floatfmt=".3f"))
    
    print("\n--- Confusion Matrix (Top 10 Most Confused Pairs) ---")
    cm = confusion_matrix(y_true, y_pred)
    
    np.fill_diagonal(cm, 0) # Focus on *errors*
    
    cm_flat = cm.flatten()
    top_indices = np.argsort(cm_flat)[-10:][::-1] 
    top_pairs = []
    
    for idx in top_indices:
        true_class = idx // cm.shape[1]
        pred_class = idx % cm.shape[1]
        count = cm[true_class, pred_class]
        if count == 0:
            continue
        top_pairs.append({
            "True Class": true_class,
            "Predicted Class": pred_class,
            "Count": count
        })
        
    conf_df = pd.DataFrame(top_pairs)
    print(conf_df.to_markdown(index=False))
    
    print("\n--- Analysis Complete ---")
    print("Use the 'Hardest Classes' (e.g., by f1-score) to inform your oversampling.")

if __name__ == "__main__":
    main()