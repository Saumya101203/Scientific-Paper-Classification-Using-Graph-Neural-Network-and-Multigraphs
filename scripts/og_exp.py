from pathlib import Path
import os
import yaml
import argparse  # Import argparse for command-line arguments

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from tqdm import tqdm
from ogb.nodeproppred import Evaluator
from torch_geometric.transforms import ToSparseTensor
# Make sure utils and models can be imported from the project root
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import utils
import models

# --- add below ---
import torch.serialization
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.data.storage import NodeStorage, EdgeStorage, GlobalStorage, BaseStorage
import pandas as pd
torch.serialization.add_safe_globals([
    HeteroData, NodeStorage, EdgeStorage, GlobalStorage, BaseStorage, pd.Series
])
# --- end block ---


def train(model, data, train_idx, optimizer, model_name):
    """
    Corrected train function to handle model output keyed by canonical edge types.
    """
    model.train()
    optimizer.zero_grad()
    
    out_raw = model(data.x_dict, data.adj_t_dict)
    
    if isinstance(out_raw, dict):
        paper_output_tensors = []
        
        # 1. Collect all output tensors that target the 'paper' node type
        for key, tensor in out_raw.items():
            # Check if the key is a tuple (canonical edge type) and targets 'paper'
            if isinstance(key, tuple) and key[2] == 'paper':
                paper_output_tensors.append(tensor)
            # Handle the case where the key might be just 'paper' (for safety, though it failed last time)
            elif key == 'paper':
                 paper_output_tensors.append(tensor)

        if not paper_output_tensors:
            print(f"\n[DEBUG] Model output dict keys: {list(out_raw.keys())}. No target 'paper' key found.")
            raise TypeError("Model output is a dictionary, but no tensors were found for the target node type 'paper'.")

        # 2. Aggregate the results (e.g., using sum or mean). Sum is a common, safe choice.
        # All tensors should have the same shape: [num_paper_nodes, num_classes]
        out = sum(paper_output_tensors)
        
    elif torch.is_tensor(out_raw):
        # Fallback for models that return a direct tensor (e.g., non-heterogeneous or simple to_hetero)
        out = out_raw
    else:
        raise TypeError(f"Model output of type {type(out_raw)} is neither a dict nor a tensor.")

    # Apply log_softmax if the model does not include it (standard for NLLLoss)
    # out = F.log_softmax(out, dim=-1)

    loss = F.nll_loss(out[train_idx], data["paper"].y[train_idx].squeeze())
    loss.backward()
    optimizer.step()
    
    return float(loss), out


@torch.no_grad()
def test(model, data, idx, dataset, model_name, out=None):
    """
    Corrected test function to handle model output keyed by canonical edge types.
    """
    model.eval()
    
    if out is None:
        out_raw = model(data.x_dict, data.adj_t_dict)
        
        if isinstance(out_raw, dict):
            paper_output_tensors = []
            
            # 1. Collect all output tensors that target the 'paper' node type
            for key, tensor in out_raw.items():
                if isinstance(key, tuple) and key[2] == 'paper':
                    paper_output_tensors.append(tensor)
                elif key == 'paper':
                    paper_output_tensors.append(tensor)

            if not paper_output_tensors:
                raise TypeError("Model output is a dictionary, but no tensors were found for the target node type 'paper'.")

            # 2. Aggregate the results (e.g., sum)
            out = sum(paper_output_tensors)
            
        elif torch.is_tensor(out_raw):
            out = out_raw
        else:
            raise TypeError(f"Model output of type {type(out_raw)} is neither a dict nor a tensor.")

    # Apply log_softmax if the model does not include it
    # out = F.log_softmax(out, dim=-1)

    loss = F.nll_loss(out[idx], data["paper"].y[idx].squeeze())
    y_pred = out[idx].argmax(dim=-1, keepdim=True)
    
    # ... (Rest of the test function logic remains the same)
    if dataset == "ogbnarxiv":
        # Assuming Evaluator is defined elsewhere
        from ogb.nodeproppred import Evaluator 
        evaluator = Evaluator(name="ogbn-arxiv")
        y_true = data["paper"].y[idx].view(-1, 1).to(torch.long) 
        
        acc = evaluator.eval({"y_true": y_true, "y_pred": y_pred})["acc"]
    else:
        acc = int((y_pred.squeeze() == data["paper"].y[idx].squeeze()).sum()) / int(idx.shape[0])

    return acc, loss
def get_model(params, data, verbose=False):
    """
    Expanded get_model function to include all defined models.
    """
    model_params = params["model"]
    model_name = model_params["name"]
    
    # Common parameters
    in_channels = data["paper"].x.shape[1]
    out_channels = len(torch.unique(data["paper"].y))
    hidden_channels = model_params["hidden_channels"]
    num_layers = model_params["num_layers"]
    dropout = model_params["dropout"]

    # Model selection
    if model_name == "GCN":
        model = models.GCN(num_layers, in_channels, out_channels, hidden_channels, dropout)
    elif model_name == "SAGE":
        model = models.SAGE(num_layers, in_channels, out_channels, hidden_channels, dropout)
    elif model_name == "GCNJKNet":
        model = models.GCNJKNet(num_layers, in_channels, out_channels, hidden_channels, dropout, mode="max")
    elif model_name == "SGC":
        model = models.SGC(num_layers, in_channels, out_channels)
    elif model_name == "APPNP":
        model = models.APPNP(num_layers, in_channels, out_channels, hidden_channels, dropout)
    elif model_name == "GAT":
        heads = model_params.get("heads", 4) # Add default heads if not in config
        model = models.GAT(in_channels, hidden_channels, out_channels, num_layers, n_heads=heads, dropout=dropout)
    elif model_name == "RevGAT":
        heads = model_params.get("heads" ,4)
        model = models.RevGAT(num_layers, in_channels, out_channels, hidden_channels, dropout, heads=heads)
    elif model_name == "GIN":
         model = models.GIN(in_channels, hidden_channels, out_channels, num_layers, dropout)
    elif model_name == "SIGN":
        # SIGN is a special case; it's an MLP that expects pre-computed features
        num_mlp_layers = model_params.get("num_mlp_layers", 2)
        # Note: For SIGN, 'in_channels' should be pre-calculated based on propagation steps
        model = models.SIGN(in_channels, out_channels, hidden_channels, dropout, num_mlp_layers)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # The SIGN model is an MLP and should not be converted to heterogeneous.
    # It operates on the 'paper' node features directly.
    if model_name != "SIGN":
        model = to_hetero(model, data.metadata(), aggr="mean")
    
    model.to(DEVICE)
    
    if verbose:
        print(f"Model: {model_name}")
        print("No. parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


if __name__ == "__main__":
    # --- 1. Argument Parsing for Config File ---
    parser = argparse.ArgumentParser(description='Run GNN experiments.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Will run on: {DEVICE}.")

    # --- 2. Load Configuration from Provided Path ---
    with open(args.config) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # Load data from the path specified in the config
    path_to_data = str(Path(params["data"]["graph_dataset"][params["dataset"]]))
    data = torch.load(path_to_data, map_location='cpu') # Load to CPU first

    # Prune graph edges if specified
    data = data.edge_type_subgraph(utils.edge_type_selection(params["edge_type_selection"][params["dataset"]]))
    
    # ===================================================================
    # == CORRECTON: Add this block to convert the data format          ==
    # ===================================================================
    print("Converting edge_index to SparseTensor format (adj_t)...")
    data = ToSparseTensor()(data)
    print("Conversion complete.")
    # ===================================================================

    # --- 3. Handle Node Embeddings ---
    # The 'preloaded_in_subset' flag skips this block, as features are already in the .pt file
    if params.get("node_embs") and params["node_embs"] not in ["preloaded_in_subset"]:
        print(f"Loading external node embeddings of type: {params['node_embs']}")
        path_embs = str(Path(params["data"][f"{params['node_embs']}_embs"][params["dataset"]]))
        if params["node_embs"] == "tape":
            # (Your TAPE loading logic here)
            pass
        else: # For simtg and others
            data["paper"].x = torch.load(path_embs).type(torch.float32)
        print(f"Loaded embeddings with shape: {data['paper'].x.shape}")
    else:
        print(f"Using node features pre-loaded in the data object. Shape: {data['paper'].x.shape}")

    data.to(DEVICE)

    path_to_model = f"models/{params['dataset']}_{params['model']['name']}"
    os.makedirs(Path(path_to_model).parent, exist_ok=True)
    all_runs_accs = []
    
    # Assume fixed splits for ogbn-arxiv as per OGB convention
    train_idx, val_idx, test_idx = data["paper"].train_idx, data["paper"].val_idx, data["paper"].test_idx

    for run in range(params["runs"]):
        model = get_model(params, data, verbose=(run == 0))
        model_name = params['model']['name']
        
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            weight_decay=params["optimizer"]["weight_decay"],
            lr=params["optimizer"]["lr"]
        )
        
        # Using a simple scheduler, can be made more complex if needed
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)

        best_val_acc = best_epoch = -1
        for epoch in tqdm(range(params["epochs"]), desc=f"Run {run:02d}"):
            train_loss, out = train(model, data, train_idx, optimizer, model_name)
            val_acc, val_loss = test(model, data, val_idx, params["dataset"], model_name, out=out)
            
            # Note: For ReduceLROnPlateau, step with the metric to monitor
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), f"{path_to_model}_run{run}_best.pth")
            
            elif epoch - best_epoch > params["early_stop_threshold"]:
                tqdm.write(f"Early stopped training for run {run:02d} at epoch {epoch+1}.")
                break

        model.load_state_dict(torch.load(f"{path_to_model}_run{run}_best.pth"))
        test_acc, test_loss = test(model, data, test_idx, params["dataset"], model_name)

        tqdm.write(f"Run {run:02d}: Best Epoch {best_epoch+1:03d}, Best Val Acc {best_val_acc:.4f}, Test Acc {test_acc:.4f}.")
        all_runs_accs.append([best_val_acc, test_acc])
        torch.cuda.empty_cache()

    all_runs_accs = torch.tensor(all_runs_accs)
    print("\n* ============================= ALL RUNS SUMMARY =============================")
    print(f"Best Val Acc: {all_runs_accs[:, 0].max().item()*100:.2f}, Corresponding Test Acc: {all_runs_accs[all_runs_accs[:,0].argmax(), 1].item()*100:.2f}")
    print(f"Avg. Test Acc: {all_runs_accs[:, 1].mean().item()*100:.2f} ± {all_runs_accs[:, 1].std().item()*100:.2f}.")
    print("* ===========================================================================")