from pathlib import Path
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from torch_geometric.utils import homophily, dropout_edge  # Changed from dropout_adj
# 1. 🚀 NEW IMPORT for Label Propagation
from torch_geometric.nn.models import LabelPropagation
from tqdm import tqdm
from ogb.nodeproppred import Evaluator

import utils
import models


class WeightedHeteroWrapper(nn.Module):
    """
    A custom heterogeneous GNN wrapper that replaces to_hetero.
    It instantiates a separate base GNN model for each edge type and
    aggregates their outputs using a weighted mean, where weights are
    based on the homophily of each subgraph.
    """
    def __init__(self, base_model_cls, model_args, metadata, data, device):
        super().__init__()
        
        self.models = nn.ModuleDict()
        
        # --- Get model-specific args ---
        # We need in/out channels for the new layers
        in_channels = model_args.get("in_channels")
        out_channels = model_args.get("out_channels")

        # Instantiate a base model for each edge type
        for edge_type in metadata[1]:
            model = base_model_cls(**model_args)
            self.models["__".join(edge_type)] = model
        
        # --- Calculate and register homophily-based weights ---
        weights = {}
        total_homophily = 0
        print("Calculating homophily weights for aggregation...")
        
        # Ensure data for homophily calc is on CPU
        y_cpu = data["paper"].y.to('cpu').squeeze()
        num_nodes = data["paper"].num_nodes

        for edge_type in metadata[1]:
            adj_t = data.adj_t_dict[edge_type]
            row, col, _ = adj_t.coo()
            edge_index = torch.stack([row, col], dim=0)
            
            edge_index_cpu = edge_index.to('cpu')
            
            # Calculate homophily
            # Removed the 'num_nodes' argument as it's not accepted by the function
            h = homophily(edge_index_cpu, y_cpu)
            weights[edge_type] = h
            total_homophily += h
            print(f"  Edge {edge_type}: {h:.4f}")
        
        # --- Normalize and register weights as non-trainable parameters ---
        # --- Renamed to 'logits' as we will apply softmax in forward() ---
        self.agg_logits = nn.ParameterDict()
        if total_homophily == 0:
            print("Warning: Total homophily is 0. Using uniform weights.")
            
        for edge_type in metadata[1]:
            # Use uniform weights if total homophily is 0
            normalized_weight = (weights[edge_type] / total_homophily) \
                if total_homophily > 0 else (1.0 / len(metadata[1]))
            
            # We initialize the logits with our homophily scores. This is a
            # smart initialization, giving the model a good starting point.
            self.agg_logits["__".join(edge_type)] = nn.Parameter(
                torch.tensor(normalized_weight, device=device), requires_grad=True
            )
            print(f"  Normalized weight {edge_type}: {normalized_weight:.4f}")

        # --- NEW: Add LayerNorm and a Linear projection for the skip connection ---
        self.layer_norm = nn.LayerNorm(out_channels).to(device)
        self.skip_proj = nn.Linear(in_channels, out_channels).to(device)


    def forward(self, x_dict, adj_t_dict):
        out_embeddings = []
        out_logits = []
        x_paper = x_dict["paper"]
        
        # Iterate over the models we created, one for each edge type
        for edge_type_key, model in self.models.items():
            edge_type = tuple(edge_type_key.split("__"))
            
            # Check if this edge type exists in the input adj_t_dict
            # (it might not, e.g., if DropEdge removed all edges)
            if edge_type not in adj_t_dict:
                continue 
                
            adj_t = adj_t_dict[edge_type]
            
            # Apply the homogeneous model to the 'paper' nodes and this edge type
            out = model(x_paper, adj_t)
            
            # --- Store the output and its corresponding logit ---
            out_embeddings.append(out)
            out_logits.append(self.agg_logits[edge_type_key])
        
        if not out_embeddings:
            # Should not happen if data is valid, but as a safeguard
            print("Warning: No edge types processed in forward pass.")
            # Return zeros in the expected shape
            return {"paper": torch.zeros((x_paper.size(0), self.skip_proj.out_features), device=x_paper.device)}

        # --- NEW: Gated Aggregation using Softmax ---
        
        # Stack all GNN outputs: (num_edge_types, num_nodes, out_channels)
        out_stack = torch.stack(out_embeddings, dim=0)
        
        # Stack all logits: (num_edge_types)
        logits_stack = torch.stack(out_logits, dim=0)
        
        # 1. Apply softmax to logits to get stable, learned weights that sum to 1
        learned_weights = F.softmax(logits_stack, dim=0)
        
        # 2. Apply weights and sum
        # 'e,end->nd' = (num_edges), (num_edges, num_nodes, dims) -> (num_nodes, dims)
        out_sum = torch.einsum('e,end->nd', learned_weights, out_stack)
        
        # --- Apply residual connection and layer norm (as before) ---
        
        # 1. Project original x for skip connection
        x_skip = self.skip_proj(x_dict["paper"]) 
        
        # 2. Add skip connection to GNN output
        out_combined = out_sum + x_skip
        
        # 3. Apply LayerNorm
        out_norm = self.layer_norm(out_combined)
        
        # Return the normalized, combined output
        return {"paper": out_norm}


def train(model, data, train_idx, optimizer):
    model.train()

    # --- Apply DropEdge ---
    # Get drop probability from params, default to 0.0 (no dropout)
    drop_prob = params["model"].get("dropedge_prob", 0.0)
    adj_t_dict = data.adj_t_dict
    
    if drop_prob > 0.0:
        # We are in training mode, apply DropEdge
        adj_t_dict_dropped = {}
        for edge_type, adj_t in data.adj_t_dict.items():
            row, col, value = adj_t.coo()
            edge_index = torch.stack([row, col], dim=0)
            
            # 1. Corrected usage of dropout_edge
            # It does not take 'edge_attr' as input.
            edge_index_dropped, edge_mask = dropout_edge(
                edge_index, p=drop_prob,
                force_undirected=True, # As per paper's preprocessing
                # num_nodes=data["paper"].num_nodes, # <--- This line caused the error
                training=True # Enable dropout
            )
            
            # 2. Use the returned mask to filter the edge attributes (values)
            edge_attr_dropped = value[edge_mask] if value is not None else None
            
            if edge_index_dropped.numel() == 0:
                # Handle case where all edges are dropped for an edge type
                adj_t_dropped = torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.long, device=DEVICE),
                    torch.empty((0), device=DEVICE),
                    adj_t.sizes()  # 2. Fixed: .size() -> .sizes()
                ).to(DEVICE).to_sparse_csr() # <-- 1. Convert to CSR
            else:
                # Create new SparseTensor from dropped edges
                adj_t_dropped = torch.sparse_coo_tensor(
                    edge_index_dropped, 
                    edge_attr_dropped if edge_attr_dropped is not None else torch.ones(edge_index_dropped.size(1), device=edge_index_dropped.device),
                    adj_t.sizes() # 3. Fixed: .size() -> .sizes()
                ).to(DEVICE).to_sparse_csr() # <-- 2. Convert to CSR
            
            # The .to_sparse_csr() conversion already handles coalescing.
            # Calling .coalesce() on a CSR tensor causes the error.
            adj_t_dict_dropped[edge_type] = adj_t_dropped # <-- REMOVED .coalesce()
        
        adj_t_dict = adj_t_dict_dropped # Use the new dict with dropped edges

    # --- Standard Training Step ---
    optimizer.zero_grad()
    out = model(data.x_dict, adj_t_dict)["paper"]
    
    # --- 2. 🚀 CHANGED TO cross_entropy (expects raw logits) ---
    loss = F.cross_entropy(out[train_idx], data["paper"].y[train_idx].squeeze())
    
    loss.backward()
    optimizer.step()
    
    return float(loss), out


# @torch.no_grad()
# def test(model, data, idx, dataset, out=None):
#     model.eval()
    
#     # --- 3. 🚀 MODIFIED to return logits if computed ---
#     out_was_none = False
#     if out is None:
#         # Use the original, non-dropped adj_t_dict for evaluation
#         out = model(data.x_dict, data.adj_t_dict)['paper']
#         out_was_none = True
        
#     # --- 4. 🚀 CHANGED TO cross_entropy (expects raw logits) ---
#     loss = F.cross_entropy(out[idx], data["paper"].y[idx].squeeze())
    
#     y_pred = out[idx].argmax(dim=-1, keepdim=True)
#     if dataset == "ogbnarxiv":
#         evaluator = Evaluator(name="ogbn-arxiv")
#         acc = evaluator.eval({"y_true": data["paper"].y[idx], "y_pred": y_pred})["acc"]
#     else:
#         acc = int((y_pred == data["paper"].y[idx]).sum()) / int(idx.shape[0])

#     # --- 5. 🚀 Return logits tensor if it was computed in this call ---
#     if out_was_none:
#         return acc, loss, out
    
#     return acc, loss

@torch.no_grad()
def test(model, data, idx, dataset, out=None):
    model.eval()
    
    if out is None:
        # Use the original, non-dropped adj_t_dict for evaluation
        out = model(data.x_dict, data.adj_t_dict)['paper']
        
    # --- CHANGED TO cross_entropy (expects raw logits) ---
    loss = F.cross_entropy(out[idx], data["paper"].y[idx].squeeze())
    
    y_pred = out[idx].argmax(dim=-1, keepdim=True)
    if dataset == "ogbnarxiv":
        evaluator = Evaluator(name="ogbn-arxiv")
        acc = evaluator.eval({"y_true": data["paper"].y[idx], "y_pred": y_pred})["acc"]
    else:
        acc = int((y_pred == data["paper"].y[idx]).sum()) / int(idx.shape[0])
    
    return acc, loss

def get_model(verbose=False):
    num_layers = params["model"]["num_layers"]
    in_channels = data["paper"].x.shape[1]
    out_channels = len(torch.unique(data["paper"].y))
    hidden_channels = params["model"]["hidden_channels"]
    dropout = params["model"]["dropout"]

    # --- Consolidate model arguments ---
    base_model_cls = None
    model_args = {
        "num_layers": num_layers,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "hidden_channels": hidden_channels,
        "dropout": dropout
    }

    if params["model"]["name"] == "GCN":
        base_model_cls = models.GCN
    elif params["model"]["name"] == "SAGE":
        base_model_cls = models.SAGE
    elif params["model"]["name"] == "GCNJKNet":
        base_model_cls = models.GCNJKNet
        model_args["mode"] = "max" # Add specific arg
    elif params["model"]["name"] == "SGC":
        base_model_cls = models.SGC
        # SGC has different args
        model_args = {
            "num_layers": num_layers,
            "in_channels": in_channels,
            "out_channels": out_channels
        }
    else:
        raise ValueError(f"Unknown model name {params['model']['name']}")

    # --- Select aggregation method ---
    custom_aggr = params["model"].get("custom_aggr", "mean") # default to original "mean"
    
    if custom_aggr == "homophily_weighted_mean":
        print("Using custom homophily-weighted aggregation.")
        model = WeightedHeteroWrapper(
            base_model_cls=base_model_cls, 
            model_args=model_args, 
            metadata=data.metadata(), 
            data=data, 
            device=DEVICE
        ).to(DEVICE)
    else:
        print(f"Using standard PyG to_hetero aggregation: {custom_aggr}")
        # Original code path
        base_model = base_model_cls(**model_args).to(DEVICE)
        model = to_hetero(base_model, data.metadata(), aggr=custom_aggr).to(DEVICE)
    
    if verbose:
        print("No. parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


# if __name__ == "__main__":
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Will run on: {DEVICE}.")

#     project_root: Path = utils.get_project_root()
#     os.chdir(project_root)
#     with open(str(project_root / "config/experiments_config.yaml")) as f:
#         params = yaml.load(f, Loader=yaml.FullLoader)
#     with open(str(project_root / "config/data_generation_config.yaml")) as f:
#         data_gen_params = yaml.load(f, Loader=yaml.FullLoader)

#     path_to_data = str(Path(params["data"]["graph_dataset"][params["dataset"]]))
#     # Set weights_only=False to load PyG data objects, as per torch 2.6+ update
#     data = torch.load(path_to_data, weights_only=False)

#     data = data.edge_type_subgraph(utils.edge_type_selection(params["edge_type_selection"][params["dataset"]]))

#     if any(i in params["node_embs"] for i in ["simtg", "tape"]):
#         path_embs = str(Path(params["data"][f"{params['node_embs']}_embs"][params["dataset"]]))
#         if params["node_embs"] == "tape":
#             init_x_shape = (19717, 768) if params["dataset"] == "pubmed" else (data["paper"].num_nodes, 768)
#             features = np.array(np.memmap(path_embs, mode='r', dtype=np.float16, shape=init_x_shape))
#             if params["dataset"] == "pubmed":
#                 features = np.delete(features, [2459], axis=0) # manually remove index corresponding to ID with no metadata from the precomputed embeddings.
#             data["paper"].x = torch.from_numpy(features).to(torch.float32)
#         else:
#             data["paper"].x = torch.load(path_embs).type(torch.float32)

#         print("Loaded pre-trained node embeddings of type={} and shape={}.".format(params["node_embs"], data["paper"].x.shape))

#     data.to(DEVICE)

#     # --- 6. 🎛️ NEW: Systematic LP Alpha Tuning (Single Test) ---
#     # Define the sweep values as requested
#     lp_alpha_sweep = [0.1] # Test just one value first as requested
    
#     # Store results for *all* sweeps
#     all_sweep_results = {}
    
#     # Check if a value is already in config, add it if not
#     current_config_lp_alpha = params["model"].get("lp_alpha", 0.9) # Get default
#     if current_config_lp_alpha not in lp_alpha_sweep:
#         # Add the default 0.9 to compare against
#         lp_alpha_sweep.insert(0, current_config_lp_alpha) 
    
#     print(f"--- Starting LP Alpha Sweep for alpha = {lp_alpha_sweep} ---")
    
#     # --- Set a default DropEdge prob (based on previous good run) ---
#     # We are no longer sweeping DropEdge, so set a good default.
#     params["model"]["dropedge_prob"] = 0.2
#     print(f"--- Using fixed dropedge_prob = {params['model']['dropedge_prob']} ---")
    
    
#     for lp_alpha in lp_alpha_sweep:
#         print(f"\n--- 🚀 STARTING SWEEP: lp_alpha = {lp_alpha} ---")
#         # Set this value in the params so the LP step can access it
#         params["model"]["lp_alpha"] = lp_alpha
        
#         path_to_model = f"models/{params['dataset']}_{params['model']['name']}_lp{lp_alpha}"
#         all_runs_accs = []
    
#         for run in range(params["runs"]):
#             # Pass data to get_model for homophily calculation if needed
#             model = get_model(verbose=True) if run == 0 else get_model()
            
#             if params["dataset"] == "ogbnarxiv" or (params["dataset"] == "pubmed" and data_gen_params["pubmed_fixed_split"]):
#                 train_idx, val_idx, test_idx = data["paper"].train_idx, data["paper"].val_idx, data["paper"].test_idx
#             else: # Randomly split nodes of each class; not compatible with SimTG (fixed split is required to finetune the LM).
#                 data.to(torch.device("cpu"))
#                 train_idx, val_idx, test_idx = utils.per_class_idx_split(data, run) # use run no. as seed.
#                 data.to(DEVICE)
    
#             optimizer = torch.optim.Adam(params=model.parameters(), 
#                 weight_decay=params["optimizer"]["weight_decay"], 
#                 lr=params["optimizer"]["lr"])
            
#             scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    
#             best_acc = best_epoch = -1
#             for epoch in tqdm(range(params["epochs"]), desc=f"Run {run:02d}"):
#                 train_loss, out = train(model, data, train_idx, optimizer)
    
#                 val_acc, val_loss = test(model, data, val_idx, params["dataset"], out=out)
#                 scheduler.step()
    
#                 if val_acc > best_acc:
#                     best_acc = val_acc
#                     best_epoch = epoch
#                     torch.save(model.state_dict(), f"{path_to_model}_run{run}_best.pth")
#                 elif epoch - best_epoch > params["early_stop_threshold"]:
#                     tqdm.write(f"Early stopped training for run {run:02d} at epoch {epoch:02d}.")
#                     break
    
#             model.load_state_dict(torch.load(f"{path_to_model}_run{run}_best.pth"))
            
#             # --- 7. 🚀 Get GNN Test Acc AND all_logits from modified test() ---
#             test_acc, test_loss, all_logits = test(model, data, test_idx, params["dataset"])
    
#             # --- 8. 🚀 Apply Label Propagation Post-Processing ---
#             # print(f"Run {run:02d}: Applying Label Propagation...")
#             with torch.no_grad():
#                 # GNN output is logits, convert to probabilities
#                 all_probs = F.softmax(all_logits, dim=1) 
    
#             # Find the 'references' subgraph, as requested
#             lp_edge_index = None
#             for edge_type in data.metadata()[1]: # data.metadata()[1] is list of edge types
#                 if 'references' in edge_type[1]:
#                     # print(f"    Using edge type '{edge_type}' for LP.")
#                     adj_t = data.adj_t_dict[edge_type]
#                     row, col, _ = adj_t.coo()
#                     lp_edge_index = torch.stack([row, col], dim=0)
#                     break
            
#             if lp_edge_index is None:
#                 print("    Warning: 'references' edge type not found. Using all edges for LP.")
#                 homo_data = data.to_homogeneous()
#                 lp_edge_index = homo_data.edge_index
            
#             lp_edge_index = lp_edge_index.to(DEVICE)
    
#             lp_model = LabelPropagation(
#                 num_layers=params["model"].get("lp_layers", 10), 
#                 alpha=params["model"].get("lp_alpha", 0.9) # This will now be set by the sweep
#             ).to(DEVICE)
            
#             # Propagate the GNN's probability predictions
#             propagated_probs = lp_model(all_probs, lp_edge_index)
            
#             # Evaluate the new propagated probabilities on the test set
#             y_pred_lp = propagated_probs[test_idx].argmax(dim=-1, keepdim=True)
            
#             if params["dataset"] == "ogbnarxiv":
#                 evaluator = Evaluator(name="ogbn-arxiv")
#                 lp_test_acc = evaluator.eval({
#                     "y_true": data["paper"].y[test_idx], 
#                     "y_pred": y_pred_lp
#                 })["acc"]
#             else:
#                 lp_test_acc = int((y_pred_lp == data["paper"].y[test_idx]).sum()) / int(test_idx.shape[0])
    
#             # --- 9. 🚀 NEW: Log both accuracies and store the LP result ---
#             tqdm.write(f"Run {run:02d}: Best Epoch {best_epoch:02d}, Best Val Acc {best_acc:.4f}")
#             tqdm.write(f"    -> GNN Test Acc: {test_acc:.4f}")
#             tqdm.write(f"    -> GNN+LP Test Acc: {lp_test_acc:.4f}")
            
#             all_runs_accs.append([best_acc, lp_test_acc]) # Store GNN Val, LP Test
#             torch.cuda.empty_cache()
        
#         # --- 10. 🎛️ NEW: Summarize results for THIS lp_alpha ---
#         all_runs_accs = torch.tensor(all_runs_accs)
#         print(f"* === SUMMARY for lp_alpha = {lp_alpha} ===")
#         avg_val_acc = all_runs_accs[:, 0].mean().item() * 100
#         std_val_acc = all_runs_accs[:, 0].std().item() * 100
#         avg_test_acc = all_runs_accs[:, 1].mean().item() * 100
#         std_test_acc = all_runs_accs[:, 1].std().item() * 100
        
#         print(f"  Avg. Val Acc: {avg_val_acc:.2f} ± {std_val_acc:.2f}", 
#               f"Avg. Test Acc: {avg_test_acc:.2f} ± {std_test_acc:.2f}.")
        
#         # Store for final comparison
#         all_sweep_results[lp_alpha] = {
#             "avg_val": avg_val_acc, "std_val": std_val_acc,
#             "avg_test": avg_test_acc, "std_test": std_test_acc
#         }
    
#     # --- 11. 🎛️ NEW: Print final summary of all sweeps ---
#     print("\n* =====================================================")
#     print("* 🎛️ FINAL SWEEP RESULTS (Avg. GNN+LP Test Acc ± Std. Dev.)")
#     print("* =====================================================")
#     best_lp_alpha = -1
#     best_test_acc = -1
    
#     # Sort by test accuracy
#     sorted_results = sorted(
#         all_sweep_results.items(), 
#         key=lambda item: item[1]['avg_test'], 
#         reverse=True
#     )
    
#     for i, (lp_alpha, results) in enumerate(sorted_results):
#         prefix = "🏆 Best" if i == 0 else "  "
#         print(f"  {prefix} alpha={lp_alpha:.2f}: {results['avg_test']:.2f} ± {results['std_test']:.2f} (Val: {results['avg_val']:.2f})")
#         if i == 0:
#             best_test_acc = results['avg_test']
#             best_lp_alpha = lp_alpha
            
#     print(f"\n--- 🏆 Best Setting: lp_alpha = {best_lp_alpha} (Test Acc: {best_test_acc:.2f}) ---")

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Will run on: {DEVICE}.")

    project_root: Path = utils.get_project_root()
    os.chdir(project_root)
    with open(str(project_root / "config/experiments_config.yaml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    with open(str(project_root / "config/data_generation_config.yaml")) as f:
        data_gen_params = yaml.load(f, Loader=yaml.FullLoader)

    path_to_data = str(Path(params["data"]["graph_dataset"][params["dataset"]]))
    # Set weights_only=False to load PyG data objects, as per torch 2.6+ update
    data = torch.load(path_to_data, weights_only=False)

    data = data.edge_type_subgraph(utils.edge_type_selection(params["edge_type_selection"][params["dataset"]]))

    if any(i in params["node_embs"] for i in ["simtg", "tape"]):
        path_embs = str(Path(params["data"][f"{params['node_embs']}_embs"][params["dataset"]]))
        if params["node_embs"] == "tape":
            init_x_shape = (19717, 768) if params["dataset"] == "pubmed" else (data["paper"].num_nodes, 768)
            features = np.array(np.memmap(path_embs, mode='r', dtype=np.float16, shape=init_x_shape))
            if params["dataset"] == "pubmed":
                features = np.delete(features, [2459], axis=0) # manually remove index corresponding to ID with no metadata from the precomputed embeddings.
            data["paper"].x = torch.from_numpy(features).to(torch.float32)
        else:
            data["paper"].x = torch.load(path_embs).type(torch.float32)

        print("Loaded pre-trained node embeddings of type={} and shape={}.".format(params["node_embs"], data["paper"].x.shape))

    data.to(DEVICE)

    path_to_model = f"models/{params['dataset']}_{params['model']['name']}"
    all_runs_accs = []

    for run in range(params["runs"]):
        # Pass data to get_model for homophily calculation if needed
        model = get_model(verbose=True) if run == 0 else get_model()
        
        if params["dataset"] == "ogbnarxiv" or (params["dataset"] == "pubmed" and data_gen_params["pubmed_fixed_split"]):
            train_idx, val_idx, test_idx = data["paper"].train_idx, data["paper"].val_idx, data["paper"].test_idx
        else: # Randomly split nodes of each class; not compatible with SimTG (fixed split is required to finetune the LM).
            data.to(torch.device("cpu"))
            train_idx, val_idx, test_idx = utils.per_class_idx_split(data, run) # use run no. as seed.
            data.to(DEVICE)

        optimizer = torch.optim.Adam(params=model.parameters(), 
            weight_decay=params["optimizer"]["weight_decay"], 
            lr=params["optimizer"]["lr"])
        
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

        best_acc = best_epoch = -1
        for epoch in tqdm(range(params["epochs"]), desc=f"Run {run:02d}"):
            train_loss, out = train(model, data, train_idx, optimizer)

            val_acc, val_loss = test(model, data, val_idx, params["dataset"], out=out)
            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), f"{path_to_model}_run{run}_best.pth")
            elif epoch - best_epoch > params["early_stop_threshold"]:
                tqdm.write(f"Early stopped training for run {run:02d} at epoch {epoch:02d}.")
                break

        model.load_state_dict(torch.load(f"{path_to_model}_run{run}_best.pth"))
        
        # --- This is the original test call, expecting 2 return values ---
        test_acc, test_loss = test(model, data, test_idx, params["dataset"])

        tqdm.write(f"Run {run:02d}: Best Epoch {best_epoch:02d}, Best Val Acc {best_acc:.4f}, Test Acc {test_acc:.4f}.")
        all_runs_accs.append([best_acc, test_acc]) # Store Val and Test
        torch.cuda.empty_cache()
    
    # --- This is the original summary ---
    all_runs_accs = torch.tensor(all_runs_accs)
    print("* ============================= ALL RUNS =============================")
    print(f"Best Val Acc: {all_runs_accs[:, 0].max().item()*100:.2f}, Best Test Acc: {all_runs_accs[:, 1].max().item()*100:.2f}.")
    print(f"Avg. Val Acc: {all_runs_accs[:, 0].mean().item()*100:.2f} ± {all_runs_accs[:, 0].std().item()*100:.2f}", 
          f"Avg. Test Acc: {all_runs_accs[:, 1].mean().item()*100:.2f} ± {all_runs_accs[:, 1].std().item()*100:.2f}.")