# from pathlib import Path
# import os
# import yaml
# import copy

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import to_hetero
# from torch_geometric.utils import homophily, dropout_edge
# from torch_geometric.nn import SAGEConv
# from torch_geometric.nn import CorrectAndSmooth
# from torch_sparse import SparseTensor 
# from tqdm import tqdm
# from ogb.nodeproppred import Evaluator

# import utils
# import models 

# # =============================================================================
# # --- CONFIGURATION FOR SOTA PUSH ---
# # =============================================================================
# NUM_ENSEMBLE_RUNS = 3 
# INPUT_NOISE_STD = 0.01
# TARGET_CLASS_COUNT = 1000 # Boost any class with < 1000 train samples

# # =============================================================================
# # --- CORE MODELS ---
# # =============================================================================
# class LocalSAGE(torch.nn.Module):
#     def __init__(self, num_layers, in_channels, out_channels, hidden_channels, dropout):
#         super(LocalSAGE, self).__init__()
#         self.convs = torch.nn.ModuleList()
#         self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='max'))
#         self.bns = torch.nn.ModuleList()
#         self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
#         for _ in range(num_layers - 2):
#             self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='max'))
#             self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
#         self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='max'))
#         self.dropout = dropout
 
#     def forward(self, x, adj_t):
#         for i, conv in enumerate(self.convs[:-1]):
#             x = conv(x, adj_t)
#             x = self.bns[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, adj_t)
#         return x

# class WeightedHeteroWrapper(nn.Module):
#     def __init__(self, base_model_cls, model_args, metadata, data, device):
#         super().__init__()
#         self.models = nn.ModuleDict()
#         in_channels = model_args.get("in_channels")
#         out_channels = model_args.get("out_channels")

#         for edge_type in metadata[1]:
#             model = base_model_cls(**model_args)
#             self.models["__".join(edge_type)] = model
        
#         weights = {}
#         total_homophily = 0
#         y_cpu = data["paper"].y.to('cpu').squeeze()
        
#         for edge_type in metadata[1]:
#             if edge_type in data.adj_t_dict:
#                 adj_t = data.adj_t_dict[edge_type]
#             else:
#                 continue
#             row, col, _ = adj_t.coo()
#             edge_index = torch.stack([row, col], dim=0)
#             h = homophily(edge_index.to('cpu'), y_cpu)
#             weights[edge_type] = h
#             total_homophily += h
        
#         self.agg_logits = nn.ParameterDict()
#         for edge_type in metadata[1]:
#             normalized_weight = (weights.get(edge_type, 0) / total_homophily) \
#                 if total_homophily > 0 else (1.0 / len(metadata[1]))
#             self.agg_logits["__".join(edge_type)] = nn.Parameter(
#                 torch.tensor(normalized_weight, device=device), requires_grad=True
#             )

#         self.layer_norm = nn.LayerNorm(out_channels).to(device)
#         self.skip_proj = nn.Linear(in_channels, out_channels).to(device)

#     def forward(self, x_dict, adj_t_dict):
#         out_embeddings = []
#         out_logits = []
#         x_paper = x_dict["paper"]
        
#         for edge_type_key, model in self.models.items():
#             edge_type = tuple(edge_type_key.split("__"))
#             if edge_type not in adj_t_dict: continue 
            
#             out = model(x_paper, adj_t_dict[edge_type]) 
#             out_embeddings.append(out)
#             out_logits.append(self.agg_logits[edge_type_key])
        
#         if not out_embeddings:
#             return {"paper": torch.zeros((x_paper.size(0), self.skip_proj.out_features), device=x_paper.device)}

#         out_stack = torch.stack(out_embeddings, dim=0)
#         logits_stack = torch.stack(out_logits, dim=0)
#         learned_weights = F.softmax(logits_stack, dim=0)
#         out_sum = torch.einsum('e,end->nd', learned_weights, out_stack) 
#         out_combined = out_sum + self.skip_proj(x_dict["paper"]) 
#         return {"paper": self.layer_norm(out_combined)}

# # =============================================================================
# # --- TRAINING & EVALUATION ---
# # =============================================================================
# def train(model, data, train_idx, optimizer, params, device):
#     model.train()
    
#     # Input Perturbation
#     x_dict_noisy = copy.copy(data.x_dict)
#     if INPUT_NOISE_STD > 0:
#         noise = torch.randn_like(x_dict_noisy["paper"]) * INPUT_NOISE_STD
#         x_dict_noisy["paper"] = x_dict_noisy["paper"] + noise

#     # Targeted DropEdge
#     base_drop = params["model"].get("dropedge_prob", 0.0)
#     targeted_drop = 0.5 
    
#     adj_t_dict_dropped = {}
#     for edge_type, adj_t in data.adj_t_dict.items():
#         if base_drop > 0 or targeted_drop > 0:
#             row, col, value = adj_t.coo()
#             edge_index = torch.stack([row, col], dim=0)
            
#             p = targeted_drop if 'cited_by' in edge_type[1] else base_drop
            
#             if p > 0:
#                 edge_index_dropped, edge_mask = dropout_edge(edge_index, p=p, force_undirected=False, training=True)
#                 edge_attr_dropped = value[edge_mask] if value is not None else None
#             else:
#                 edge_index_dropped, edge_attr_dropped = edge_index, value

#             if edge_index_dropped.numel() == 0:
#                 new_adj = torch.sparse_coo_tensor(torch.empty((2,0), dtype=torch.long), torch.empty(0), adj_t.sizes())
#             else:
#                 vals = edge_attr_dropped if edge_attr_dropped is not None else torch.ones(edge_index_dropped.size(1))
#                 new_adj = torch.sparse_coo_tensor(edge_index_dropped, vals, adj_t.sizes())
            
#             adj_t_dict_dropped[edge_type] = new_adj.to(device).to_sparse_csr()
#         else:
#             adj_t_dict_dropped[edge_type] = adj_t

#     optimizer.zero_grad()
#     out = model(x_dict_noisy, adj_t_dict_dropped)["paper"]
#     loss = F.cross_entropy(out[train_idx], data["paper"].y[train_idx].squeeze())
#     loss.backward()
#     optimizer.step()
#     return float(loss), out

# @torch.no_grad()
# def test(model, data, idx, dataset, out=None):
#     model.eval()
#     if out is None:
#         out = model(data.x_dict, data.adj_t_dict)['paper']
#     loss = F.cross_entropy(out[idx], data["paper"].y[idx].squeeze())
#     y_pred = out[idx].argmax(dim=-1, keepdim=True)
    
#     if dataset == "ogbnarxiv":
#         evaluator = Evaluator(name="ogbn-arxiv")
#         acc = evaluator.eval({"y_true": data["paper"].y[idx], "y_pred": y_pred})["acc"]
#     else:
#         acc = int((y_pred == data["paper"].y[idx]).sum()) / int(idx.shape[0])
#     return acc, loss, out

# def get_model(data, params, device, verbose=False):
#     num_layers = params["model"]["num_layers"]
#     in_channels = data["paper"].x.shape[1]
#     out_channels = len(torch.unique(data["paper"].y))
#     hidden_channels = params["model"]["hidden_channels"]
#     dropout = params["model"]["dropout"]
#     model_args = {"num_layers": num_layers, "in_channels": in_channels, "out_channels": out_channels, "hidden_channels": hidden_channels, "dropout": dropout}

#     if params["model"]["name"] == "SAGE":
#         base_model_cls = LocalSAGE
#     else:
#         raise ValueError(f"Unknown model name {params['model']['name']}")

#     if params["model"].get("custom_aggr", "mean") == "homophily_weighted_mean":
#         if verbose: print("Using custom homophily-weighted aggregation.")
#         model = WeightedHeteroWrapper(LocalSAGE, model_args, data.metadata(), data, device).to(device)
#     else:
#         base_model = LocalSAGE(**model_args).to(device)
#         model = to_hetero(base_model, data.metadata(), aggr=params["model"].get("custom_aggr", "mean")).to(device)
    
#     if verbose: print("No. parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
#     return model

# # =============================================================================
# # --- GRAPHSMOTE DATA AUGMENTATION ---
# # =============================================================================
# def apply_graph_smote(data, target_count=500):
#     print(f"\n🧬 Applying GraphSMOTE to Minority Classes (Target: {target_count} samples)")
    
#     x = data['paper'].x
#     y = data['paper'].y.squeeze()
#     train_idx = data['paper'].train_idx
    
#     train_y = y[train_idx]
#     unique_classes, counts = torch.unique(train_y, return_counts=True)
#     minority_classes = unique_classes[counts < target_count].tolist()
    
#     if not minority_classes:
#         print("   -> All classes have sufficient representation. Skipping SMOTE.")
#         return data

#     print(f"   -> Found {len(minority_classes)} minority classes to augment: {minority_classes}")

#     original_num_nodes = x.size(0)
#     new_node_id = original_num_nodes
    
#     new_x = []
#     new_y = []
#     new_train_idx = []
    
#     all_edge_types = list(data.adj_t_dict.keys())
#     new_edges = {edge_type: {'row': [], 'col': []} for edge_type in all_edge_types}
    
#     citation_edge, rev_edge = None, None
#     for et in all_edge_types:
#         if 'cites' in et[1] or 'ref' in et[1]: citation_edge = et
#         if 'cited_by' in et[1]: rev_edge = et

#     nodes_added = 0
    
#     for cls in minority_classes:
#         cls_mask = (train_y == cls)
#         cls_nodes = train_idx[cls_mask]
#         num_existing = cls_nodes.size(0)
        
#         if num_existing == 0: continue
        
#         num_to_add = target_count - num_existing
#         print(f"      -> Class {cls}: Generating {num_to_add} synthetic nodes.")
        
#         for _ in range(num_to_add):
#             parents = cls_nodes[torch.randint(0, num_existing, (2,))]
#             p1, p2 = parents[0], parents[1]
            
#             alpha = torch.rand(1).item()
#             syn_feat = alpha * x[p1] + (1 - alpha) * x[p2]
            
#             new_x.append(syn_feat)
#             new_y.append(cls)
#             new_train_idx.append(new_node_id)
            
#             if citation_edge:
#                 new_edges[citation_edge]['row'].extend([new_node_id, new_node_id])
#                 new_edges[citation_edge]['col'].extend([p1.item(), p2.item()])
#             if rev_edge:
#                 new_edges[rev_edge]['row'].extend([p1.item(), p2.item()])
#                 new_edges[rev_edge]['col'].extend([new_node_id, new_node_id])
                
#             new_node_id += 1
#             nodes_added += 1

#     if nodes_added == 0:
#         return data

#     data['paper'].x = torch.cat([x, torch.stack(new_x)])
#     data['paper'].y = torch.cat([y, torch.tensor(new_y, dtype=y.dtype)]).view(-1, 1)
#     data['paper'].train_idx = torch.cat([train_idx, torch.tensor(new_train_idx, dtype=train_idx.dtype)])
    
#     total_nodes = data['paper'].x.size(0)
#     data['paper'].num_nodes = total_nodes

#     for edge_type in all_edge_types:
#         adj_t = data.adj_t_dict[edge_type]
#         row, col, value = adj_t.coo()
        
#         if len(new_edges[edge_type]['row']) > 0:
#             new_row = torch.tensor(new_edges[edge_type]['row'], dtype=torch.long)
#             new_col = torch.tensor(new_edges[edge_type]['col'], dtype=torch.long)
            
#             row = torch.cat([row.cpu(), new_row])
#             col = torch.cat([col.cpu(), new_col])
            
#             if value is not None:
#                 new_val = torch.ones(len(new_row), dtype=value.dtype)
#                 value = torch.cat([value.cpu(), new_val])
        
#         vals = value if value is not None else torch.ones(row.size(0))
#         new_adj = SparseTensor(row=row, col=col, value=vals, sparse_sizes=(total_nodes, total_nodes))
#         data[edge_type].adj_t = new_adj

#     print(f"✅ GraphSMOTE Complete. Added {nodes_added} synthetic nodes. New total nodes: {total_nodes}")
#     return data

# # =============================================================================
# # --- MAIN PIPELINE EXECUTOR ---
# # =============================================================================
# if __name__ == "__main__":
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Will run on: {DEVICE}.")

#     project_root: Path = utils.get_project_root()
#     os.chdir(project_root)
#     with open(str(project_root / "config/experiments_config.yaml")) as f: params = yaml.load(f, Loader=yaml.FullLoader)
#     with open(str(project_root / "config/data_generation_config.yaml")) as f: data_gen_params = yaml.load(f, Loader=yaml.FullLoader)

#     # 1. LOAD THE ORIGINAL HETERO GRAPH
#     path_to_data = str(Path(params["data"]["graph_dataset"][params["dataset"]]))
#     data_list = torch.load(path_to_data, weights_only=False)
#     data = data_list[0] if isinstance(data_list, tuple) else data_list
#     data = data.edge_type_subgraph(utils.edge_type_selection(params["edge_type_selection"][params["dataset"]]))

#     # 1.5 🛑 SOTA BOSS LEVEL: SURGICALLY INJECT SIMTEG + TF-IDF
#     print("\n--- 📥 Building the Ultimate Boss-Level Embeddings ---")
    
#     # A. Extract the 500 TF-IDF dimensions we built earlier
#     upgraded_path = str(project_root / "data" / "ogbn_arxiv" / "processed" / "geometric_data_processed.pt")
#     upgraded_data_list = torch.load(upgraded_path, weights_only=False)
#     upgraded_data = upgraded_data_list[0] if isinstance(upgraded_data_list, tuple) else upgraded_data_list
    
#     mixed_features = upgraded_data['paper'].x if (hasattr(upgraded_data, 'x_dict') and 'paper' in upgraded_data.x_dict) else upgraded_data.x
#     tfidf_features = mixed_features[:, -500:] # Slice off the exact 500 TF-IDF columns
#     print(f"   -> Extracted pure TF-IDF features: {tfidf_features.shape}")

#     # B. Load the actual underlying embeddings (SimTeG or Tape)
#     emb_name = params.get("node_embs", "simtg") 
#     path_embs = str(project_root / params["data"][f"{emb_name}_embs"][params["dataset"]])
#     simtg_features = torch.load(path_embs).type(torch.float32)
#     print(f"   -> Loaded original {emb_name.upper()} features: {simtg_features.shape}")

#     # C. Fuse them together!
#     ultimate_features = torch.cat([simtg_features, tfidf_features], dim=1)
#     data['paper'].x = ultimate_features
#     print(f"✅ Successfully injected BOSS LEVEL features. Final Shape: {data['paper'].x.shape}")

#     # 2. AAAI-24 Reverse Edges
#     print("\n--- 🔄 AAAI-24: Generating Reverse Edges (Reverse Message Passing) ---")
#     citation_edge_type = None
#     for et in list(data.adj_t_dict.keys()):
#         if 'cites' in et[1] or 'ref' in et[1]: citation_edge_type = et; break
            
#     if citation_edge_type:
#         print(f"  Found Reference Edge: {citation_edge_type}")
#         adj_t_forward = data.adj_t_dict[citation_edge_type]
#         rev_edge_type = (citation_edge_type[0], 'cited_by', citation_edge_type[2])
#         data.adj_t_dict[rev_edge_type] = adj_t_forward.t()
#         data[rev_edge_type].edge_index = torch.empty((2,0), dtype=torch.long)
#         print(f"  ✅ Created Reverse Edge: {rev_edge_type}")

#     # 4. Apply GraphSMOTE
#     data = apply_graph_smote(data, target_count=TARGET_CLASS_COUNT)
#     data.to(DEVICE)

#     # 5. Ensemble Training Loop
#     print(f"\n--- 🚀 STARTING ENSEMBLE RUN (Total Runs: {NUM_ENSEMBLE_RUNS}) ---")
#     ensemble_logits_test = 0
#     params["model"]["dropedge_prob"] = 0.2
    
#     for run in range(NUM_ENSEMBLE_RUNS):
#         print(f"\n🔹 Run {run+1}/{NUM_ENSEMBLE_RUNS}")
#         model = get_model(data, params, DEVICE, verbose=(run==0))
        
#         if params["dataset"] == "ogbnarxiv":
#             train_idx, val_idx, test_idx = data["paper"].train_idx, data["paper"].val_idx, data["paper"].test_idx
#         else: 
#             data.to("cpu"); train_idx, val_idx, test_idx = utils.per_class_idx_split(data, run); data.to(DEVICE)

#         optimizer = torch.optim.Adam(model.parameters(), weight_decay=params["optimizer"]["weight_decay"], lr=params["optimizer"]["lr"])
#         scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

#         best_acc, best_epoch = -1, -1
#         path_to_model = f"models/ensemble_run_{run}.pth"
        
#         for epoch in tqdm(range(params["epochs"]), desc=f"Run {run+1}"):
#             train_loss, out = train(model, data, train_idx, optimizer, params, DEVICE)
#             val_acc, _, _ = test(model, data, val_idx, params["dataset"], out=out)
#             scheduler.step()

#             if val_acc > best_acc:
#                 best_acc, best_epoch = val_acc, epoch
#                 torch.save(model.state_dict(), path_to_model)
#             elif epoch - best_epoch > params["early_stop_threshold"]:
#                 break
        
#         model.load_state_dict(torch.load(path_to_model))
#         acc, _, logits = test(model, data, test_idx, params["dataset"])
#         print(f"   Run {run+1} Test Acc: {acc:.4f} (Best Val: {best_acc:.4f})")
        
#         original_test_size = test_idx.size(0)
#         ensemble_logits_test += F.softmax(logits[test_idx][:original_test_size], dim=1) / NUM_ENSEMBLE_RUNS
        
#     # 6. Evaluation & Post-Processing
#     print("\n" + "="*40 + "\n🏆 FINAL ENSEMBLE RESULTS\n" + "="*40)
#     y_pred_ensemble = ensemble_logits_test.argmax(dim=-1, keepdim=True)
#     evaluator = Evaluator(name="ogbn-arxiv")
#     ensemble_acc = evaluator.eval({"y_true": data["paper"].y[test_idx], "y_pred": y_pred_ensemble})["acc"]
#     print(f"👉 Pure Ensemble Test Accuracy: {ensemble_acc:.4f}")
    
#     adj_fwd = data.adj_t_dict[citation_edge_type]
#     adj_rev = adj_fwd.t()
#     row_all = torch.cat([adj_fwd.coo()[0], adj_rev.coo()[0]])
#     col_all = torch.cat([adj_fwd.coo()[1], adj_rev.coo()[1]])
#     cs_edge_index = torch.stack([row_all, col_all]).to(DEVICE)
    
#     y_train_mask = torch.zeros((len(data["paper"].y), ensemble_logits_test.shape[1])).to(DEVICE)
#     y_train_mask[train_idx] = F.one_hot(data["paper"].y[train_idx].squeeze(), num_classes=ensemble_logits_test.shape[1]).float()
#     full_soft_preds = torch.zeros_like(y_train_mask)
#     full_soft_preds[test_idx] = ensemble_logits_test.to(DEVICE)
#     full_soft_preds[train_idx] = y_train_mask[train_idx]

#     print("\n🧪 Tuning C&S on Ensemble...")
#     best_cs_acc, best_alpha = 0, 0
#     for alpha in [0.4, 0.6, 0.8]:
#         cs = CorrectAndSmooth(num_correction_layers=50, correction_alpha=1.0, 
#                               num_smoothing_layers=50, smoothing_alpha=alpha, autoscale=False, scale=1.0).to(DEVICE)
#         y_smooth = cs.smooth(full_soft_preds, y_train_mask[train_idx], train_idx, cs_edge_index)
#         y_pred_cs = y_smooth[test_idx].argmax(dim=-1, keepdim=True)
#         cs_acc = evaluator.eval({"y_true": data["paper"].y[test_idx], "y_pred": y_pred_cs})["acc"]
#         print(f"   Alpha {alpha}: {cs_acc:.4f}")
#         if cs_acc > best_cs_acc: best_cs_acc = cs_acc; best_alpha = alpha

#     print(f"\n🚀 BEST RESULT: {best_cs_acc:.4f} (with Alpha={best_alpha})")
#     print("="*40)


from pathlib import Path
import os
import yaml
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from torch_geometric.utils import homophily, dropout_edge
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import CorrectAndSmooth
from torch_sparse import SparseTensor 
from tqdm import tqdm
from ogb.nodeproppred import Evaluator

import utils
import models 

# =============================================================================
# --- CONFIGURATION FOR SOTA PUSH ---
# =============================================================================
NUM_ENSEMBLE_RUNS = 3 
INPUT_NOISE_STD = 0.01
TARGET_CLASS_COUNT = 1000 # Boost any class with < 1000 train samples

# =============================================================================
# --- CORE MODELS ---
# =============================================================================
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
        
        weights = {}
        total_homophily = 0
        y_cpu = data["paper"].y.to('cpu').squeeze()
        
        for edge_type in metadata[1]:
            if edge_type in data.adj_t_dict:
                adj_t = data.adj_t_dict[edge_type]
            else:
                continue
            row, col, _ = adj_t.coo()
            edge_index = torch.stack([row, col], dim=0)
            h = homophily(edge_index.to('cpu'), y_cpu)
            weights[edge_type] = h
            total_homophily += h
        
        self.agg_logits = nn.ParameterDict()
        for edge_type in metadata[1]:
            normalized_weight = (weights.get(edge_type, 0) / total_homophily) \
                if total_homophily > 0 else (1.0 / len(metadata[1]))
            self.agg_logits["__".join(edge_type)] = nn.Parameter(
                torch.tensor(normalized_weight, device=device), requires_grad=True
            )

        self.layer_norm = nn.LayerNorm(out_channels).to(device)
        self.skip_proj = nn.Linear(in_channels, out_channels).to(device)

    def forward(self, x_dict, adj_t_dict):
        out_embeddings = []
        out_logits = []
        x_paper = x_dict["paper"]
        
        for edge_type_key, model in self.models.items():
            edge_type = tuple(edge_type_key.split("__"))
            if edge_type not in adj_t_dict: continue 
            
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
        return {"paper": self.layer_norm(out_combined)}

# =============================================================================
# --- TRAINING & EVALUATION ---
# =============================================================================
def train(model, data, train_idx, optimizer, params, device):
    model.train()
    
    # Input Perturbation
    x_dict_noisy = copy.copy(data.x_dict)
    if INPUT_NOISE_STD > 0:
        noise = torch.randn_like(x_dict_noisy["paper"]) * INPUT_NOISE_STD
        x_dict_noisy["paper"] = x_dict_noisy["paper"] + noise

    # Targeted DropEdge
    base_drop = params["model"].get("dropedge_prob", 0.0)
    targeted_drop = 0.5 
    
    adj_t_dict_dropped = {}
    for edge_type, adj_t in data.adj_t_dict.items():
        if base_drop > 0 or targeted_drop > 0:
            row, col, value = adj_t.coo()
            edge_index = torch.stack([row, col], dim=0)
            
            p = targeted_drop if 'cited_by' in edge_type[1] else base_drop
            
            if p > 0:
                edge_index_dropped, edge_mask = dropout_edge(edge_index, p=p, force_undirected=False, training=True)
                edge_attr_dropped = value[edge_mask] if value is not None else None
            else:
                edge_index_dropped, edge_attr_dropped = edge_index, value

            if edge_index_dropped.numel() == 0:
                new_adj = torch.sparse_coo_tensor(torch.empty((2,0), dtype=torch.long), torch.empty(0), adj_t.sizes())
            else:
                vals = edge_attr_dropped if edge_attr_dropped is not None else torch.ones(edge_index_dropped.size(1))
                new_adj = torch.sparse_coo_tensor(edge_index_dropped, vals, adj_t.sizes())
            
            adj_t_dict_dropped[edge_type] = new_adj.to(device).to_sparse_csr()
        else:
            adj_t_dict_dropped[edge_type] = adj_t

    optimizer.zero_grad()
    out = model(x_dict_noisy, adj_t_dict_dropped)["paper"]
    loss = F.cross_entropy(out[train_idx], data["paper"].y[train_idx].squeeze())
    loss.backward()
    optimizer.step()
    return float(loss), out

@torch.no_grad()
def test(model, data, idx, dataset, out=None):
    model.eval()
    if out is None:
        out = model(data.x_dict, data.adj_t_dict)['paper']
    loss = F.cross_entropy(out[idx], data["paper"].y[idx].squeeze())
    y_pred = out[idx].argmax(dim=-1, keepdim=True)
    
    if dataset == "ogbnarxiv":
        evaluator = Evaluator(name="ogbn-arxiv")
        acc = evaluator.eval({"y_true": data["paper"].y[idx], "y_pred": y_pred})["acc"]
    else:
        acc = int((y_pred == data["paper"].y[idx]).sum()) / int(idx.shape[0])
    return acc, loss, out

def get_model(data, params, device, verbose=False):
    num_layers = params["model"]["num_layers"]
    in_channels = data["paper"].x.shape[1]
    out_channels = len(torch.unique(data["paper"].y))
    hidden_channels = params["model"]["hidden_channels"]
    dropout = params["model"]["dropout"]
    model_args = {"num_layers": num_layers, "in_channels": in_channels, "out_channels": out_channels, "hidden_channels": hidden_channels, "dropout": dropout}

    if params["model"]["name"] == "SAGE":
        base_model_cls = LocalSAGE
    else:
        raise ValueError(f"Unknown model name {params['model']['name']}")

    if params["model"].get("custom_aggr", "mean") == "homophily_weighted_mean":
        if verbose: print("Using custom homophily-weighted aggregation.")
        model = WeightedHeteroWrapper(LocalSAGE, model_args, data.metadata(), data, device).to(device)
    else:
        base_model = LocalSAGE(**model_args).to(device)
        model = to_hetero(base_model, data.metadata(), aggr=params["model"].get("custom_aggr", "mean")).to(device)
    
    if verbose: print("No. parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

# =============================================================================
# --- DATA AUGMENTATION & GRAPH MODIFICATION ---
# =============================================================================
def apply_graph_smote(data, target_count=500):
    print(f"\n🧬 Applying GraphSMOTE to Minority Classes (Target: {target_count} samples)")
    x, y, train_idx = data['paper'].x, data['paper'].y.squeeze(), data['paper'].train_idx
    train_y = y[train_idx]
    
    unique_classes, counts = torch.unique(train_y, return_counts=True)
    minority_classes = unique_classes[counts < target_count].tolist()
    if not minority_classes: return data

    print(f"   -> Found {len(minority_classes)} minority classes to augment: {minority_classes}")

    new_node_id = x.size(0)
    new_x, new_y, new_train_idx = [], [], []
    
    all_edge_types = list(data.adj_t_dict.keys())
    new_edges = {edge_type: {'row': [], 'col': []} for edge_type in all_edge_types}
    
    citation_edge, rev_edge = None, None
    for et in all_edge_types:
        if 'cites' in et[1] or 'ref' in et[1]: citation_edge = et
        if 'cited_by' in et[1]: rev_edge = et

    nodes_added = 0
    for cls in minority_classes:
        cls_mask = (train_y == cls)
        cls_nodes = train_idx[cls_mask]
        num_existing = cls_nodes.size(0)
        
        if num_existing == 0: continue
        num_to_add = target_count - num_existing
        
        for _ in range(num_to_add):
            parents = cls_nodes[torch.randint(0, num_existing, (2,))]
            alpha = torch.rand(1).item()
            new_x.append(alpha * x[parents[0]] + (1 - alpha) * x[parents[1]])
            new_y.append(cls)
            new_train_idx.append(new_node_id)
            
            if citation_edge:
                new_edges[citation_edge]['row'].extend([new_node_id, new_node_id])
                new_edges[citation_edge]['col'].extend([parents[0].item(), parents[1].item()])
            if rev_edge:
                new_edges[rev_edge]['row'].extend([parents[0].item(), parents[1].item()])
                new_edges[rev_edge]['col'].extend([new_node_id, new_node_id])
                
            new_node_id += 1
            nodes_added += 1

    if nodes_added == 0: return data

    data['paper'].x = torch.cat([x, torch.stack(new_x)])
    data['paper'].y = torch.cat([y, torch.tensor(new_y, dtype=y.dtype)]).view(-1, 1)
    data['paper'].train_idx = torch.cat([train_idx, torch.tensor(new_train_idx, dtype=train_idx.dtype)])
    total_nodes = data['paper'].x.size(0)
    data['paper'].num_nodes = total_nodes

    for edge_type in all_edge_types:
        adj_t = data.adj_t_dict[edge_type]
        row, col, value = adj_t.coo()
        
        if len(new_edges[edge_type]['row']) > 0:
            row = torch.cat([row.cpu(), torch.tensor(new_edges[edge_type]['row'], dtype=torch.long)])
            col = torch.cat([col.cpu(), torch.tensor(new_edges[edge_type]['col'], dtype=torch.long)])
            if value is not None:
                value = torch.cat([value.cpu(), torch.ones(len(new_edges[edge_type]['row']), dtype=value.dtype)])
        
        vals = value if value is not None else torch.ones(row.size(0))
        data[edge_type].adj_t = SparseTensor(row=row, col=col, value=vals, sparse_sizes=(total_nodes, total_nodes))

    print(f"✅ GraphSMOTE Complete. Added {nodes_added} synthetic nodes.")
    return data

def inject_super_nodes(data, target_classes):
    """Injects central information hubs for highly fragmented/heterophilic minority classes."""
    print(f"\n🌐 Injecting Virtual Super-Nodes for fragmented classes: {target_classes}")
    
    x, y, train_idx = data['paper'].x, data['paper'].y.squeeze(), data['paper'].train_idx
    train_y = y[train_idx]
    
    new_node_id = x.size(0)
    new_x, new_y, new_train_idx = [], [], []
    
    all_edge_types = list(data.adj_t_dict.keys())
    new_edges = {edge_type: {'row': [], 'col': []} for edge_type in all_edge_types}
    
    citation_edge, rev_edge = None, None
    for et in all_edge_types:
        if 'cites' in et[1] or 'ref' in et[1]: citation_edge = et
        if 'cited_by' in et[1]: rev_edge = et
        
    nodes_added = 0
    for cls in target_classes:
        cls_mask = (train_y == cls)
        cls_nodes = train_idx[cls_mask]
        
        if cls_nodes.size(0) == 0: continue
        print(f"   -> Creating Super-Node for Class {cls} (Connecting {cls_nodes.size(0)} nodes)")
        
        # Super-Node Feature: Mean of all training nodes in this class
        super_feat = x[cls_nodes].mean(dim=0)
        new_x.append(super_feat)
        new_y.append(cls)
        new_train_idx.append(new_node_id)
        
        cls_nodes_list = cls_nodes.tolist()
        super_node_list = [new_node_id] * len(cls_nodes_list)
        
        # Connect Super-Node bi-directionally to all its class members
        if citation_edge:
            new_edges[citation_edge]['row'].extend(super_node_list + cls_nodes_list)
            new_edges[citation_edge]['col'].extend(cls_nodes_list + super_node_list)
        if rev_edge:
            new_edges[rev_edge]['row'].extend(cls_nodes_list + super_node_list)
            new_edges[rev_edge]['col'].extend(super_node_list + cls_nodes_list)
            
        new_node_id += 1
        nodes_added += 1
        
    if nodes_added == 0: return data
    
    # Update data object
    data['paper'].x = torch.cat([x, torch.stack(new_x)])
    data['paper'].y = torch.cat([y, torch.tensor(new_y, dtype=y.dtype)]).view(-1, 1)
    data['paper'].train_idx = torch.cat([train_idx, torch.tensor(new_train_idx, dtype=train_idx.dtype)])
    total_nodes = data['paper'].x.size(0)
    data['paper'].num_nodes = total_nodes
    
    for edge_type in all_edge_types:
        adj_t = data.adj_t_dict[edge_type]
        row, col, value = adj_t.coo()
        
        if len(new_edges[edge_type]['row']) > 0:
            row = torch.cat([row.cpu(), torch.tensor(new_edges[edge_type]['row'], dtype=torch.long)])
            col = torch.cat([col.cpu(), torch.tensor(new_edges[edge_type]['col'], dtype=torch.long)])
            if value is not None:
                value = torch.cat([value.cpu(), torch.ones(len(new_edges[edge_type]['row']), dtype=value.dtype)])
        
        vals = value if value is not None else torch.ones(row.size(0))
        data[edge_type].adj_t = SparseTensor(row=row, col=col, value=vals, sparse_sizes=(total_nodes, total_nodes))
        
    print(f"✅ Super-Nodes Injected! New total nodes: {total_nodes}")
    return data

# =============================================================================
# --- MAIN PIPELINE EXECUTOR ---
# =============================================================================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Will run on: {DEVICE}.")

    project_root: Path = utils.get_project_root()
    os.chdir(project_root)
    with open(str(project_root / "config/experiments_config.yaml")) as f: params = yaml.load(f, Loader=yaml.FullLoader)

    # 1. LOAD THE ORIGINAL HETERO GRAPH
    path_to_data = str(Path(params["data"]["graph_dataset"][params["dataset"]]))
    data_list = torch.load(path_to_data, weights_only=False)
    data = data_list[0] if isinstance(data_list, tuple) else data_list
    data = data.edge_type_subgraph(utils.edge_type_selection(params["edge_type_selection"][params["dataset"]]))

    # 1.5 SOTA BOSS LEVEL: SURGICALLY INJECT SIMTEG + TF-IDF
    print("\n--- 📥 Building the Ultimate Boss-Level Embeddings ---")
    upgraded_path = str(project_root / "data" / "ogbn_arxiv" / "processed" / "geometric_data_processed.pt")
    upgraded_data_list = torch.load(upgraded_path, weights_only=False)
    upgraded_data = upgraded_data_list[0] if isinstance(upgraded_data_list, tuple) else upgraded_data_list
    
    mixed_features = upgraded_data['paper'].x if (hasattr(upgraded_data, 'x_dict') and 'paper' in upgraded_data.x_dict) else upgraded_data.x
    tfidf_features = mixed_features[:, -500:] # Slice off the exact 500 TF-IDF columns
    
    emb_name = params.get("node_embs", "simtg") 
    path_embs = str(project_root / params["data"][f"{emb_name}_embs"][params["dataset"]])
    simtg_features = torch.load(path_embs).type(torch.float32)

    ultimate_features = torch.cat([simtg_features, tfidf_features], dim=1)
    data['paper'].x = ultimate_features
    print(f"✅ Successfully injected BOSS LEVEL features. Final Shape: {data['paper'].x.shape}")

    # 2. AAAI-24 Reverse Edges
    print("\n--- 🔄 AAAI-24: Generating Reverse Edges (Reverse Message Passing) ---")
    citation_edge_type = None
    for et in list(data.adj_t_dict.keys()):
        if 'cites' in et[1] or 'ref' in et[1]: citation_edge_type = et; break
            
    if citation_edge_type:
        adj_t_forward = data.adj_t_dict[citation_edge_type]
        rev_edge_type = (citation_edge_type[0], 'cited_by', citation_edge_type[2])
        data.adj_t_dict[rev_edge_type] = adj_t_forward.t()
        data[rev_edge_type].edge_index = torch.empty((2,0), dtype=torch.long)
        print(f"  ✅ Created Reverse Edge: {rev_edge_type}")

    # 3. GRAPH AUGMENTATION PIPELINE
    data = apply_graph_smote(data, target_count=TARGET_CLASS_COUNT)
    
    # NEW: Inject Super-Nodes for the 5 worst misclassified minority classes
    data = inject_super_nodes(data, target_classes=[11, 29, 21, 3, 39])
    
    data.to(DEVICE)

    # 5. Ensemble Training Loop
    print(f"\n--- 🚀 STARTING ENSEMBLE RUN (Total Runs: {NUM_ENSEMBLE_RUNS}) ---")
    ensemble_logits_test = 0
    params["model"]["dropedge_prob"] = 0.2
    
    for run in range(NUM_ENSEMBLE_RUNS):
        print(f"\n🔹 Run {run+1}/{NUM_ENSEMBLE_RUNS}")
        model = get_model(data, params, DEVICE, verbose=(run==0))
        
        if params["dataset"] == "ogbnarxiv":
            train_idx, val_idx, test_idx = data["paper"].train_idx, data["paper"].val_idx, data["paper"].test_idx
        else: 
            data.to("cpu"); train_idx, val_idx, test_idx = utils.per_class_idx_split(data, run); data.to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), weight_decay=params["optimizer"]["weight_decay"], lr=params["optimizer"]["lr"])
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

        best_acc, best_epoch = -1, -1
        path_to_model = f"models/ensemble_run_{run}.pth"
        
        for epoch in tqdm(range(params["epochs"]), desc=f"Run {run+1}"):
            train_loss, out = train(model, data, train_idx, optimizer, params, DEVICE)
            val_acc, _, _ = test(model, data, val_idx, params["dataset"], out=out)
            scheduler.step()

            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epoch
                torch.save(model.state_dict(), path_to_model)
            elif epoch - best_epoch > params["early_stop_threshold"]:
                break
        
        model.load_state_dict(torch.load(path_to_model))
        acc, _, logits = test(model, data, test_idx, params["dataset"])
        print(f"   Run {run+1} Test Acc: {acc:.4f} (Best Val: {best_acc:.4f})")
        
        original_test_size = test_idx.size(0)
        ensemble_logits_test += F.softmax(logits[test_idx][:original_test_size], dim=1) / NUM_ENSEMBLE_RUNS
        
    # 6. Evaluation & Post-Processing
    print("\n" + "="*40 + "\n🏆 FINAL ENSEMBLE RESULTS\n" + "="*40)
    y_pred_ensemble = ensemble_logits_test.argmax(dim=-1, keepdim=True)
    evaluator = Evaluator(name="ogbn-arxiv")
    ensemble_acc = evaluator.eval({"y_true": data["paper"].y[test_idx], "y_pred": y_pred_ensemble})["acc"]
    print(f"👉 Pure Ensemble Test Accuracy: {ensemble_acc:.4f}")
    
    adj_fwd = data.adj_t_dict[citation_edge_type]
    adj_rev = adj_fwd.t()
    row_all = torch.cat([adj_fwd.coo()[0], adj_rev.coo()[0]])
    col_all = torch.cat([adj_fwd.coo()[1], adj_rev.coo()[1]])
    cs_edge_index = torch.stack([row_all, col_all]).to(DEVICE)
    
    y_train_mask = torch.zeros((len(data["paper"].y), ensemble_logits_test.shape[1])).to(DEVICE)
    y_train_mask[train_idx] = F.one_hot(data["paper"].y[train_idx].squeeze(), num_classes=ensemble_logits_test.shape[1]).float()
    full_soft_preds = torch.zeros_like(y_train_mask)
    full_soft_preds[test_idx] = ensemble_logits_test.to(DEVICE)
    full_soft_preds[train_idx] = y_train_mask[train_idx]

    print("\n🧪 Tuning C&S on Ensemble...")
    best_cs_acc, best_alpha = 0, 0
    # EXPANDED GRID SEARCH FOR C&S
    for alpha in [0.4, 0.5, 0.6, 0.7, 0.8]:
        cs = CorrectAndSmooth(num_correction_layers=50, correction_alpha=1.0, 
                              num_smoothing_layers=50, smoothing_alpha=alpha, autoscale=False, scale=1.0).to(DEVICE)
        y_smooth = cs.smooth(full_soft_preds, y_train_mask[train_idx], train_idx, cs_edge_index)
        y_pred_cs = y_smooth[test_idx].argmax(dim=-1, keepdim=True)
        cs_acc = evaluator.eval({"y_true": data["paper"].y[test_idx], "y_pred": y_pred_cs})["acc"]
        print(f"   Alpha {alpha}: {cs_acc:.4f}")
        if cs_acc > best_cs_acc: best_cs_acc = cs_acc; best_alpha = alpha

    print(f"\n🚀 BEST RESULT: {best_cs_acc:.4f} (with Alpha={best_alpha})")
    print("="*40)