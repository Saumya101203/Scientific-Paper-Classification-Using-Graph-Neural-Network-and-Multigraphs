from pathlib import Path
import os
import yaml
import torch
import numpy as np
import pandas as pd
import utils
import experiments

def load_official_mapping(dataset_root):
    """
    Locates and loads the official OGB class mapping file.
    """
    possible_paths = [
        dataset_root / "mapping" / "labelidx2arxivcategeory.csv.gz",
        dataset_root / "ogbn_arxiv" / "mapping" / "labelidx2arxivcategeory.csv.gz",
        dataset_root.parent / "mapping" / "labelidx2arxivcategeory.csv.gz" 
    ]
    
    mapping_file = None
    for p in possible_paths:
        if p.exists():
            mapping_file = p
            break
            
    if mapping_file:
        print(f"✅ Found Official Class Mapping: {mapping_file}")
        df = pd.read_csv(mapping_file)
        return dict(zip(df['label idx'], df['arxiv category']))
    else:
        print("⚠️ Could not find 'labelidx2arxivcategeory.csv.gz'. Using raw IDs.")
        return {}

def analyze_classes():
    # 1. SETUP
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🕵️‍♂️ Running Deep Error Analysis on: {DEVICE}")

    project_root = utils.get_project_root()
    os.chdir(project_root)
    
    with open(str(project_root / "config/experiments_config.yaml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    # Path logic
    raw_data_path = Path(params["data"]["graph_dataset"][params["dataset"]])
    data = torch.load(str(raw_data_path), weights_only=False)
    
    # --- LOAD OFFICIAL MAPPING ---
    dataset_folder = raw_data_path.parent.parent
    arxiv_mapping = load_official_mapping(dataset_folder)
    
    # --- EDGE FILTERING ---
    print("✂️ Filtering graph edges to match training config...")
    selection = utils.edge_type_selection(params["edge_type_selection"][params["dataset"]])
    data = data.edge_type_subgraph(selection)

    # --- RECONSTRUCT REVERSE EDGES ---
    citation_edge_type = None
    for et in list(data.adj_t_dict.keys()):
        if 'cites' in et[1] or 'ref' in et[1]:
            citation_edge_type = et
            break   
    if citation_edge_type:
        adj_t_forward = data.adj_t_dict[citation_edge_type]
        rev_edge_type = (citation_edge_type[0], 'cited_by', citation_edge_type[2])
        data.adj_t_dict[rev_edge_type] = adj_t_forward.t()
        data[rev_edge_type].edge_index = torch.empty((2,0), dtype=torch.long)

    # --- 🛑 SOTA BOSS LEVEL INJECTION (Matches Training Script) ---
    print("\n--- 📥 Building the Ultimate Boss-Level Embeddings for Analysis ---")
    upgraded_path = str(project_root / "data" / "ogbn_arxiv" / "processed" / "geometric_data_processed.pt")
    upgraded_data_list = torch.load(upgraded_path, weights_only=False)
    upgraded_data = upgraded_data_list[0] if isinstance(upgraded_data_list, tuple) else upgraded_data_list
    
    mixed_features = upgraded_data['paper'].x if (hasattr(upgraded_data, 'x_dict') and 'paper' in upgraded_data.x_dict) else upgraded_data.x
    tfidf_features = mixed_features[:, -500:] 
    
    emb_name = params.get("node_embs", "simtg") 
    path_embs = str(project_root / params["data"][f"{emb_name}_embs"][params["dataset"]])
    simtg_features = torch.load(path_embs).type(torch.float32)
    
    ultimate_features = torch.cat([simtg_features, tfidf_features], dim=1)
    data['paper'].x = ultimate_features
    print(f"✅ Reconstructed Features Shape: {data['paper'].x.shape}")

    data.to(DEVICE)

    # --- LOAD MODEL ---
    model_path = "models/ensemble_run_0.pth"
    print(f"\n📂 Loading weights from: {model_path}")
    model = experiments.get_model(data, params, DEVICE) 
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # --- GET PREDICTIONS ---
    test_idx = data["paper"].test_idx
    with torch.no_grad():
        out = model(data.x_dict, data.adj_t_dict)["paper"]
        y_pred = out[test_idx].argmax(dim=-1).cpu()
        y_true = data["paper"].y[test_idx].squeeze().cpu()

    # --- COMPUTE CONFUSION ANALYSIS ---
    print("\n" + "="*100)
    print(f"{'ID':<3} | {'Class Name':<25} | {'Count':<6} | {'Acc':<6} | {'Major Misclassification'}")
    print("="*100)

    class_stats = []
    unique_classes = torch.unique(y_true).numpy()
    unique_classes.sort()

    for cls_id in unique_classes:
        mask = (y_true == cls_id)
        count = mask.sum().item()
        
        # Get predictions for this specific class
        class_preds = y_pred[mask]
        correct = (class_preds == cls_id).sum().item()
        acc = (correct / count) * 100 if count > 0 else 0.0
        
        # FIND THE "PRIMARY CONFUSION"
        pred_counts = torch.bincount(class_preds, minlength=40)
        
        pred_counts_no_correct = pred_counts.clone()
        pred_counts_no_correct[cls_id] = -1
        
        most_confused_id = pred_counts_no_correct.argmax().item()
        most_confused_count = pred_counts_no_correct[most_confused_id].item()
        confused_pct = (most_confused_count / count) * 100 if count > 0 else 0.0
        
        # Resolve Names
        cls_name = arxiv_mapping.get(cls_id, f"Class {cls_id}")
        confused_name = arxiv_mapping.get(most_confused_id, f"Class {most_confused_id}")
        
        class_stats.append({
            "id": cls_id,
            "name": cls_name,
            "count": count,
            "acc": acc,
            "confused_id": most_confused_id,
            "confused_name": confused_name,
            "confused_pct": confused_pct
        })

    df = pd.DataFrame(class_stats)
    df_sorted = df.sort_values(by="acc", ascending=True)
    
    for _, row in df_sorted.iterrows():
        status = "🚨" if row['acc'] < 50 else ""
        misclass_str = f"-> {row['confused_name']} ({row['confused_pct']:.1f}%)"
        print(f"{int(row['id']):<3} | {row['name']:<25} | {int(row['count']):<6} | {row['acc']:.1f}% {status} | {misclass_str}")

    print("="*100)
    
    # --- MACRO VS MICRO ACCURACY CALCULATION ---
    micro_acc = (df['count'] * df['acc'] / 100).sum() / df['count'].sum() * 100
    macro_acc = df['acc'].mean()
    
    print(f"\n📊 SYSTEM ACCURACY METRICS")
    print(f"   Micro Accuracy (Global): {micro_acc:.2f}% (Biased toward big classes)")
    print(f"   Macro Accuracy (Class-Averaged): {macro_acc:.2f}% (True measure of class fairness)")
    print("="*100)
    
    # HYPOTHESIS CHECK
    worst_class = df_sorted.iloc[0]
    print(f"\n🔍 DIAGNOSIS: Worst Class '{worst_class['name']}'")
    print(f"   It is primarily confused with: '{worst_class['confused_name']}'")
    print(f"   If these two fields are unrelated (e.g. Networking -> NLP), your embeddings are misaligned.")

if __name__ == "__main__":
    analyze_classes()