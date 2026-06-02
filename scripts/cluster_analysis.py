from pathlib import Path
import os
import yaml
import torch
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities
import utils

def load_official_mapping(dataset_root):
    possible_paths = [
        dataset_root / "mapping" / "labelidx2arxivcategeory.csv.gz",
        dataset_root / "ogbn_arxiv" / "mapping" / "labelidx2arxivcategeory.csv.gz",
        dataset_root.parent / "mapping" / "labelidx2arxivcategeory.csv.gz" 
    ]
    for p in possible_paths:
        if p.exists():
            df = pd.read_csv(p)
            return dict(zip(df['label idx'], df['arxiv category']))
    return {}

def run_cluster_analysis():
    print("🕵️‍♂️ Running Graph Clustering & Heterophily Analysis...")
    project_root = utils.get_project_root()
    os.chdir(project_root)
    
    with open(str(project_root / "config/experiments_config.yaml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    # 1. Load the original raw graph
    raw_data_path = Path(params["data"]["graph_dataset"][params["dataset"]])
    data_list = torch.load(str(raw_data_path), weights_only=False)
    data = data_list[0] if isinstance(data_list, tuple) else data_list
    
    dataset_folder = raw_data_path.parent.parent
    arxiv_mapping = load_official_mapping(dataset_folder)

    # Use standard citation edges
    citation_edge_type = None
    for et in list(data.adj_t_dict.keys()):
        if 'cites' in et[1] or 'ref' in et[1]:
            citation_edge_type = et
            break
            
    if not citation_edge_type:
        print("❌ Could not find citation edges!")
        return

    adj_t = data.adj_t_dict[citation_edge_type]
    row, col, _ = adj_t.coo()
    y = data['paper'].y.squeeze().numpy()
    
    # Let's analyze the two worst offending classes
    target_classes = [11, 29]
    
    for target_class in target_classes:
        target_name = arxiv_mapping.get(target_class, f"Class {target_class}")
        print("\n" + "="*80)
        print(f"🔍 ANALYZING CLASS {target_class}: {target_name}")
        print("="*80)
        
        # --- PART 1: EDGE LEAKAGE (HETEROPHILY) ---
        # Find all nodes belonging to the target class
        target_nodes = torch.where(torch.tensor(y) == target_class)[0]
        
        # Find all edges originating FROM this class
        mask = torch.isin(row, target_nodes)
        source_nodes = row[mask]
        dest_nodes = col[mask]
        
        # Look at the labels of the nodes they cite
        dest_labels = y[dest_nodes.numpy()]
        
        print("\n📊 1. EDGE LEAKAGE (Where do their citations go?)")
        unique_dests, counts = np.unique(dest_labels, return_counts=True)
        leakage_stats = list(zip(unique_dests, counts))
        leakage_stats.sort(key=lambda x: x[1], reverse=True)
        
        total_edges = len(dest_labels)
        for dest_cls, count in leakage_stats[:5]:
            dest_name = arxiv_mapping.get(dest_cls, f"Class {dest_cls}")
            pct = (count / total_edges) * 100 if total_edges > 0 else 0
            flag = "✅ (Homophily)" if dest_cls == target_class else "🚨 (Heterophily Leak)"
            print(f"   -> Cites {dest_name:<25}: {count:<5} edges ({pct:.1f}%) {flag}")
            
        # --- PART 2: GRAPH CLUSTERING (LOUVAIN) ---
        print("\n🧬 2. SUB-COMMUNITY DETECTION (Louvain Clustering)")
        print("   Extracting local subgraph and running Louvain algorithm...")
        
        # We build a subgraph using ONLY the target class nodes to see if it's fragmented
        subgraph_edges_mask = torch.isin(row, target_nodes) & torch.isin(col, target_nodes)
        sub_row = row[subgraph_edges_mask].numpy()
        sub_col = col[subgraph_edges_mask].numpy()
        
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(target_nodes.numpy())
        edges = list(zip(sub_row, sub_col))
        G.add_edges_from(edges)
        
        # Find isolated nodes (papers that cite nothing else within their own class)
        isolated = list(nx.isolates(G))
        G.remove_nodes_from(isolated)
        
        print(f"   -> Found {len(isolated)} isolated nodes (no internal citations).")
        print(f"   -> Clustering the remaining {G.number_of_nodes()} highly connected nodes...")
        
        if G.number_of_nodes() > 0:
            # Run Louvain Community Detection
            communities = louvain_communities(G)
            communities = sorted(communities, key=len, reverse=True)
            
            print(f"   ✅ Discovered {len(communities)} distinct structural sub-communities!")
            for i, comm in enumerate(communities[:5]):
                print(f"      * Cluster {i+1}: {len(comm)} nodes")
            if len(communities) > 5:
                print(f"      * ... and {len(communities) - 5} smaller clusters.")
        else:
            print("   ⚠️ Not enough internal edges to form communities. This class is entirely scattered!")

if __name__ == "__main__":
    run_cluster_analysis()