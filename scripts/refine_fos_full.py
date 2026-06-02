# scripts/refine_fos_full.py
import os
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import torch
from tqdm import tqdm
import torch.serialization
from torch_geometric.data.hetero_data import HeteroData, NodeStorage, EdgeStorage, BaseStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
import pandas as pd
import torch_geometric.utils # Import for coalesce

torch.serialization.add_safe_globals([
    HeteroData, NodeStorage, EdgeStorage, BaseStorage,
    DataEdgeAttr, DataTensorAttr, pd.Series
])

# This set contains all 40 target labels from OGBN-ArXiv, in lowercase.
LABEL_EXCLUSION_SET_LOWER = {
    'cs.ai', 'cs.ar', 'cs.cc', 'cs.ce', 'cs.cg', 'cs.cl', 'cs.cr', 'cs.cv',
    'cs.cy', 'cs.db', 'cs.dc', 'cs.dl', 'cs.dm', 'cs.ds', 'cs.et', 'cs.fl',
    'cs.gl', 'cs.gr', 'cs.gt', 'cs.hc', 'cs.ir', 'cs.it', 'cs.lg', 'cs.lo',
    'cs.ma', 'cs.mm', 'cs.ms', 'cs.na', 'cs.ne', 'cs.ni', 'cs.oh', 'cs.os',
    'cs.pf', 'cs.pl', 'cs.ro', 'cs.sc', 'cs.sd', 'cs.se', 'cs.si', 'cs.sy'
}

# Skip any keyword shared by more than this many papers.
GENERIC_KEYWORD_THRESHOLD = 5000 

# --- ADDED THIS ---
# Keep only the top K strongest neighbors for each paper
TOP_K_NEIGHBORS = 20
# --- END ADD ---


FULL_PATH = Path("data/ogbnarxiv_heterodata_v2.pt")
OUT_PATH  = Path("data/ogbnarxiv_heterodata_v2_refined.pt")

print(f"Loading full graph from: {FULL_PATH}")
data = torch.load(FULL_PATH, weights_only=False)
num_nodes = int(data["paper"].num_nodes)
fos_lists = data["paper"].fos    # already list[list[str]]

# --- ALGORITHM PART 1 (Unchanged) ---
print("Building efficient lookup tables...")
paper_to_fos = defaultdict(set)
fos_to_papers = defaultdict(list)
num_filtered_labels = 0
num_filtered_generic = 0

# 1. Build paper_to_fos map
for pid in tqdm(range(num_nodes), desc="Pass 1: Building paper_to_fos"):
    if fos_lists[pid] is None:
        continue
    for fos in fos_lists[pid]:
        if not isinstance(fos, str):
            continue
        cleaned_fos = fos.strip().lower() 
        if cleaned_fos in LABEL_EXCLUSION_SET_LOWER:
            num_filtered_labels += 1
            continue
        paper_to_fos[pid].add(cleaned_fos)

# 2. Build fos_to_papers map
for pid, fos_set in tqdm(paper_to_fos.items(), desc="Pass 2: Building fos_to_papers"):
    for fos in fos_set:
        fos_to_papers[fos].append(pid)

# 3. Clean fos_to_papers by removing generic keywords
generic_keys = set()
for fos, plist in fos_to_papers.items():
    if len(plist) > GENERIC_KEYWORD_THRESHOLD:
        generic_keys.add(fos)
        num_filtered_generic += 1
for fos_key in generic_keys:
    del fos_to_papers[fos_key]
for pid in tqdm(range(num_nodes), desc="Pass 3: Cleaning paper_to_fos"):
    paper_to_fos[pid].difference_update(generic_keys)

print(f"Filtered out {num_filtered_labels} label occurrences (e.g., 'cs.lg').")
print(f"Filtered out {num_filtered_generic} generic keywords (e.g., 'machine learning').")
print(f"Unique FoS (after filtering) = {len(fos_to_papers)}")
# --- END ALGORITHM PART 1 ---


# --- MODIFIED ALGORITHM: PART 2 (WITH TOP_K) ---
print(f"Finding paper-paper pairs (Top {TOP_K_NEIGHBORS} neighbors, >=2 shared FoS)...")

all_edge_tensors = []  
current_chunk_pairs = [] 
CHUNK_SIZE = 2_000_000   
total_pairs_found = 0

for pid in tqdm(range(num_nodes), desc="Finding shared pairs"):
    keywords = paper_to_fos[pid]
    if not keywords:
        continue

    neighbor_counts = defaultdict(int)
    for fos in keywords:
        plist = fos_to_papers.get(fos) 
        if plist is None:
            continue
            
        for neighbor_pid in plist:
            if neighbor_pid > pid: 
                neighbor_counts[neighbor_pid] += 1
    
    # --- THIS IS THE FIX ---
    # Sort neighbors by count (strongest connection first) and take top K
    sorted_neighbors = sorted(neighbor_counts.items(), key=lambda item: item[1], reverse=True)
    
    for neighbor_pid, count in sorted_neighbors[:TOP_K_NEIGHBORS]:
        # Still enforce the '>= 2' rule for quality
        if count >= 2:
            current_chunk_pairs.append((pid, neighbor_pid))
            total_pairs_found += 1
    # --- END FIX ---
    
    # If chunk is full, convert it to a tensor and clear the list
    if len(current_chunk_pairs) >= CHUNK_SIZE:
        chunk_tensor = torch.tensor(current_chunk_pairs, dtype=torch.long).t().contiguous()
        all_edge_tensors.append(chunk_tensor)
        current_chunk_pairs = [] # Reset the chunk list

# Add any remaining pairs from the last chunk
if current_chunk_pairs:
    chunk_tensor = torch.tensor(current_chunk_pairs, dtype=torch.long).t().contiguous()
    all_edge_tensors.append(chunk_tensor)

print(f"Found {total_pairs_found} unique pairs with >= 2 shared FoS (Top {TOP_K_NEIGHBORS}).")

# Now, create the final tensors from the chunks
if all_edge_tensors:
    pairs_tensor = torch.cat(all_edge_tensors, dim=1)
    edge_index = torch.cat([pairs_tensor, pairs_tensor.flip(0)], dim=1)
    edge_index, _ = torch_geometric.utils.coalesce(edge_index, num_nodes=num_nodes)
else:
    edge_index = torch.empty((2, 0), dtype=torch.long)
# --- END MODIFIED ALGORITHM: PART 2 ---


# --- ADDED THIS FIX ---
# Your previous log showed a 1D shape. This block forces it to be 2D.
if edge_index.dim() == 1:
    num_edges = edge_index.numel() // 2
    print(f"Warning: edge_index was 1D. Reshaping {edge_index.shape} -> torch.Size([2, {num_edges}])")
    edge_index = edge_index.reshape(2, num_edges)
# --- END ADDED FIX ---


print("Final 'shares_fos_with' edges:", edge_index.shape)

data[("paper","shares_fos_with","paper")].edge_index = edge_index

torch.save(data, OUT_PATH)
print(f"REFINED graph saved to: {OUT_PATH}")