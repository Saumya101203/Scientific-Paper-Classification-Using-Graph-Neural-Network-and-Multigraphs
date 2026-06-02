from pathlib import Path
import os
import yaml
import csv
import dbm
import collections
import numpy as np
import pandas as pd
import random
from itertools import combinations
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm
import torch.serialization
import torch_geometric
import utils
import torch.serialization
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])



def preprocess(df, col, id, attr=None):
    temp_df = df.copy().dropna(subset=[col]).reset_index()[[id, col]]

    if attr is not None:
        if col == "venue":
            temp_df[col] = pd.Series([x[attr] if attr in x.keys() else None for x in temp_df[col]], dtype=object)
        else:
            temp_df[col] = [[x[attr] for x in xs] for xs in temp_df[col]]

    grouped = temp_df.drop_duplicates(subset=[id]).explode(col).groupby(col)[id].apply(lambda x: x.tolist())
    return grouped


def generate_edges(data, kind, seed, threshold=None, k_per_node=5):
    """
    Build edges for a relation by connecting each paper to up to k others
    inside the same group. This avoids O(n^2) blow-ups.

    Args:
        data (pd.Series): group -> list[papers]
        kind (str): relation name
        seed (int): rng seed
        threshold (int|None): if not None and len(group) > threshold, we still
            keep the whole group but we limit degree via k_per_node.
        k_per_node (int): number of neighbors per node within a group.

    Returns:
        dict: {kind: list[(u,v)]} with u < v (deduped)
    """
    rng = random.Random(seed)
    edge_set = set()

    for ids in tqdm(data, desc=f"Generating '{kind}' Edges (k={k_per_node})"):
        if not ids or len(ids) < 2:
            continue

        # De-dup within the group to avoid wasting degree on repeats:
        ids = list(dict.fromkeys(ids))
        n = len(ids)

        # For small groups, connect everyone to up to k others.
        # For large groups (n >> k), this stays O(n * k).
        for i, u in enumerate(ids):
            # choose neighbors from the *rest* of the group
            # do NOT include u itself
            if n <= 1:
                continue
            # Select up to k distinct neighbors uniformly at random
            # We random sample from indices except i
            pool = ids[:i] + ids[i+1:]
            m = min(k_per_node, len(pool))
            if m == 0:
                continue
            nbrs = rng.sample(pool, m)
            for v in nbrs:
                a, b = (u, v) if u < v else (v, u)
                edge_set.add((a, b))

    # Turn set into list of pairs
    edge_list = list(edge_set)
    return {kind: edge_list}


def generate_fos_edges_min_two(df, generic_fields_set, max_group_size=2000, desc="Generating 'fos_min_two' Edges"):
    """
    Generates an edge list for papers that share at least two
    non-generic fields of study.
    
    This version uses pandas-based batch processing to count 
    pairs efficiently without running out of RAM or segfaulting.
    
    Args:
        df (pd.DataFrame): The main metadata dataframe.
        generic_fields_set (set): A set of generic FOS names to ignore.
        max_group_size (int): The maximum size of a FOS group to process.
    
    Returns:
        dict: {"fos": list_of_edges}
    """
    print("Starting FOS edge generation (min 2 shared, pandas-batch)...")
    
    # 1. Create a Series of [paper_index] -> {set of fos_names}
    print("  Step 1: Creating cleaned FOS sets for each paper...")
    temp_df = df.copy().dropna(subset=['fos']).reset_index()[['index', 'fos']]
    
    def get_cleaned_fos_set(fos_list):
        if not isinstance(fos_list, (list, np.ndarray)):
            return set()
        return {
            x['name'] for x in fos_list 
            if 'name' in x and x['name'] is not None 
            and x['name'] not in generic_fields_set
        }
        
    paper_to_fos_sets = temp_df.set_index('index')['fos'].apply(get_cleaned_fos_set)
    paper_to_fos_sets = paper_to_fos_sets[paper_to_fos_sets.apply(len) >= 2]
    
    # 2. Invert the index: [fos_name] -> [list of paper_indices]
    print("  Step 2: Inverting index (FOS -> [papers])...")
    
    # --- THIS IS THE USER'S CORRECTED LOGIC ---
    # Explode() puts FOS names in the values, reset_index() saves paper IDs
    exploded_df = paper_to_fos_sets.explode().reset_index()
    # Group by the FOS name (now in column 'fos'), select the paper IDs (in column 'index')
    fos_to_papers = exploded_df.groupby('fos')['index'].apply(list)
    # --- END CORRECTION ---
    
    # 3. Filter out massive groups
    print(f"  Step 3: Filtering FOS groups larger than {max_group_size} papers...")
    original_count = len(fos_to_papers)
    fos_to_papers = fos_to_papers[fos_to_papers.apply(len) <= max_group_size]
    print(f"  Filtered {original_count - len(fos_to_papers)} FOS groups.")

    # 4. Count co-occurrences using pandas in batches
    print("  Step 4: Counting paper co-occurrences (pandas-batch)...")
    
    BATCH_SIZE = 10_000_000  # Process 10 million pairs at a time
    batch_pairs = []
    main_counter = collections.Counter()

    for paper_list in tqdm(fos_to_papers, desc=desc):
        if len(paper_list) < 2:
            continue
            
        pairs = combinations(sorted(paper_list), 2)
        batch_pairs.extend(pairs)
        
        # When batch is full, process it
        if len(batch_pairs) > BATCH_SIZE:
            print(f"\n  Processing batch of {len(batch_pairs)} pairs...")
            df_batch = pd.DataFrame(batch_pairs, columns=['u', 'v'])
            batch_counts = df_batch.groupby(['u', 'v']).size()
            main_counter.update(batch_counts.to_dict())
            batch_pairs = [] # Clear batch

    # Process the final leftover batch
    if batch_pairs:
        print(f"\n  Processing final batch of {len(batch_pairs)} pairs...")
        df_batch = pd.DataFrame(batch_pairs, columns=['u', 'v'])
        batch_counts = df_batch.groupby(['u', 'v']).size()
        main_counter.update(batch_counts.to_dict())

    # 5. Filter for pairs that appeared >= 2 times
    print("  Step 5: Filtering for edges with >= 2 shared FOS...")
    final_edge_list = [
        edge for edge, count in main_counter.items() if count >= 2
    ]
    
    print(f"  Finished: Found {len(final_edge_list)} edges for papers sharing >= 2 FOS.")
    return {"fos": final_edge_list}

def multigraph_dataset(df, edge_lists, device, data=None, node_map=None):
    """
    Instantiates or updates a HeteroData object.
    
    This robust version handles empty edge lists by creating
    a 2x0 empty tensor, preventing crashes.
    """
    
    # --- ROBUST TENSOR CREATION ---
    processed_edge_lists = {}
    for d in edge_lists:
        for k, v in d.items():
            # (KEY CHANGE) We now explicitly check that v is a non-empty list
            # This prevents the 'str' error if v is somehow malformed.
            if isinstance(v, list) and v:
                # Original logic for non-empty lists
                processed_edge_lists[k] = torch.tensor(v, dtype=torch.long).T.contiguous()
            else:
                # Create a 2x0 empty tensor if list is empty or v is not a list
                processed_edge_lists[k] = torch.empty((2, 0), dtype=torch.long)
    # --- END CORRECTION ---

    if data is not None: # Data object already exists.
        data[("paper", "shares_author_with", "paper")].edge_index = processed_edge_lists["authorship"]
        data[("paper", "shares_fos_with", "paper")].edge_index = processed_edge_lists["fos"]
        data[("paper", "shares_venue_with", "paper")].edge_index = processed_edge_lists["venue"]
    else: # instantiate new Data object for PubMed.
        assert node_map is not None, "Must specify `node_map` for PubMed."
        data = HeteroData(
            paper={
                "y" : torch.tensor(df["label"].values, dtype=torch.long, device=device).unsqueeze(0).T, 
                "num_nodes" : len(node_map)
            },
            paper__references__paper={"edge_index" : processed_edge_lists["references"]},
            paper__shares_author_with__paper={"edge_index" : processed_edge_lists["authorship"]},
            paper__shares_journal_with__paper={"edge_index" : processed_edge_lists["srcid"]},
            paper__shares_mesh_with__paper={"edge_index" : processed_edge_lists["mesh"]}
        ).to(device)

    data = T.ToUndirected()(data)
    return data
    
def main_ogbnarxiv():
    dataset = PygNodePropPredDataset("ogbn-arxiv", root="data/")
    data = dataset[0]
    splits = dataset.get_idx_split()
    data = data.to_heterogeneous(
        node_type_names=["paper"],
        edge_type_names=[("paper", "references", "paper")]
    )
    data["paper"].train_idx = splits["train"]
    data["paper"].val_idx = splits["valid"]
    data["paper"].test_idx = splits["test"]

    # ========= build edges from MAG table =========

    # ---- AUTHORS FILTER ----
    # (Filter: 2-20 papers, k=3)
    authorship_input = preprocess(df, "authors", "index", "id")
    authorship_input = authorship_input[authorship_input.apply(lambda x: 2 <= len(x) <= 20)]
    authorship_output = generate_edges(authorship_input, "authorship", params["seed"], k_per_node=3)

    
    # ---- FOS FILTER (Cleaned Unigram Logic) ----
    print("Starting FOS edge generation (Cleaned Unigram)...")

    # 1. Get lists of FOS names for each paper
    #    (This handles the np.ndarray type from the parquet file)
    temp_df = df.copy().dropna(subset=['fos']).reset_index()[['index', 'fos']]
    temp_df['fos_names'] = temp_df['fos'].apply(
        lambda xs: [x['name'] for x in xs if 'name' in x and x['name'] is not None] if isinstance(xs, (list, np.ndarray)) else []
    )

    # 2. Explode to get (paper_id, fos_name) pairs and group by fos_name
    fos_input_groups = temp_df.drop_duplicates(subset=['index']).explode('fos_names')
    fos_input = fos_input_groups.groupby('fos_names')['index'].apply(lambda x: x.tolist())
    
    # 3. (KEY) Filter groups by size (3 <= len <= 2000)
    #    This removes both "generic" (too large) and "noisy" (too small) groups.
    print(f"  Original FOS groups: {len(fos_input)}")
    fos_input = fos_input[fos_input.apply(
        lambda x: 3 <= len(x) <= 2000
    )]
    print(f"  Filtered down to {len(fos_input)} FOS unigram groups (size 3-2000).")

    # 4. Generate edges with k=3
    fos_output = generate_edges(fos_input, "fos", params["seed"], k_per_node=5)


    # ---- VENUE CLEAN (Data Leak Fix) ----
    # (Filter: doc_type in {'Conference', 'Journal'}, k=2)
    print("Filtering venues to 'Conference' and 'Journal' to prevent data leak...")
    venue_df = df[df['doc_type'].isin(['Conference', 'Journal'])]
    print(f"Using {len(venue_df)} of {len(df)} papers for venue edge generation.")
    
    venue_input = preprocess(venue_df, "venue", "index", "id")
    venue_input = venue_input[venue_input.index.notna()]  # remove None
    venue_output = generate_edges(venue_input, "venue", params["seed"], k_per_node=2)

    
    # --- Combine Edge Lists ---
    edge_lists = [authorship_output, fos_output, venue_output]

    # ========= transform =========
    # (Using the robust multigraph_dataset function)
    data = multigraph_dataset(df, edge_lists, torch.device("cpu"), data)

    # ========= attach attributes AFTER transform =========
    def normalize_fos(x):
        arr = []
        if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
            for v in x:
                if isinstance(v, dict) and ('name' in v):
                    arr.append(v['name'])
                elif isinstance(v, str):
                    arr.append(v)
        return arr

    def parse_venue(v):
        if isinstance(v, dict):
            return v.get("id")
        return None

    data["paper"].fos = df["fos"].apply(normalize_fos).tolist()
    data["paper"].venue = df["venue"].apply(parse_venue).tolist()

    # ========= save =========
    # (Setting a new version name for this "sweet spot" graph)
    save_path = Path(params["data"]["save_path"]["ogbnarxiv"]).with_name("ogbnarxiv_heterodata_v8_cleaned_unigram.pt")
    torch.save(data, save_path)
    print("SAVED:", save_path)

def main_pubmed():
    records = []
    with open(params["data"]["path_to_metadata"]["pubmed_refs"]) as tsv: # Citation edge list stored in separate TSV. 
        for line in csv.reader(tsv, delimiter="\t"):
            records.append(line)
    refs_edgelist = [(int(r[1].split(":")[1]), int(r[-1].split(":")[1])) for r in records[2:]]
    refs_edgelist = [(e1, e2) for (e1, e2) in refs_edgelist if e1 != 17874530 and e2 != 17874530] # 17874530 is the one PMID with no available metadata; manually remove.
    refs_output = {"references" : refs_edgelist}

    authorship_input = preprocess(df, "authors", "pmid")
    authorship_output = generate_edges(authorship_input, "authorship", params["seed"])

    srcid_input = preprocess(df, "journal", "pmid")
    srcid_output = generate_edges(srcid_input, "srcid", params["seed"])

    mesh_input = preprocess(df, "mesh", "pmid")
    mesh_output = generate_edges(mesh_input, "mesh", params["seed"], int(np.mean(mesh_input.apply(len))))

    def remapper(internal_eids, edge_lists):
        node_map = dict(zip(range(len(internal_eids)), internal_eids))
        inv_node_map = {v : k for k, v in node_map.items()}
        edge_lists_remapped = [{k : [(inv_node_map[eid_1], inv_node_map[eid_2]) 
            for (eid_1, eid_2) in tqdm(v)]} for edge_list in edge_lists for k, v in edge_list.items()]
        
        return node_map, edge_lists_remapped

    node_map, edge_lists = remapper(df["pmid"].to_list(), [refs_output, authorship_output, mesh_output, srcid_output])

    data = multigraph_dataset(df, edge_lists, torch.device("cpu"), node_map=node_map)
    data["paper"].x = torch.tensor(np.stack(df["tfidf"])).float() # attach the default TFIDF features.

    if params["pubmed_fixed_split"]:
        splits = utils.per_class_idx_split(data, params["seed"])
        data["paper"].train_idx, data["paper"].val_idx, data["paper"].test_idx = splits

    save_path = str(Path(params["data"]["save_path"]["pubmed"]))
    torch.save(data, save_path)

if __name__ == "__main__":
    project_root: Path = utils.get_project_root()
    os.chdir(project_root)
    with open(str(project_root / "config/data_generation_config.yaml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    path_to_metadata = str(Path(params["data"]["path_to_metadata"][params["dataset"]]))
    df = pd.read_parquet(path_to_metadata)

    if params["dataset"] == "ogbnarxiv":
        main_ogbnarxiv()
    elif params["dataset"] == "pubmed":
        main_pubmed()