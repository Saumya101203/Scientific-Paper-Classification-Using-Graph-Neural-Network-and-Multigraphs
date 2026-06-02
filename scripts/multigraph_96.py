from pathlib import Path
import os
import yaml
import csv

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


def multigraph_dataset(df, edge_lists, device, data=None, node_map=None):
    edge_lists = {k : torch.tensor(v, dtype=torch.long).T.contiguous() for d in edge_lists for k, v in d.items()}

    if data is not None: # Data object already exists.
        data[("paper", "shares_author_with", "paper")].edge_index = edge_lists["authorship"]
        data[("paper", "shares_fos_with", "paper")].edge_index = edge_lists["fos"]
        data[("paper", "shares_venue_with", "paper")].edge_index = edge_lists["venue"]
    else: # instantiate new Data object for PubMed.
        assert node_map is not None, "Must specify `node_map` for PubMed."
        data = HeteroData(
            paper={
                "y" : torch.tensor(df["label"].values, dtype=torch.long, device=device).unsqueeze(0).T, 
                "num_nodes" : len(node_map)
            },
            paper__references__paper={"edge_index" : edge_lists["references"]},
            paper__shares_author_with__paper={"edge_index" : edge_lists["authorship"]},
            paper__shares_journal_with__paper={"edge_index" : edge_lists["srcid"]},
            paper__shares_mesh_with__paper={"edge_index" : edge_lists["mesh"]}
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
    authorship_input = preprocess(df, "authors", "index", "id")
    authorship_output = generate_edges(authorship_input, "authorship", params["seed"], threshold=200, k_per_node=5)

    fos_input = preprocess(df, "fos", "index", "name")
    fos_output = generate_edges(fos_input, "fos", params["seed"], threshold=50, k_per_node=3)

    venue_input = preprocess(df, "venue", "index", "id")
    venue_output = generate_edges(venue_input,"venue",params["seed"],threshold=int(np.mean(venue_input.apply(len))),  k_per_node=2)


    edge_lists = [authorship_output, fos_output, venue_output]

    # ========= transform ONCE (this will do ToUndirected + ToSparseTensor) =========
    data = multigraph_dataset(df, edge_lists, torch.device("cpu"), data)


    # ========= attach attributes AFTER transform =========

    def normalize_fos(x):
        arr=[]
        if isinstance(x,(list,tuple,np.ndarray,pd.Series)):
            for v in x:
                if isinstance(v,dict) and ('name' in v):
                    arr.append(v['name'])
                elif isinstance(v,str):
                    arr.append(v)
        return arr

    def parse_venue(v):
        if isinstance(v,dict):
            return v.get("id")
        return None

    data["paper"].fos   = df["fos"].apply(normalize_fos).tolist()
    data["paper"].venue = df["venue"].apply(parse_venue).tolist()

    # ========= save =========
    save_path = Path(params["data"]["save_path"]["ogbnarxiv"]).with_name("ogbnarxiv_heterodata_v2.pt")
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