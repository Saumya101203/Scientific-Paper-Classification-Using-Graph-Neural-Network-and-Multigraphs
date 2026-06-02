import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import os
import utils

def upgrade_embeddings():
    project_root = utils.get_project_root()
    os.chdir(project_root)

    print("==================================================")
    print("🚀 FUSING TF-IDF WITH SIMTEG EMBEDDINGS")
    print("==================================================")

    print("1. Loading Raw Text Data...")
    parquet_path = project_root / "data" / "tables" / "ogbnarxiv_mag_metadata.parquet.gzip"
    df = pd.read_parquet(parquet_path)
    
    # Combine title and abstract into a single text block
    df['text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')

    print("2. Loading Original PyG Graph...")
    # UPDATED PATH: Pointing directly to your geometric_data_processed.pt
    data_path = project_root / "data" / "ogbn_arxiv" / "processed" / "geometric_data_processed.pt"
    
    # We use torch.load but we index [0] because PyG saves lists of data objects in these files
    data_list = torch.load(data_path, weights_only=False)
    
    # Handle the fact that geometric_data_processed.pt is usually a tuple: (data, slices)
    if isinstance(data_list, tuple):
        data = data_list[0]
    else:
        data = data_list

    print("3. Calculating TF-IDF Vectors...")
    # We take the top 500 most mathematically distinct words across the whole dataset
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    
    # Fit and transform the text
    tfidf_sparse = vectorizer.fit_transform(df['text'])

    print("4. Converting to PyTorch Tensors...")
    # Convert the SciPy sparse matrix to a dense PyTorch tensor
    tfidf_tensor = torch.tensor(tfidf_sparse.toarray(), dtype=torch.float32)

    # Note: Depending on your exact PyG object structure, the node features 
    # might be under data.x or data['paper'].x. We will handle both safely.
    if hasattr(data, 'x_dict') and 'paper' in data.x_dict:
        original_x = data['paper'].x
        is_hetero = True
    else:
        original_x = data.x
        is_hetero = False

    print(f"   -> Original features shape: {original_x.shape}")
    print(f"   -> TF-IDF explicit features shape: {tfidf_tensor.shape}")

    print("5. Concatenating Features...")
    # Stitch them together! 
    new_x = torch.cat([original_x, tfidf_tensor], dim=1)
    
    if is_hetero:
        data['paper'].x = new_x
    else:
        data.x = new_x
    
    print(f"   -> 🔥 NEW Upgraded features shape: {new_x.shape}")

    print("6. Saving Upgraded Dataset...")
    # Save it as a NEW file right next to the original
    new_data_path = project_root / "data" / "ogbn_arxiv" / "processed" / "geometric_data_augmented.pt"
    
    # Save it back in the exact same format PyG expects
    if isinstance(data_list, tuple):
        torch.save((data, data_list[1]), new_data_path)
    else:
        torch.save(data, new_data_path)
        
    print(f"✅ Success! Upgraded graph saved to: {new_data_path}")
    print("==================================================")

if __name__ == "__main__":
    upgrade_embeddings()