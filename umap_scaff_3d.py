import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler
import umap

# 1. Load dataset
DATA_PATH = "/Users/victoriamedina/Thesis_Project/Thesis/Visualizations/chembl.csv"
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows from {DATA_PATH}")

# 2. Filter to valid SMILES and pChEMBL values
df = df.dropna(subset=["Smiles", "pChEMBL Value"]).reset_index(drop=True)
print(f"{len(df):,} rows with valid SMILES and pChEMBL")

# 3. Sample a subset for visualization (adjust n if needed)
df = df.sample(n=40324)

# 4. Convert SMILES to Morgan fingerprints and extract scaffolds
def smiles_to_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

def extract_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

fps = []
valid_indices = []
scaffolds = []

for i, smi in enumerate(df["Smiles"]):
    fp = smiles_to_fp(smi)
    scaffold = extract_scaffold(smi)
    if fp:
        fps.append(np.array(fp))
        valid_indices.append(i)
        scaffolds.append(scaffold)

fps = np.array(fps)
df = df.iloc[valid_indices].copy()
df["Scaffold"] = scaffolds
print(f"Generated fingerprints and scaffolds for {len(df):,} molecules")

# 5. Bucket pChEMBL into potency categories
def bucket_pchembl(p):
    if p >= 8:
        return "High"
    elif p >= 6:
        return "Moderate"
    else:
        return "Low"

df["Potency_Bucket"] = df["pChEMBL Value"].apply(bucket_pchembl)

# 6. Run UMAP in 3D
print("Running UMAP...")
scaled_fps = StandardScaler().fit_transform(fps)
reducer = umap.UMAP(n_components=3)
embedding = reducer.fit_transform(scaled_fps)
print("UMAP completed.")

# 7. Export to JSON
json_data = [
    {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "label": potency
    }
    for (x, y, z), potency in zip(embedding, df["Potency_Bucket"])
]

json_path = Path("/Users/victoriamedina/Thesis_Project/chembl-pfalciparum/umap_embedding.json")
with open(json_path, "w") as f:
    json.dump(json_data, f)
print(f"UMAP 3D embedding exported to {json_path}")

# 8. Optional: Plot only first 2D components for static figure
plt.figure(figsize=(10, 6))
colors = {"High": "#FDE725FF", "Moderate": "#7AD151FF", "Low": "#22A884FF"}

for category, color in colors.items():
    mask = df["Potency_Bucket"] == category
    plt.scatter(embedding[mask, 0], embedding[mask, 1],
                c=color, label=category, alpha=0.6, s=10)
