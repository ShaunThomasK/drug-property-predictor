import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

print("Loading datasets...")
zinc = pd.read_csv("250k_rndm_zinc_drugs_clean_3.csv")
qm9  = pd.read_csv("qm9.csv")

zinc["smiles"] = zinc["smiles"].str.strip()
qm9["smiles"]  = qm9["smiles"].str.strip()

qm9_clean = qm9[["smiles", "mu", "alpha", "homo", "lumo", "gap", "u0"]].copy()
qm9_clean.columns = ["smiles", "dipole", "polarizability", "homo", "lumo", "gap", "total_energy"]

zinc_clean = zinc[["smiles", "logP", "qed"]].copy()

def calc_logp(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return round(Descriptors.MolLogP(mol), 3) if mol else None
    except:
        return None

def calc_mw(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return round(Descriptors.MolWt(mol), 2) if mol else None
    except:
        return None

print("Computing logP for QM9 molecules...")
qm9_clean["logP"] = qm9_clean["smiles"].apply(calc_logp)
qm9_clean["MW"]   = qm9_clean["smiles"].apply(calc_mw)
qm9_clean = qm9_clean.dropna()

print("Computing MW for ZINC molecules...")
zinc_clean["MW"] = zinc_clean["smiles"].apply(calc_mw)
zinc_clean = zinc_clean.dropna()

all_smiles = pd.concat([
    zinc_clean[["smiles", "logP", "MW"]],
    qm9_clean[["smiles",  "logP", "MW"]]
]).drop_duplicates(subset="smiles")

low  = all_smiles[all_smiles["logP"] <  0].sample(min(3000,  len(all_smiles[all_smiles["logP"] < 0])),  random_state=42)
mid  = all_smiles[(all_smiles["logP"] >= 0) & (all_smiles["logP"] <= 4)].sample(10000, random_state=42)
high = all_smiles[all_smiles["logP"] >  4].sample(min(5000,  len(all_smiles[all_smiles["logP"] > 4])),  random_state=42)
balanced = pd.concat([low, mid, high])

print(f"\nBalanced logP dataset: {len(balanced)} molecules")
print(f"  logP < 0  : {len(low)}")
print(f"  logP 0-4  : {len(mid)}")
print(f"  logP > 4  : {len(high)}")

balanced.to_csv("balanced_logp_dataset.csv", index=False)
\
qm9_sample = qm9_clean.sample(n=20000, random_state=42)
qm9_sample.to_csv("quantum_dataset.csv", index=False)

print(f"\nQuantum dataset: {len(qm9_sample)} molecules")
print(f"  dipole range : {qm9_sample['dipole'].min():.2f} to {qm9_sample['dipole'].max():.2f}")
print(f"  gap range    : {qm9_sample['gap'].min():.4f} to {qm9_sample['gap'].max():.4f}")
print("\nDone! Saved balanced_logp_dataset.csv and quantum_dataset.csv")