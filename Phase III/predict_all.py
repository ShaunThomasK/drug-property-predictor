import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def featurize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = list(gen.GetFingerprint(mol))
    extra = [
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolMR(mol),
        Descriptors.FractionCSP3(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumRings(mol),
        mol.GetNumHeavyAtoms(),
    ]
    return fp + extra

models = {
    "LogP":           joblib.load("model_LogP.pkl"),
    "Dipole (D)":     joblib.load("model_Dipole_Moment.pkl"),
    "HOMO (Ha)":      joblib.load("model_HOMO.pkl"),
    "LUMO (Ha)":      joblib.load("model_LUMO.pkl"),
    "Gap (Ha)":       joblib.load("model_HOMO_LUMO_Gap.pkl"),
    "Total E (Ha)":   joblib.load("model_Total_Energy.pkl"),
    "Polarizability": joblib.load("model_Polarizability.pkl"),
}

def predict_all(smiles):
    fp = featurize(smiles)
    if fp is None:
        print("Invalid SMILES")
        return
    X = np.array([fp])
    print(f"\n{'='*45}")
    print(f"  Predictions for: {smiles[:40]}")
    print(f"{'='*45}")
    for prop, model in models.items():
        val = model.predict(X)[0]
        print(f"  {prop:<20} : {val:.4f}")
    print(f"{'='*45}")

tests = {
    "Ethanol":       "CCO",
    "Fexofenadine":  "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",
    "Caffeine":      "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
}
for name, smi in tests.items():
    print(f"\n{name}")
    predict_all(smi)

print("\n\nEnter any SMILES to predict all properties (or 'quit'):")
while True:
    smi = input("SMILES: ").strip()
    if smi.lower() == "quit":
        break
    predict_all(smi)