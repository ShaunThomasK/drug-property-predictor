import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def featurize(smi):
    try:
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
    except:
        return None

def load_and_featurize(csv, target_col):
    df = pd.read_csv(csv).dropna(subset=["smiles", target_col])
    df["smiles"] = df["smiles"].str.strip()
    print(f"  Featurizing {len(df)} molecules...")
    df["fp"] = df["smiles"].apply(featurize)
    df = df.dropna(subset=["fp"])
    X = np.array(df["fp"].tolist())
    y = df[target_col].values
    return X, y

def train_and_evaluate(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"  R² = {r2:.4f}  |  MAE = {mae:.4f}")

    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    plt.xlabel(f"Actual {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"{name} — R²={r2:.3f}, MAE={mae:.4f}")
    plt.tight_layout()
    plt.savefig(f"plot_{name}.png", dpi=150)
    plt.close()

    return model, r2, mae

tasks = [
    ("balanced_logp_dataset.csv", "logP",         "LogP"),
    ("quantum_dataset.csv",       "dipole",        "Dipole_Moment"),
    ("quantum_dataset.csv",       "homo",          "HOMO"),
    ("quantum_dataset.csv",       "lumo",          "LUMO"),
    ("quantum_dataset.csv",       "gap",           "HOMO_LUMO_Gap"),
    ("quantum_dataset.csv",       "total_energy",  "Total_Energy"),
    ("quantum_dataset.csv",       "polarizability","Polarizability"),
]

summary = []
for csv, target, name in tasks:
    print(f"\nTraining {name} model...")
    X, y = load_and_featurize(csv, target)
    model, r2, mae = train_and_evaluate(X, y, name)
    joblib.dump(model, f"model_{name}.pkl")
    print(f"  Saved model_{name}.pkl")
    summary.append({"Property": name, "R²": round(r2, 4), "MAE": round(mae, 4)})

print(f"\n{'='*50}")
print(f"  ALL MODELS TRAINED")
print(f"{'='*50}")
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
summary_df.to_csv("model_summary.csv", index=False)