import pandas as pd
import subprocess
import shutil
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import FindMolChiralCenters

NUM_CONFORMERS   = 100  
TOP_N            = 3    
OUTPUT_DIR       = "pipeline_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("molecules.csv")
print(f"Loaded {len(df)} molecules\n")

def get_rdkit_properties(mol):
    return {
        "MW":           round(Descriptors.MolWt(mol), 2),
        "LogP":         round(Descriptors.MolLogP(mol), 2),
        "HBD":          rdMolDescriptors.CalcNumHBD(mol),
        "HBA":          rdMolDescriptors.CalcNumHBA(mol),
        "TPSA":         round(Descriptors.TPSA(mol), 2),
        "RotBonds":     rdMolDescriptors.CalcNumRotatableBonds(mol),
        "AromaticRings":rdMolDescriptors.CalcNumAromaticRings(mol),
        "Stereocenters":len(FindMolChiralCenters(mol, includeUnassigned=True)),
        "FractionCSP3": round(Descriptors.FractionCSP3(mol), 3),
        "Ro5Pass":      (
            Descriptors.MolWt(mol)       <= 500 and
            Descriptors.MolLogP(mol)     <=   5 and
            rdMolDescriptors.CalcNumHBD(mol) <= 5 and
            rdMolDescriptors.CalcNumHBA(mol) <= 10
        )
    }

def get_top_conformers(mol, n=TOP_N):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=NUM_CONFORMERS, randomSeed=42)
    energies = []
    for conf_id in range(mol.GetNumConformers()):
        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        if ff:
            energies.append((conf_id, ff.CalcEnergy()))
    energies.sort(key=lambda x: x[1])
    return mol, energies[:n]

def write_xyz(mol, conf_id, path):
    conf = mol.GetConformer(conf_id)
    atoms = list(mol.GetAtoms())
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\nconf_{conf_id}\n")
        for atom in atoms:
            pos = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()}  {pos.x:.6f}  {pos.y:.6f}  {pos.z:.6f}\n")

def run_xtb(xyz_path, work_dir):
    os.makedirs(work_dir, exist_ok=True)
    shutil.copy(xyz_path, work_dir)
    fname = os.path.basename(xyz_path)
    result = subprocess.run(
        ["xtb", fname, "--opt", "--chrg", "0", "--uhf", "0"],
        capture_output=True, text=True, cwd=work_dir
    )
    output = result.stdout

    energy, homo, lumo, gap, dipole = None, None, None, None, None
    for i, line in enumerate(output.splitlines()):
        if "TOTAL ENERGY" in line:
            try: energy = float(line.split()[3])
            except: pass
        if "(HOMO)" in line:
            try: homo = float(line.split()[-2])
            except: pass
        if "(LUMO)" in line:
            try: lumo = float(line.split()[-2])
            except: pass
        if "HOMO-LUMO gap" in line and "::" in line:
            try: gap = float(line.split()[3])
            except: pass
        if "full" in line.lower() and dipole is None:
            try:
                val = float(line.split()[-1])
                if 0 < val < 50:
                    dipole = val
            except: pass

    return {"xtb_energy": energy, "HOMO_eV": homo,
            "LUMO_eV": lumo, "gap_eV": gap, "dipole_D": dipole}

all_results = []

for _, row in df.iterrows():
    name   = row["name"]
    smiles = row["smiles"]
    print(f"Processing {name}...")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  Could not parse SMILES for {name}, skipping")
        continue

    props = get_rdkit_properties(mol)

    try:
        mol3d, top_confs = get_top_conformers(mol)
    except Exception as e:
        print(f"  Conformer generation failed: {e}")
        continue
    best_xtb = None
    for rank, (conf_id, mmff_e) in enumerate(top_confs):
        xyz_path = os.path.join(OUTPUT_DIR, f"{name}_conf{rank}.xyz")
        work_dir = os.path.join(OUTPUT_DIR, f"{name}_xtb{rank}")
        write_xyz(mol3d, conf_id, xyz_path)
        xtb_result = run_xtb(xyz_path, work_dir)

        if xtb_result["xtb_energy"] is not None:
            if best_xtb is None or xtb_result["xtb_energy"] < best_xtb["xtb_energy"]:
                best_xtb = xtb_result

    if best_xtb:
        print(f"  ✓ xTB energy: {best_xtb['xtb_energy']:.4f} Hartree  |  gap: {best_xtb.get('gap_eV')} eV")
        props.update(best_xtb)
    else:
        print(f"  ✗ xTB failed")

    props["name"] = name
    props["smiles"] = smiles
    all_results.append(props)

results_df = pd.DataFrame(all_results)

cols = ["name", "smiles", "MW", "LogP", "HBD", "HBA", "TPSA",
        "RotBonds", "AromaticRings", "Stereocenters", "FractionCSP3",
        "Ro5Pass", "xtb_energy", "HOMO_eV", "LUMO_eV", "gap_eV", "dipole_D"]
results_df = results_df[[c for c in cols if c in results_df.columns]]

output_csv = "results.csv"
results_df.to_csv(output_csv, index=False)

print(f"\n{'='*55}")
print(f"  Done! {len(results_df)} molecules processed")
print(f"  Results saved to {output_csv}")
print(f"{'='*55}")
print(results_df[["name", "MW", "LogP", "Ro5Pass", "gap_eV"]].to_string(index=False))