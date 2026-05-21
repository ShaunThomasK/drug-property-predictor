import subprocess
import shutil
import os
import sys
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, FindMolChiralCenters

parser = argparse.ArgumentParser(description="Full conformer + property analysis for any molecule")
parser.add_argument("--smiles", type=str, help="SMILES string of the molecule")
parser.add_argument("--name",   type=str, default="molecule", help="Name for output files/folders")
parser.add_argument("--confs",  type=int, default=200, help="Number of conformers to generate (default: 200)")
parser.add_argument("--top",    type=int, default=10,  help="Top N conformers to pass to xTB (default: 10)")
args = parser.parse_args()

if not args.smiles:
    args.smiles = input("Enter SMILES string: ").strip()
if not args.smiles:
    print("No SMILES provided. Exiting.")
    sys.exit(1)

SMILES = args.smiles
NAME   = args.name.replace(" ", "_")

print(f"\n{'='*55}")
print(f"  Analyzing: {NAME}")
print(f"  SMILES   : {SMILES[:50]}{'...' if len(SMILES)>50 else ''}")
print(f"{'='*55}\n")

mol = Chem.MolFromSmiles(SMILES)
if mol is None:
    print("ERROR: Invalid SMILES string. Please check your input.")
    sys.exit(1)

print(f"[1/4] Generating {args.confs} conformers...")
mol3d = Chem.AddHs(mol)
AllChem.EmbedMultipleConfs(mol3d, numConfs=args.confs, randomSeed=42)

energies = []
for conf_id in range(mol3d.GetNumConformers()):
    AllChem.MMFFOptimizeMolecule(mol3d, confId=conf_id)
    props = AllChem.MMFFGetMoleculeProperties(mol3d)
    ff    = AllChem.MMFFGetMoleculeForceField(mol3d, props, confId=conf_id)
    if ff is not None:
        energies.append((conf_id, ff.CalcEnergy()))

if not energies:
    print("ERROR: Could not generate valid conformers for this molecule.")
    sys.exit(1)

energies.sort(key=lambda x: x[1])
top_confs = energies[:args.top]
print(f"  Generated {len(energies)} conformers. Top {args.top} selected.")
print(f"  Best MMFF energy: {top_confs[0][1]:.2f} kcal/mol (conformer {top_confs[0][0]})")

print(f"\n[2/4] Running xTB optimization on top {args.top} conformers...")

def write_xyz(mol, conf_id, path):
    conf  = mol.GetConformer(conf_id)
    atoms = list(mol.GetAtoms())
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\nconf_{conf_id}\n")
        for atom in atoms:
            pos = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()}  {pos.x:.6f}  {pos.y:.6f}  {pos.z:.6f}\n")

xtb_results = []
work_base   = f"{NAME}_xtb_work"
os.makedirs(work_base, exist_ok=True)

for rank, (conf_id, mmff_e) in enumerate(top_confs):
    xyz_name = f"conf_{rank}.xyz"
    xyz_path = os.path.join(work_base, xyz_name)
    work_dir = os.path.join(work_base, f"conf_{rank}")
    os.makedirs(work_dir, exist_ok=True)

    write_xyz(mol3d, conf_id, xyz_path)
    shutil.copy(xyz_path, work_dir)

    result = subprocess.run(
        ["xtb", xyz_name, "--opt", "--chrg", "0", "--uhf", "0"],
        capture_output=True, text=True, cwd=work_dir
    )
    out_file = os.path.join(work_dir, "full_output.txt")
    with open(out_file, "w") as f:
        f.write(result.stdout)
    energy = None
    for line in result.stdout.splitlines():
        if "TOTAL ENERGY" in line:
            try:
                energy = float(line.split()[3])
                break
            except:
                pass

    if energy is not None:
        xtb_results.append((rank, energy, work_dir))
        print(f"  Conformer {rank}: {energy:.6f} Hartree")
    else:
        print(f"  Conformer {rank}: xTB failed (skipping)")

if not xtb_results:
    print("ERROR: xTB failed on all conformers. Is xtb installed and in PATH?")
    sys.exit(1)

xtb_results.sort(key=lambda x: x[1])
winner_rank, winner_energy, winner_dir = xtb_results[0]

print(f"\n  ✓ Most stable conformer: conf_{winner_rank}")
print(f"    xTB energy: {winner_energy:.6f} Hartree")
print(f"    Geometry saved in: {winner_dir}/xtbopt.xyz")

print(f"\n[3/4] Parsing quantum properties from xTB output...")

dipole, homo, lumo, gap = None, None, None, None
winner_output = os.path.join(winner_dir, "full_output.txt")

try:
    with open(winner_output, "r") as f:
        lines = f.readlines()

    for j, line in enumerate(lines):
        if "molecular dipole:" in line.lower():
            for k in range(j+1, j+8):
                if k < len(lines) and "full" in lines[k].lower():
                    try:
                        dipole = float(lines[k].split()[-1])
                    except:
                        pass
        if "(HOMO)" in line:
            try:
                homo = float(line.split()[-2])
            except:
                pass
        if "(LUMO)" in line:
            try:
                lumo = float(line.split()[-2])
            except:
                pass
        if "HOMO-LUMO gap" in line and "::" in line:
            try:
                gap = float(line.split()[3])
            except:
                pass
except FileNotFoundError:
    print("  Warning: could not read xTB output for quantum properties.")

print(f"[4/4] Computing RDKit molecular properties...")

mw              = Descriptors.MolWt(mol)
logp            = Descriptors.MolLogP(mol)
hbd             = rdMolDescriptors.CalcNumHBD(mol)
hba             = rdMolDescriptors.CalcNumHBA(mol)
tpsa            = Descriptors.TPSA(mol)
rot_bonds       = rdMolDescriptors.CalcNumRotatableBonds(mol)
arom_rings      = rdMolDescriptors.CalcNumAromaticRings(mol)
rings           = rdMolDescriptors.CalcNumRings(mol)
frac_csp3       = Descriptors.FractionCSP3(mol)
mol_refract     = Descriptors.MolMR(mol)
heavy_atoms     = mol.GetNumHeavyAtoms()
formal_charge   = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
stereo_centers  = len(FindMolChiralCenters(mol, includeUnassigned=True))
lipinski_pass   = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)

print(f"\n{'='*55}")
print(f"   MOLECULAR PROPERTIES: {NAME.upper()}")
print(f"{'='*55}")

print("\n── Structure ──────────────────────────────────────────")
print(f"  Molecular Weight       : {mw:.2f} Da")
print(f"  Heavy Atom Count       : {heavy_atoms}")
print(f"  Formal Charge          : {formal_charge}")
print(f"  Stereocenters          : {stereo_centers}")
print(f"  Aromatic Rings         : {arom_rings}")
print(f"  Total Rings            : {rings}")
print(f"  Rotatable Bonds        : {rot_bonds}")
print(f"  Fraction sp3 Carbons   : {frac_csp3:.3f}")

print("\n── Lipinski Rule of 5 ─────────────────────────────────")
print(f"  Molecular Weight       : {mw:.2f}  (limit: ≤500)")
print(f"  LogP                   : {logp:.2f}  (limit: ≤5)")
print(f"  H-Bond Donors          : {hbd}     (limit: ≤5)")
print(f"  H-Bond Acceptors       : {hba}     (limit: ≤10)")
print(f"  ✓ PASSES Rule of 5     : {lipinski_pass}")

print("\n── ADMET-relevant ─────────────────────────────────────")
print(f"  TPSA                   : {tpsa:.2f} Å²  (good oral absorption if <140)")
print(f"  Molar Refractivity     : {mol_refract:.2f}  (drug-like range: 40-130)")

print("\n── Quantum Properties (xTB) ───────────────────────────")
print(f"  Total Energy           : {winner_energy:.6f} Hartree")
if dipole: print(f"  Dipole Moment          : {dipole:.3f} Debye")
if homo:   print(f"  HOMO Energy            : {homo:.4f} eV")
if lumo:   print(f"  LUMO Energy            : {lumo:.4f} eV")
if gap:    print(f"  HOMO-LUMO Gap          : {gap:.4f} eV")

print(f"\n{'='*55}")
print(f"  Most stable conformer  : {winner_dir}/xtbopt.xyz")
print(f"  Winning conformer rank : {winner_rank} of {args.top} xTB-optimized")
print(f"{'='*55}\n")
