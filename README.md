# Drug Property Predictor

Hey! This terminal-based drug property predictor was a passion project of mine, and I'm really happy that I've gotten to complete it! The idea for this came about when I'd approached a prof, interested to do research under him, and he suggested that I learn how to figure out the properties for a drug molecule of my choice. This led me down a rabbit hole of the possibilities I could explore using the RDKit, xTB and sklearn modules in Python, and I learnt how to train a model in Python for the first time! Do take a look through the README before you go on to run any of the code, as it has the instructions to run said code + the prerequisite Python modules you need to download, etc. Hope you enjoy looking through my project! :D

---

## Prerequisites

### 1. Set up the conda environment

```bash
conda create -n drugdiscovery python=3.11
conda activate drugdiscovery
```

### 2. Install dependencies

```bash
conda install -c conda-forge rdkit xtb
pip install scikit-learn matplotlib seaborn pandas joblib
```

Every time you open a new terminal session, run `conda activate drugdiscovery` before running any scripts.

### 3. Download the required datasets (for Phase III only)

```bash
# ZINC dataset — 250k drug-like molecules
wget https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv

# QM9 dataset — 130k molecules with quantum properties
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv
```

---

## Phase I: Single Molecule Analysis

**File:** `analyze_molecule.py`

This was the starting point of the project. Given any molecule's SMILES string as input, this script runs a full computational chemistry pipeline on it, with no prior knowledge of the molecule's 3D structure needed. The pipeline has three stages:

1. **Conformer generation:** RDKit generates up to 200 possible 3D shapes (conformers) of the molecule by rotating its bonds into different configurations, then minimizes each one using the MMFF94 classical force field
2. **Quantum optimization:** the top 10 lowest-energy conformers are passed to xTB, a semi-empirical quantum chemistry tool, which re-optimizes each one using real quantum mechanical calculations. The conformer with the lowest xTB energy is selected as the most stable
3. **Property extraction:** molecular properties are computed from both RDKit (structural and drug-likeness properties) and the xTB output (quantum properties)

The properties reported include:

| Property | What it means |
|---|---|
| Molecular Weight | Size of the molecule in Daltons |
| LogP | Fat-solubility — how well the molecule dissolves in lipids vs water |
| H-Bond Donors/Acceptors | Hydrogen bonding ability, affects membrane permeability |
| TPSA | Polar surface area — predicts gut and blood-brain barrier absorption |
| Lipinski Rule of 5 | Standard drug-likeness check for oral bioavailability |
| HOMO / LUMO energies | Frontier molecular orbitals — indicate chemical reactivity |
| HOMO-LUMO Gap | Larger gap = more chemically stable and less reactive |
| Dipole Moment | Overall polarity of the molecule in its most stable 3D shape |
| Total Energy | Ground state energy of the optimized conformer in Hartree |

### Usage

To run Phase I, just type the command `python analyze_molecule.py` in the conda environment. To optimize the process & add additional information to the molecule whose SMILES string you're providing, you can carry out the commands as shown below.

**With arguments:**
```bash
python analyze_molecule.py --smiles "CCO" --name Ethanol
python analyze_molecule.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --name Aspirin
```

**Optional flags:**
```bash
--confs 200    # number of conformers to generate (default: 200)
--top 10       # number of conformers to pass to xTB (default: 10)
```

You can look up the SMILES string for any drug on [PubChem](https://pubchem.ncbi.nlm.nih.gov).

---

## Phase II: Multi-Molecule Pipeline

**Files:** `pipeline.py`, `molecules.csv`

Once the single-molecule workflow was working, the natural next step was scaling it up. This phase runs the same conformer -> xTB -> properties pipeline automatically across an entire list of molecules defined in `molecules.csv`, and outputs a single ranked CSV with all results side by side.

The included `molecules.csv` has 10 well-known drugs spanning a range of sizes and uses to demonstrate the breadth of what the pipeline can handle. You can swap in any list of molecules by editing the CSV; the only required columns are `name` and `smiles`.

This is essentially what virtual screening pipelines in drug discovery do; automatically profiling large numbers of candidate molecules so that researchers can filter and rank them by desired properties.

### Usage

Edit `molecules.csv` with your molecules of choice, then:

```bash
python pipeline.py
```

Output is saved to `results.csv`. Each row is one molecule; columns include all structural, Lipinski, ADMET, and xTB quantum properties.

---

## Phase III: ML Property Predictor

**Files:** `build_dataset.py`, `train_all_models.py`, `predict_all.py`

As I was building Phase II, I started thinking about the possibility of making an interactive terminal-based property predictor that takes in a molecule's SMILES string and uses that to calculate all the relevant properties of the molecule. But rather than having to run this with xTB every time, I wanted to implement this by training models on a large-enough dataset and use that to predict the properties of a molecule! This was my first time working with the models from the scikit-learn library in Python, and I learned a LOT from implemeting this.

Seven separate Gradient Boosting models were trained, one per property. Each molecule is represented as a **Morgan fingerprint** (a 2048-bit vector encoding the presence or absence of various structural features around each atom) combined with a set of RDKit descriptors. This vector is what the model actually learns from.

Training data comes from two sources:
- **ZINC** (250k drug-like molecules) for logP coverage across a wide range of molecular sizes
- **QM9** (130k molecules with DFT-computed quantum properties) for HOMO, LUMO, gap, dipole, and total energy

The two datasets are merged and balanced before training so the models aren't biased toward the middle of any property range.

### Model performance

| Property | R² | MAE |
|---|---|---|
| LogP | 0.9498 | 0.31 |
| Total Energy | 0.9946 | 1.69 Ha |
| Polarizability | 0.9713 | 0.97 |
| LUMO | 0.9265 | 0.009 Ha |
| HOMO-LUMO Gap | 0.8796 | 0.012 Ha |
| HOMO | 0.7700 | 0.008 Ha |
| Dipole Moment | 0.5494 | 0.71 D |

Dipole moment is the hardest to predict because it depends heavily on the molecule's 3D geometry, which is something a 2D fingerprint can only approximate, not exactly replicate. All other properties perform strongly.

### Usage

**Step 1** - build the training datasets (run once):
```bash
python build_dataset.py
```

**Step 2** - train all 7 models (takes ~15-20 minutes):
```bash
python train_all_models.py
```

This saves one `.pkl` file per property. Once trained, you never need to retrain unless you want to update the models.

**Step 3** - run the predictor:
```bash
python predict_all.py
```

The predictor runs interactively: enter any SMILES string and all 7 properties are predicted instantly, without xTB.

---

## Project Structure

```
└── Phase I
    └── analyze_molecule.py       # single molecule analysis
└── Phase II
    ├── pipeline.py               # multi-molecule pipeline
    ├── molecules.csv             # input molecule list
    └── results.csv               # pipeline output (generated by pipeline.py)
└── Phase III
    ├── build_dataset.py          # dataset preparation
    ├── train_all_models.py       # model training
    ├── predict_all.py            # interactive predictor
    └── model_*.pkl               # binary file (pickled) of trained models (generated by train_all_models.py)
```

---

## Limitations

This project is a learning exercise, not a production tool. A few known limitations worth being aware of:

- The ML models were trained on ZINC and QM9, which skew toward small-to-medium drug-like molecules. Predictions for very large or very unusual molecules may be less accurate
- Dipole moment predictions are weaker because the property is inherently 3D-dependent, and fingerprints encode 2D structure
- The conformer search in Phase I is stochastic, in the sense that different random seeds may find slightly different "most stable" conformers for flexible molecules
- Graph neural networks (GNNs) such as MPNN or SchNet would likely outperform fingerprint-based models here. I just decided to use the Morgan fingerprint model as I was just getting started on how to work with models, and I wanted to learn about the basics 