# DrugForge AI

DrugForge AI is a Flask-based computational drug discovery platform that integrates ligand prioritization, de novo molecule generation, re-screening, protein structure refinement, molecular docking, blind docking, ADMET prediction, QSAR modeling, hit-to-lead optimization, and protein–ligand interaction analysis within a single workflow.

## Features

- Graph neural network-based ligand prioritization
- Re-screening of uploaded molecular libraries
- De novo molecule generation
- Protein structure refinement
- Focused molecular docking
- Blind docking across multiple candidate sites
- ADMET prediction for single molecules and batch CSV inputs
- QSAR model training and external QSAR prediction
- Hit-to-lead optimization and analog prioritization
- Protein–ligand interaction analysis with downloadable tables and plots

## Main Modules

### 1. Ligand Prioritization
Upload a CSV file containing molecular data for prioritization using the graph-based screening workflow.

### 2. Re-screening
Upload a new molecular library for secondary filtering and prioritization.

### 3. De Novo Molecule Generation
Generate new candidate molecules and download CSV or SDF ZIP outputs.

### 4. QSAR Modeling
Train classification or regression QSAR models and reuse trained models for external prediction.

### 5. ADMET Prediction
Run ADMET prediction on a single SMILES string or a batch CSV file containing a `smiles` column.

### 6. Structure Refinement
Upload a protein PDB file for refinement and download processed structures and plots.

### 7. Molecular Docking
Run focused docking using a protein PDB file, ligand ZIP archive, and user-defined docking box coordinates.

### 8. Blind Docking
Run whole-protein site scanning and blind docking for uploaded ligands.

### 9. Protein–Ligand Interaction Analysis
Analyze a protein–ligand complex PDB file and generate interaction tables, plots, and reports.

### 10. Hit-to-Lead Optimization
Upload a CSV file with a `smiles` column to rank analogs and select follow-up candidates.

## Input Requirements

### General
- Protein files: `.pdb`
- Ligand screening files: `.csv`
- Docking ligand bundle: `.zip`
- Some workflows also generate `.sdf`, `.pdbqt`, `.json`, and `.png` outputs

### CSV expectations
- Ligand prioritization / screening workflows use CSV input
- ADMET batch prediction requires a `smiles` column
- QSAR training requires a `smiles` column and one target column
- QSAR external prediction requires a `smiles` column
- Hit-to-lead optimization requires a `smiles` column

## Project Structure

```text
DrugForgeAI/
├── app.py
├── gnn_model.py
├── gnn_utils.py
├── check_libraries.py
├── admet_bridge.py
├── qsar_bridge.py
├── hit_to_lead_bridge.py
├── plip_bridge.py
├── templates/
├── static/
├── models/
├── sample_input/
├── runtime/
├── third_party/
├── .gitignore
├── .env.example
├── requirements.txt
├── README.md
└── LICENSE