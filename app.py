import os
os.environ["OMP_NUM_THREADS"] = "1"

from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
import csv
import json
import time
import uuid
import zipfile
import shutil
import random
import logging
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from datetime import datetime

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import torch
import torch.nn as nn

from flask import (
    Flask,
    render_template,
    request,
    flash,
    redirect,
    url_for,
    send_from_directory,
    jsonify,
    abort,
    send_file,
    current_app
)
from werkzeug.utils import secure_filename

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, rdMolDescriptors, QED, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
    davies_bouldin_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from gnn_model import GNN
from gnn_utils import MoleculeDataset, collate, standardize_smiles_and_mol

from openmm.app import PDBFile, Modeller, ForceField, Simulation, PME, HBonds
from openmm import LangevinMiddleIntegrator
from openmm.unit import kelvin, picosecond, picoseconds, nanometer
from pdbfixer import PDBFixer

app = Flask(__name__)

BASE_DIR = app.root_path
RUNTIME_DIR = os.path.join(BASE_DIR, 'runtime')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
THIRD_PARTY_DIR = os.path.join(BASE_DIR, 'third_party')

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-this-before-production')

# Runtime / storage paths
app.config['UPLOAD_FOLDER'] = os.path.join(RUNTIME_DIR, 'uploads')
app.config['UPLOADED_FILES_DIR'] = app.config['UPLOAD_FOLDER']
app.config['GENERATED_FILES_DIR'] = os.path.join(RUNTIME_DIR, 'generated_files')
app.config['DOCKING_RESULTS_DIR'] = os.path.join(RUNTIME_DIR, 'docking_results')
app.config['BLIND_JOBS_DIR'] = os.path.join(RUNTIME_DIR, 'blind_jobs')
app.config['ADMET_JOBS_DIR'] = os.path.join(RUNTIME_DIR, 'admet_jobs')
app.config['QSAR_JOBS_DIR'] = os.path.join(RUNTIME_DIR, 'qsar_jobs')
app.config['HIT_TO_LEAD_JOBS_DIR'] = os.path.join(RUNTIME_DIR, 'hit_to_lead_jobs')
app.config['PLIP_JOBS_DIR'] = os.path.join(RUNTIME_DIR, 'plip_jobs')

# Model / script paths
app.config['QSAR_MODELS_DIR'] = os.path.join(MODELS_DIR, 'qsar_saved_models')
app.config['ADMET_BRIDGE_SCRIPT'] = os.path.join(BASE_DIR, 'admet_bridge.py')
app.config['QSAR_BRIDGE_SCRIPT'] = os.path.join(BASE_DIR, 'qsar_bridge.py')
app.config['HIT_TO_LEAD_BRIDGE_SCRIPT'] = os.path.join(BASE_DIR, 'hit_to_lead_bridge.py')
app.config['PLIP_BRIDGE_SCRIPT'] = os.path.join(BASE_DIR, 'plip_bridge.py')

# Python executables
app.config['ADMET_PYTHON_EXE'] = os.environ.get('ADMET_PYTHON_EXE', sys.executable)
app.config['QSAR_PYTHON_EXE'] = os.environ.get('QSAR_PYTHON_EXE', sys.executable)
app.config['HIT_TO_LEAD_PYTHON_EXE'] = os.environ.get('HIT_TO_LEAD_PYTHON_EXE', app.config['QSAR_PYTHON_EXE'])
app.config['PLIP_PYTHON_EXE'] = os.environ.get('PLIP_PYTHON_EXE', app.config['QSAR_PYTHON_EXE'])

# Third-party tool paths
app.config['THIRD_PARTY_DIR'] = THIRD_PARTY_DIR
app.config['AUTOGROW4_DIR'] = os.path.join(THIRD_PARTY_DIR, 'AutoGrow4')
app.config['PLIP_TOOL_DIR'] = os.path.join(THIRD_PARTY_DIR, 'PLIP')

# General settings
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'zip', 'pdb', 'sdf'}
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 200 * 1024 * 1024))
app.config['VINA_EXECUTABLE'] = os.environ.get('VINA_EXECUTABLE', '')

for directory in [
    RUNTIME_DIR,
    STATIC_DIR,
    MODELS_DIR,
    THIRD_PARTY_DIR,
    app.config['UPLOAD_FOLDER'],
    app.config['GENERATED_FILES_DIR'],
    app.config['DOCKING_RESULTS_DIR'],
    app.config['BLIND_JOBS_DIR'],
    app.config['ADMET_JOBS_DIR'],
    app.config['QSAR_JOBS_DIR'],
    app.config['QSAR_MODELS_DIR'],
    app.config['HIT_TO_LEAD_JOBS_DIR'],
    app.config['PLIP_JOBS_DIR'],
    app.config['AUTOGROW4_DIR'],
    app.config['PLIP_TOOL_DIR'],
    os.path.join(STATIC_DIR, 'css'),
    os.path.join(STATIC_DIR, 'js'),
    os.path.join(STATIC_DIR, 'img')
]:
    os.makedirs(directory, exist_ok=True)

SCREENING_THRESHOLD = 0.30
FALLBACK_TOP_K = 100
MAX_PLOT_POINTS = 3000
TOP_HITS_TO_ALWAYS_PLOT = 200
PLOT_RANDOM_SEED = 42
MODEL_HIDDEN_SIZE = 128
TRAIN_BATCH_SIZE = 64
TRAIN_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(level=logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("umap").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# =========================
# SAFETY / SCIENTIFIC HELPERS
# =========================

def resolve_executable(candidates, description):
    for candidate in candidates:
        if not candidate:
            continue
        candidate = str(candidate).strip()
        if not candidate:
            continue
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"{description} not found. Checked: {candidates}")


def resolve_python_executable(configured_path):
    return resolve_executable([configured_path, os.environ.get('PYTHON_EXECUTABLE'), sys.executable, 'python', 'python3'], 'Python executable')


def resolve_vina_executable():
    return resolve_executable([app.config.get('VINA_EXECUTABLE'), os.environ.get('VINA_EXECUTABLE'), os.path.join(app.root_path, 'vina.exe'), os.path.join(app.root_path, 'vina'), '.\vina.exe', 'vina.exe', 'vina'], 'AutoDock Vina executable')


def secure_job_path(base_dir, relative_path):
    base_dir_real = os.path.realpath(base_dir)
    target_real = os.path.realpath(os.path.join(base_dir, relative_path))
    if not (target_real == base_dir_real or target_real.startswith(base_dir_real + os.sep)):
        raise FileNotFoundError('Invalid file path requested.')
    return target_real


def safe_send_job_file(base_dir, relative_path, as_attachment=False, download_name=None):
    target_path = secure_job_path(base_dir, relative_path)
    if not os.path.exists(target_path):
        raise FileNotFoundError(f'File not found: {relative_path}')
    return send_file(target_path, as_attachment=as_attachment, download_name=download_name or os.path.basename(target_path))


def safe_extract_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            target = os.path.realpath(os.path.join(extract_dir, member.filename))
            base = os.path.realpath(extract_dir)
            if not (target == base or target.startswith(base + os.sep)):
                raise RuntimeError(f'Unsafe path in ZIP archive: {member.filename}')
        zip_ref.extractall(extract_dir)


def run_command_capture(cmd, timeout=3600, cwd=None, env=None, check=True):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}.\nCommand: {' '.join(map(str, cmd))}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return result


def compute_lipinski_violations(mol):
    violations = 0
    if Descriptors.MolWt(mol) > 500:
        violations += 1
    if Descriptors.MolLogP(mol) > 5:
        violations += 1
    if Lipinski.NumHDonors(mol) > 5:
        violations += 1
    if Lipinski.NumHAcceptors(mol) > 10:
        violations += 1
    return violations


def molecule_profile_from_smiles(smiles):
    canonical_smiles, mol = standardize_smiles_and_mol(smiles)
    if canonical_smiles is None or mol is None:
        return None
    return {
        'CanonicalSMILES': canonical_smiles,
        'MW': round(Descriptors.MolWt(mol), 3),
        'LogP': round(Descriptors.MolLogP(mol), 3),
        'TPSA': round(rdMolDescriptors.CalcTPSA(mol), 3),
        'HBA': int(Lipinski.NumHAcceptors(mol)),
        'HBD': int(Lipinski.NumHDonors(mol)),
        'RotBonds': int(Lipinski.NumRotatableBonds(mol)),
        'Rings': int(rdMolDescriptors.CalcNumRings(mol)),
        'HeavyAtoms': int(mol.GetNumHeavyAtoms()),
        'QED': round(float(QED.qed(mol)), 4),
        'LipinskiViolations': int(compute_lipinski_violations(mol)),
        'Scaffold': MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or canonical_smiles
    }


PROFILE_COLUMNS = ['MW', 'LogP', 'TPSA', 'HBA', 'HBD', 'RotBonds', 'Rings', 'HeavyAtoms', 'QED', 'LipinskiViolations', 'Scaffold']
PRIORITY_COLUMNS = ['Confidence', 'DrugLikeScore', 'PriorityScore']


def add_molecule_profiles(df, smiles_col='Compound'):
    if df is None or len(df) == 0 or smiles_col not in df.columns:
        return df

    out = df.copy().reset_index(drop=True)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    drop_cols = ['CanonicalSMILES'] + [c for c in PROFILE_COLUMNS if c in out.columns]
    out = out.drop(columns=drop_cols, errors='ignore')

    smiles_values = out[smiles_col].astype(str).tolist()
    profiles = []
    for smiles in smiles_values:
        profile = molecule_profile_from_smiles(smiles)
        if profile is None:
            profile = {
                'CanonicalSMILES': smiles,
                'MW': np.nan,
                'LogP': np.nan,
                'TPSA': np.nan,
                'HBA': np.nan,
                'HBD': np.nan,
                'RotBonds': np.nan,
                'Rings': np.nan,
                'HeavyAtoms': np.nan,
                'QED': np.nan,
                'LipinskiViolations': np.nan,
                'Scaffold': 'NA'
            }
        profiles.append(profile)

    prof_df = pd.DataFrame(profiles).reset_index(drop=True)
    out = out.drop(columns=[smiles_col], errors='ignore')
    out.insert(0, smiles_col, prof_df['CanonicalSMILES'].astype(str))
    prof_df = prof_df.drop(columns=['CanonicalSMILES'], errors='ignore')
    out = pd.concat([out.reset_index(drop=True), prof_df.reset_index(drop=True)], axis=1)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def add_priority_scores(df):
    if df is None or len(df) == 0:
        return df

    df = df.copy().reset_index(drop=True)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.drop(columns=[c for c in PRIORITY_COLUMNS if c in df.columns], errors='ignore')
    df = add_molecule_profiles(df, 'Compound')

    prob = pd.to_numeric(df['Probability'], errors='coerce').fillna(0.0)
    df['Confidence'] = (2.0 * np.abs(prob - 0.5)).round(4)
    qed_term = pd.to_numeric(df['QED'], errors='coerce').fillna(0.0).clip(0.0, 1.0)
    lipinski_bonus = (1.0 - (pd.to_numeric(df['LipinskiViolations'], errors='coerce').fillna(4).clip(0, 4) / 4.0)).clip(0.0, 1.0)
    df['DrugLikeScore'] = (0.65 * qed_term + 0.35 * lipinski_bonus).round(4)
    df['PriorityScore'] = (
        0.65 * prob +
        0.20 * df['Confidence'].fillna(0.0) +
        0.15 * df['DrugLikeScore'].fillna(0.0)
    ).round(4)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def diversify_hits_by_scaffold(df, max_per_scaffold=3):
    if df is None or len(df) == 0:
        return df
    working = add_priority_scores(df.copy())
    working = working.sort_values(['PriorityScore', 'Probability'], ascending=[False, False]).reset_index(drop=True)
    selected = []
    scaffold_counts = {}
    for _, row in working.iterrows():
        scaffold = row.get('Scaffold', 'NA') or 'NA'
        if scaffold_counts.get(scaffold, 0) >= max_per_scaffold:
            continue
        selected.append(row)
        scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1
    return pd.DataFrame(selected) if selected else working


def align_screening_arrays(smiles, probs, labels=None, preds=None):
    lengths = [len(smiles), len(probs)]
    if labels is not None:
        lengths.append(len(labels))
    if preds is not None:
        lengths.append(len(preds))
    n = min(lengths) if lengths else 0
    smiles = list(smiles[:n])
    probs = np.asarray(probs[:n])
    labels = np.asarray(labels[:n]) if labels is not None else None
    preds = np.asarray(preds[:n]) if preds is not None else None
    return smiles, probs, labels, preds


def embed_and_optimize_mol(mol, random_seed=RANDOM_SEED):
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        AllChem.EmbedMolecule(mol, randomSeed=int(random_seed), useRandomCoords=True)
    AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    return mol


def write_3d_sdf_with_properties(smiles, out_path, properties=None, mol_name='Ligand'):
    canonical_smiles, mol = standardize_smiles_and_mol(smiles)
    if canonical_smiles is None or mol is None:
        return False
    mol = embed_and_optimize_mol(mol)
    mol.SetProp('_Name', mol_name)
    if properties:
        for key, value in properties.items():
            mol.SetProp(str(key), str(value))
    writer = Chem.SDWriter(out_path)
    writer.write(mol)
    writer.close()
    return True


def plot_probability_distribution(df, path, title='Probability Distribution of Screened Molecules'):
    if df is None or len(df) == 0:
        return None
    plt.figure(figsize=(8, 5))
    plt.hist(df['Probability'], bins=30, edgecolor='black')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def plot_top_hits_bar(df, path, score_col='PriorityScore', title='Top Ranked Hits', top_n=20):
    if df is None or len(df) == 0:
        return None
    working = df.copy()
    if score_col not in working.columns:
        score_col = 'Probability'
    working = working.sort_values(score_col, ascending=False).head(top_n)
    labels = [f'Hit {i+1}' for i in range(len(working))]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, working[score_col])
    plt.xticks(rotation=60, ha='right')
    plt.ylabel(score_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def plot_drug_space(df, path, title='Drug-Likeness Space (MW vs LogP)'):
    if df is None or len(df) == 0:
        return None
    working = add_molecule_profiles(df.copy(), 'Compound')
    working = working.dropna(subset=['MW', 'LogP'])
    if len(working) == 0:
        return None
    plt.figure(figsize=(8, 6))
    plt.scatter(working['MW'], working['LogP'], alpha=0.75)
    plt.xlabel('Molecular Weight')
    plt.ylabel('LogP')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def plot_cluster_distribution(df, path, title='Cluster Distribution'):
    if df is None or len(df) == 0 or 'Cluster' not in df.columns:
        return None
    counts = df['Cluster'].value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    counts.plot(kind='bar')
    plt.xlabel('Cluster')
    plt.ylabel('Molecule Count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def plot_roc_curve_image(labels, probs, path, title='ROC Curve'):
    if labels is None or probs is None or len(np.unique(labels)) < 2:
        return None
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def plot_pr_curve_image(labels, probs, path, title='Precision-Recall Curve'):
    if labels is None or probs is None or len(np.unique(labels)) < 2:
        return None
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def plot_confusion_matrix_image(labels, preds, path, title='Confusion Matrix'):
    if labels is None or preds is None or len(labels) == 0:
        return None
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, [str(i) for i in tick_marks])
    plt.yticks(tick_marks, [str(i) for i in tick_marks])
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def plot_training_history(history, path, title='Training / Validation Loss'):
    if not history:
        return None
    plt.figure(figsize=(8, 5))
    if history.get('train_loss'):
        plt.plot(history['train_loss'], label='Train Loss')
    if history.get('val_loss'):
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def render_image_card_html(title, filename):
    if not filename:
        return ''
    return f"<div style='display:inline-block;vertical-align:top;margin:12px;width:31%;min-width:260px;'><div style='border:1px solid #ddd;border-radius:10px;padding:10px;background:#fff;'><div style='font-weight:600;margin-bottom:8px;text-align:center;'>{title}</div><img src='/images/{filename}' alt='{title}' style='width:100%;height:auto;border-radius:6px;'><div style='text-align:center;margin-top:8px;'><a href='/images/{filename}' download='{filename}' style='text-decoration:none;'>Download</a></div></div></div>"


def build_dashboard_html(metrics_rows, image_items, heading='Analysis Summary'):
    metric_rows_html = ''.join([f"<tr><td style='padding:6px 10px;border:1px solid #ddd;'>{k}</td><td style='padding:6px 10px;border:1px solid #ddd;'>{v}</td></tr>" for k, v in metrics_rows])
    image_html = ''.join([render_image_card_html(title, filename) for title, filename in image_items if filename])
    return f"<div style='margin:16px 0 24px 0;'><h3 style='margin-bottom:12px;'>{heading}</h3><table style='border-collapse:collapse;width:100%;max-width:680px;background:#fff;'>{metric_rows_html}</table><div style='margin-top:18px;'>{image_html}</div></div>"


def get_first_column(columns, candidates):
    lowered = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def generate_hit_to_lead_fallback_artifacts(results_csv_path, job_dir):
    if not results_csv_path or not os.path.exists(results_csv_path):
        return None, None, None
    df = pd.read_csv(results_csv_path)
    if len(df) == 0:
        return None, None, None
    smiles_col = get_first_column(df.columns, ['smiles', 'analog_smiles', 'compound', 'molecule'])
    score_col = get_first_column(df.columns, ['priorityscore', 'priority_score', 'similarity', 'finalscore', 'final_score', 'score'])
    rank_plot_path = os.path.join(job_dir, 'plots', 'ranking_plot.png')
    prop_plot_path = os.path.join(job_dir, 'plots', 'property_distribution.png')
    summary_json_path = os.path.join(job_dir, 'reports', 'summary.json')
    os.makedirs(os.path.dirname(rank_plot_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)
    working = df.copy()
    if smiles_col:
        working = working.rename(columns={smiles_col: 'Compound'})
        working = add_molecule_profiles(working, 'Compound')
        plot_drug_space(working[['Compound']].drop_duplicates(), prop_plot_path, title='Hit-to-Lead Drug Space')
    if score_col and score_col in working.columns:
        tmp = working.rename(columns={score_col: 'PriorityScore'})
        plot_top_hits_bar(tmp, rank_plot_path, score_col='PriorityScore', title='Hit-to-Lead Ranking', top_n=min(20, len(tmp)))
    summary = {'rows': int(len(working)), 'columns': list(map(str, working.columns)), 'unique_scaffolds': int(working['Scaffold'].nunique()) if 'Scaffold' in working.columns else None}
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return rank_plot_path if os.path.exists(rank_plot_path) else None, prop_plot_path if os.path.exists(prop_plot_path) else None, summary_json_path


def generate_plip_fallback_artifacts(interactions_csv_path, job_dir):
    if not interactions_csv_path or not os.path.exists(interactions_csv_path):
        return None, None, None
    df = pd.read_csv(interactions_csv_path)
    if len(df) == 0:
        return None, None, None
    interaction_col = get_first_column(df.columns, ['interaction_type', 'interaction', 'type'])
    residue_col = get_first_column(df.columns, ['residue', 'residue_name', 'restype'])
    counts_csv_path = os.path.join(job_dir, 'output', 'interaction_counts.csv')
    interaction_plot_path = os.path.join(job_dir, 'plots', 'interaction_barplot.png')
    residue_plot_path = os.path.join(job_dir, 'plots', 'binding_site_overview.png')
    summary_json_path = os.path.join(job_dir, 'reports', 'summary.json')
    os.makedirs(os.path.dirname(counts_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)
    if interaction_col:
        counts_df = df.groupby(interaction_col).size().reset_index(name='count').sort_values('count', ascending=False)
        counts_df.to_csv(counts_csv_path, index=False)
        plt.figure(figsize=(8, 5))
        plt.bar(counts_df[interaction_col].astype(str), counts_df['count'])
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Count')
        plt.title('Interaction Type Distribution')
        plt.tight_layout()
        plt.savefig(interaction_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    if residue_col:
        residue_df = df.groupby(residue_col).size().reset_index(name='count').sort_values('count', ascending=False).head(20)
        plt.figure(figsize=(9, 5))
        plt.bar(residue_df[residue_col].astype(str), residue_df['count'])
        plt.xticks(rotation=60, ha='right')
        plt.ylabel('Contacts')
        plt.title('Top Contact Residues')
        plt.tight_layout()
        plt.savefig(residue_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    summary = {'rows': int(len(df)), 'columns': list(map(str, df.columns))}
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return counts_csv_path if os.path.exists(counts_csv_path) else None, interaction_plot_path if os.path.exists(interaction_plot_path) else None, residue_plot_path if os.path.exists(residue_plot_path) else None


def generate_docking_summary_plots(csv_path, output_dir, prefix='docking'):
    if not csv_path or not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    if len(df) == 0 or 'binding_affinity' not in df.columns:
        return None, None
    hist_path = os.path.join(output_dir, f'{prefix}_affinity_hist.png')
    bar_path = os.path.join(output_dir, f'{prefix}_top_hits.png')
    plt.figure(figsize=(8, 5))
    plt.hist(df['binding_affinity'], bins=20, edgecolor='black')
    plt.xlabel('Binding Affinity (kcal/mol)')
    plt.ylabel('Count')
    plt.title('Docking Score Distribution')
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    label_col = 'file_name' if 'file_name' in df.columns else ('ligand_name' if 'ligand_name' in df.columns else None)
    if label_col:
        top_df = df.sort_values('binding_affinity', ascending=True).head(15)
        plt.figure(figsize=(10, 6))
        plt.bar(top_df[label_col].astype(str), top_df['binding_affinity'])
        plt.xticks(rotation=60, ha='right')
        plt.ylabel('Binding Affinity (kcal/mol)')
        plt.title('Top Docking Hits')
        plt.tight_layout()
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
    return hist_path if os.path.exists(hist_path) else None, bar_path if os.path.exists(bar_path) else None


DENOVO_ACID_FRAGMENTS = ['O=C(O)c1ccccc1', 'O=C(O)c1ccncc1', 'CC(=O)O', 'O=C(O)c1ccc(Cl)cc1', 'O=C(O)c1ccc(F)cc1', 'O=C(O)c1ccc(CN)cc1', 'O=C(O)c1ccoc1', 'O=C(O)c1ccsc1', 'O=C(O)CCc1ccccc1', 'O=C(O)c1ncccc1']
DENOVO_AMINE_FRAGMENTS = ['NC1CCCCC1', 'NCCO', 'NCCN', 'Nc1ccccc1', 'NCc1ccccc1', 'N1CCOCC1', 'N1CCNCC1', 'CCN', 'CC(C)N', 'NCCc1ccccc1', 'NCC1=CC=CC=C1F', 'NCC1=CN=CC=C1', 'NCCOC', 'NCc1nccs1', 'NCc1ccoc1', 'NCC(C)O', 'NCCC(N)=O', 'NCCS', 'NCC1CC1', 'NCc1ccc(Cl)cc1']


def generate_denovo_library(num_molecules, apply_lipinski=True):
    reaction = AllChem.ReactionFromSmarts('[C:1](=[O:2])[OH].[N:3]>>[C:1](=[O:2])[N:3]')
    generated = []
    seen = set()
    acid_mols = [Chem.MolFromSmiles(s) for s in DENOVO_ACID_FRAGMENTS]
    amine_mols = [Chem.MolFromSmiles(s) for s in DENOVO_AMINE_FRAGMENTS]
    for acid in acid_mols:
        for amine in amine_mols:
            try:
                products = reaction.RunReactants((acid, amine))
            except Exception:
                products = []
            for prod_tuple in products:
                if not prod_tuple:
                    continue
                mol = prod_tuple[0]
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    continue
                smiles = Chem.MolToSmiles(mol, canonical=True)
                if smiles in seen:
                    continue
                if apply_lipinski and compute_lipinski_violations(mol) > 1:
                    continue
                generated.append((smiles, 1))
                seen.add(smiles)
                if len(generated) >= num_molecules:
                    return generated
    return generated


# =========================
# GENERAL HELPERS
# =========================

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass




def save_data_to_csv(data, filename):
    with open(filename, 'w', newline='') as csv_file:
        fieldnames = ['SMILES', 'Activity']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for smiles, activity in data:
            writer.writerow({'SMILES': smiles, 'Activity': activity})


def preprocess_csv(file):
    try:
        df = pd.read_csv(file)

        if 'SMILES' not in df.columns or 'Activity' not in df.columns:
            flash('The CSV file must have "SMILES" and "Activity" columns.', 'error')
            return None

        df['SMILES'] = df['SMILES'].apply(
            lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), canonical=True)
            if Chem.MolFromSmiles(x) is not None else None
        )

        df.dropna(subset=['SMILES'], inplace=True)

        timestamp = int(time.time())
        filename = f'uploaded_data_{timestamp}.csv'
        save_data_to_csv(df.values.tolist(), filename)

        return df
    except Exception as e:
        flash(f'Error processing the CSV file: {str(e)}', 'error')
        return None


def resolve_case_insensitive_column(columns, desired_name):
    for col in columns:
        if str(col).strip().lower() == str(desired_name).strip().lower():
            return col
    return None


def first_existing_path(path_list):
    for path in path_list:
        if path and os.path.exists(path):
            return path
    return None


def safe_read_csv_to_html(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        return df.to_html(classes='table table-striped table-bordered table-hover', index=False)
    except Exception:
        return None


def safe_read_json_file(json_path):
    if not json_path or not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def safe_read_text_file(text_path):
    if not text_path or not os.path.exists(text_path):
        return None
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return None


def clear_workspace(workspace_path):
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path)
    os.makedirs(workspace_path, exist_ok=True)


def create_zip_from_directory(directory_path):
    zip_in_memory = BytesIO()
    with zipfile.ZipFile(zip_in_memory, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, directory_path))
    zip_in_memory.seek(0)
    return zip_in_memory


def clean_virtual_screening_training_dataframe(df):
    smiles_col = resolve_case_insensitive_column(df.columns, 'SMILES')
    activity_col = resolve_case_insensitive_column(df.columns, 'Activity')

    if smiles_col is None or activity_col is None:
        raise ValueError('The CSV file must contain "SMILES" and "Activity" columns.')

    clean_df = df[[smiles_col, activity_col]].copy()
    clean_df.columns = ['SMILES', 'Activity']
    clean_df = clean_df.dropna(subset=['SMILES', 'Activity']).copy()
    clean_df['SMILES'] = clean_df['SMILES'].astype(str).str.strip()
    clean_df['Activity'] = pd.to_numeric(clean_df['Activity'], errors='coerce')
    clean_df = clean_df.dropna(subset=['Activity']).copy()
    clean_df['Activity'] = clean_df['Activity'].astype(int)
    clean_df = clean_df[clean_df['Activity'].isin([0, 1])].copy()

    clean_df['SMILES'] = clean_df['SMILES'].apply(lambda s: standardize_smiles_and_mol(s)[0])
    clean_df = clean_df.dropna(subset=['SMILES']).copy()
    clean_df = clean_df.drop_duplicates(subset=['SMILES']).copy()

    if clean_df.empty:
        raise ValueError('No valid molecules remained after cleaning the uploaded training CSV.')

    if clean_df['Activity'].nunique() < 2:
        raise ValueError('Training data must contain both Activity classes 0 and 1.')

    return clean_df.reset_index(drop=True)


def clean_smiles_only_dataframe(df):
    smiles_col = resolve_case_insensitive_column(df.columns, 'SMILES')
    if smiles_col is None:
        raise ValueError('The CSV file must contain a "SMILES" column.')

    clean_df = df[[smiles_col]].copy()
    clean_df.columns = ['SMILES']
    clean_df = clean_df.dropna(subset=['SMILES']).copy()
    clean_df['SMILES'] = clean_df['SMILES'].astype(str).str.strip()
    clean_df['SMILES'] = clean_df['SMILES'].apply(lambda s: standardize_smiles_and_mol(s)[0])
    clean_df = clean_df.dropna(subset=['SMILES']).copy()
    clean_df = clean_df.drop_duplicates(subset=['SMILES']).copy()

    if clean_df.empty:
        raise ValueError('No valid molecules remained after cleaning the uploaded library CSV.')

    return clean_df.reset_index(drop=True)


def build_train_val_test_indices(smiles, labels, seed=RANDOM_SEED):
    outer_split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val_indices, test_indices = next(outer_split.split(smiles, labels))

    train_val_smiles = [smiles[i] for i in train_val_indices]
    train_val_labels = [labels[i] for i in train_val_indices]

    inner_split = StratifiedShuffleSplit(n_splits=1, test_size=0.17647058823529413, random_state=seed)
    train_rel_idx, val_rel_idx = next(inner_split.split(train_val_smiles, train_val_labels))

    train_indices = [train_val_indices[i] for i in train_rel_idx]
    val_indices = [train_val_indices[i] for i in val_rel_idx]

    return train_indices, val_indices, test_indices


# =========================
# VIRTUAL SCREENING HELPERS
# =========================

def train_and_evaluate_model(train_dataloader, val_dataloader, model, optimizer, criterion, scheduler):
    best_val_loss = float('inf')
    stop_counter = 0
    checkpoint_path = os.path.join(app.root_path, 'best_model.pth')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batched_graph, batched_features, batched_labels in train_dataloader:
            if batched_graph is None:
                continue
            batched_graph = batched_graph.to(DEVICE)
            batched_features = batched_features.to(DEVICE)
            batched_labels = batched_labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batched_graph, batched_features)
            loss = criterion(outputs, batched_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        train_loss /= max(train_batches, 1)
        history['train_loss'].append(float(train_loss))
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        val_batches = 0
        for batched_graph, batched_features, batched_labels in val_dataloader:
            if batched_graph is None:
                continue
            batched_graph = batched_graph.to(DEVICE)
            batched_features = batched_features.to(DEVICE)
            batched_labels = batched_labels.to(DEVICE)
            with torch.no_grad():
                outputs = model(batched_graph, batched_features)
                loss = criterion(outputs, batched_labels)
            val_loss += loss.item()
            val_batches += 1

        val_loss /= max(val_batches, 1)
        history['val_loss'].append(float(val_loss))
        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stop_counter += 1

        if stop_counter >= EARLY_STOPPING_PATIENCE:
            print('Early stopping triggered.')
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    return model, history




def allow_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}


def get_compound_name_from_pubchem(smiles_string):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles_string}/synonyms/JSON"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        name = data['InformationList']['Information'][0]['Synonym'][0]
        return name
    except Exception:
        return None


def get_screening_hits(plot_df, threshold=SCREENING_THRESHOLD, fallback_top_k=FALLBACK_TOP_K):
    working = add_priority_scores(plot_df.copy())
    hits_df = working[working['Probability'] >= threshold].copy()
    used_fallback = False
    if hits_df.empty:
        used_fallback = True
        hits_df = working.sort_values(['PriorityScore', 'Probability'], ascending=[False, False]).head(min(fallback_top_k, len(working))).copy()
    else:
        hits_df = diversify_hits_by_scaffold(hits_df, max_per_scaffold=3)
    hits_df = hits_df.sort_values(['PriorityScore', 'Probability'], ascending=[False, False]).reset_index(drop=True)
    return hits_df, used_fallback




def align_smiles_and_probabilities(valid_smiles, class_1_probs):
    total = min(len(valid_smiles), len(class_1_probs))
    return valid_smiles[:total], np.asarray(class_1_probs[:total])


def select_plot_subset(
    probability_df,
    max_points=MAX_PLOT_POINTS,
    top_hits_to_keep=TOP_HITS_TO_ALWAYS_PLOT,
    random_seed=PLOT_RANDOM_SEED
):
    if len(probability_df) <= max_points:
        return probability_df.reset_index(drop=True)

    sorted_df = probability_df.sort_values('Probability', ascending=False).reset_index(drop=True)
    top_n = min(top_hits_to_keep, max_points, len(sorted_df))
    top_df = sorted_df.head(top_n)
    remaining_df = sorted_df.iloc[top_n:]
    remaining_slots = max_points - len(top_df)

    if remaining_slots > 0 and len(remaining_df) > 0:
        sampled_df = remaining_df.sample(
            n=min(remaining_slots, len(remaining_df)),
            random_state=random_seed
        )
        subset_df = pd.concat([top_df, sampled_df], ignore_index=True)
    else:
        subset_df = top_df.copy()

    return subset_df.reset_index(drop=True)


# =========================
# DOWNLOAD ROUTES
# =========================

@app.route('/download', methods=['GET'])
def download():
    file_path = os.path.join(app.config['GENERATED_FILES_DIR'], 'generated_molecules.csv')
    return send_file(file_path, as_attachment=True)


@app.route('/download_molecules', methods=['GET'])
def download_molecules():
    file_path = os.path.join(app.config['GENERATED_FILES_DIR'], 'Molecules.csv')
    if not os.path.exists(file_path):
        return abort(404, description="Generated molecules file not found.")
    return send_file(file_path, as_attachment=True, download_name='Molecules.csv')


@app.route('/download/sdf_zip', defaults={'filename': 'compounds_sdf.zip'})
@app.route('/download/sdf_zip/<path:filename>')
def download_sdf_zip(filename):
    file_path = os.path.join(app.config['GENERATED_FILES_DIR'], filename)
    if not os.path.exists(file_path):
        return abort(404, description="SDF ZIP file not found.")
    return send_from_directory(app.config['GENERATED_FILES_DIR'], filename, as_attachment=True)


@app.route('/images/<path:filename>')
def uploaded_file(filename):
    static_path = os.path.join(app.root_path, 'static', filename)
    generated_path = os.path.join(app.root_path, app.config['GENERATED_FILES_DIR'], filename)

    if os.path.exists(static_path):
        return send_from_directory(os.path.join(app.root_path, 'static'), filename)
    if os.path.exists(generated_path):
        return send_from_directory(os.path.join(app.root_path, app.config['GENERATED_FILES_DIR']), filename)

    return abort(404, description="Requested file not found.")


@app.route('/downloads/<path:filename>')
def downloads(filename):
    directory = app.config['GENERATED_FILES_DIR']
    try:
        return send_from_directory(directory, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found.", 404


@app.route('/files/<path:filename>')
def uploa(filename):
    directory = current_app.root_path
    return send_from_directory(directory, filename)




@app.route('/generate', methods=['POST'])
def generate():
    try:
        set_seed(RANDOM_SEED)
        num_molecules = max(1, min(request.form.get('num_molecules', type=int) or 20, 500))
        option = (request.form.get('options') or 'lipinski').strip().lower()
        apply_lipinski = option != 'no_lipinski'

        generated_molecules = generate_denovo_library(num_molecules, apply_lipinski=apply_lipinski)
        if not generated_molecules:
            flash('De novo molecule generation did not produce any valid molecules.', 'error')
            return render_template('upload.html', active_tab='home')

        molecules_df = pd.DataFrame(generated_molecules, columns=['SMILES', 'Activity'])
        molecules_df.to_csv(os.path.join(app.config['GENERATED_FILES_DIR'], 'generated_molecules.csv'), index=False)
        molecules_df.to_csv(os.path.join(app.config['GENERATED_FILES_DIR'], 'Molecules.csv'), index=False)

        compounds_sdf_dir = os.path.join(app.config['GENERATED_FILES_DIR'], 'denovo_sdf')
        clear_workspace(compounds_sdf_dir)
        for idx, (smiles, activity) in enumerate(generated_molecules):
            sdf_path = os.path.join(compounds_sdf_dir, f'denovo_{idx+1}.sdf')
            write_3d_sdf_with_properties(smiles, sdf_path, properties={'Activity': activity}, mol_name=f'denovo_{idx+1}')

        sdf_zipfile_path = os.path.join(app.config['GENERATED_FILES_DIR'], 'compounds_sdf.zip')
        with zipfile.ZipFile(sdf_zipfile_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(compounds_sdf_dir):
                for sdf_file in files:
                    file_path = os.path.join(root, sdf_file)
                    zipf.write(file_path, arcname=os.path.relpath(file_path, compounds_sdf_dir))

        denovo_plot_name = f'denovo_drug_space_{int(time.time())}.png'
        plot_drug_space(pd.DataFrame({'Compound': molecules_df['SMILES']}), os.path.join('static', denovo_plot_name), title='De Novo Drug Space')

        return render_template('upload.html', generated_molecules=generated_molecules, generated_file_path=os.path.join(app.config['GENERATED_FILES_DIR'], 'Molecules.csv'), sdf_zip_file='compounds_sdf.zip', pca_plot_file_path=denovo_plot_name, active_tab='home')
    except Exception as e:
        flash(f'De novo molecule generation failed: {str(e)}', 'error')
        return render_template('upload.html', active_tab='home')


# =========================
# HOME / VIRTUAL SCREENING
# =========================

@app.route('/', methods=['GET', 'POST'])
def index():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    final_compounds_filename = f'final_compounds_{timestamp}.csv'
    final_clusters_filename = f'final_clusters_{timestamp}.csv'

    pca_plot_filename = f'cluster_plot_pca_{timestamp}.png'
    tsne_plot_filename = f'cluster_plot_tsne_{timestamp}.png'
    umap_plot_filename = f'cluster_plot_umap_{timestamp}.png'

    virtual_screening = False
    uploaded_file_path = None
    generated_file_path = None
    generated_molecules = None

    pca_plot_file_path = None
    tsne_plot_file_path = None
    umap_plot_file_path = None

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']

        if file.filename != '':
            filename = f'Molecules_{timestamp}.csv'
            uploaded_file_path = os.path.join(app.config['UPLOADED_FILES_DIR'], filename)
            file.save(uploaded_file_path)

            try:
                set_seed(RANDOM_SEED)

                raw_data = pd.read_csv(uploaded_file_path)
                data = clean_virtual_screening_training_dataframe(raw_data)
                smiles = data["SMILES"].tolist()
                labels = data["Activity"].astype(int).tolist()

                train_indices, val_indices, test_indices = build_train_val_test_indices(smiles, labels)

                full_dataset = MoleculeDataset(smiles, labels)
                train_dataset = [full_dataset[i] for i in train_indices]
                val_dataset = [full_dataset[i] for i in val_indices]
                test_dataset = [full_dataset[i] for i in test_indices]

                class_counts = np.bincount(np.array([labels[i] for i in train_indices]), minlength=2)
                class_weights = len(train_indices) / (2.0 * np.maximum(class_counts, 1))
                class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

                model = GNN(1, MODEL_HIDDEN_SIZE, 2).to(DEVICE)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=3
                )

                train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
                val_dataloader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate)
                test_dataloader = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate)

                model, training_history = train_and_evaluate_model(
                    train_dataloader, val_dataloader, model, optimizer, criterion, scheduler
                )

                all_predictions, all_targets = [], []
                all_probabilities = []
                valid_test_smiles = []

                for idx in test_indices:
                    canonical_smiles, mol = standardize_smiles_and_mol(smiles[idx])
                    if canonical_smiles is None or mol is None:
                        continue
                    sample = full_dataset[idx]
                    if sample[0] is not None:
                        valid_test_smiles.append(canonical_smiles)

                model.eval()
                for batched_graph, batched_features, batched_labels in test_dataloader:
                    if batched_graph is None:
                        continue

                    batched_graph = batched_graph.to(DEVICE)
                    batched_features = batched_features.to(DEVICE)
                    batched_labels = batched_labels.to(DEVICE)

                    with torch.no_grad():
                        outputs = model(batched_graph, batched_features)
                        probabilities = torch.softmax(outputs, dim=1)

                    _, predicted = torch.max(outputs, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(batched_labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

                all_probabilities = np.array(all_probabilities)
                true_labels = np.array(all_targets)

                if len(all_probabilities) == 0:
                    flash('No valid molecules remained after filtering.', 'warning')
                    return render_template('upload.html', active_tab='virtual_screening')

                class_1_probs = all_probabilities[:, 1]
                valid_test_smiles, class_1_probs, true_labels, all_predictions = align_screening_arrays(valid_test_smiles, class_1_probs, true_labels, np.asarray(all_predictions))

                accuracy = accuracy_score(true_labels, all_predictions) if len(true_labels) else float('nan')
                precision = precision_score(true_labels, all_predictions, zero_division=0) if len(true_labels) else float('nan')
                recall = recall_score(true_labels, all_predictions, zero_division=0) if len(true_labels) else float('nan')
                f1 = f1_score(true_labels, all_predictions, zero_division=0) if len(true_labels) else float('nan')

                print(f"Test Accuracy: {accuracy * 100:.2f}%")
                print(f"Precision: {precision * 100:.2f}%")
                print(f"Recall: {recall * 100:.2f}%")
                print(f"F1 Score: {f1 * 100:.2f}%")

                if len(np.unique(true_labels)) > 1 and len(class_1_probs) == len(true_labels):
                    auc = roc_auc_score(true_labels, class_1_probs)
                    print(f"AUC: {auc * 100:.2f}%")
                else:
                    auc = float('nan')
                    print("AUC: N/A (only one class present in evaluation labels)")

                full_probability_df = pd.DataFrame({
                    'Compound': valid_test_smiles,
                    'Probability': class_1_probs
                })
                full_probability_df = add_priority_scores(full_probability_df)

                final_df, used_fallback = get_screening_hits(full_probability_df)
                plot_input_df = select_plot_subset(full_probability_df)

                plot_smiles = plot_input_df['Compound'].tolist()
                plot_dataset = MoleculeDataset(plot_smiles, [0] * len(plot_smiles))
                plot_dataloader = DataLoader(plot_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate)

                all_embeddings = []
                for batched_graph, batched_features, _ in plot_dataloader:
                    if batched_graph is None:
                        continue
                    batched_graph = batched_graph.to(DEVICE)
                    batched_features = batched_features.to(DEVICE)
                    with torch.no_grad():
                        embeddings = model.get_features(batched_graph, batched_features)
                    all_embeddings.extend(embeddings.cpu().numpy())

                X = np.array(all_embeddings)

                if len(X) < 2:
                    flash('Not enough valid clustering data available.', 'warning')
                    return render_template('upload.html', active_tab='virtual_screening')

                n_clusters = 2 if len(X) >= 10 else 1

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                if n_clusters == 1:
                    all_clusters = np.zeros(len(X_scaled), dtype=int)
                    gmm = None
                else:
                    gmm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_SEED)
                    all_clusters = gmm.fit_predict(X_scaled)

                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)

                tsne_perplexity = min(30, max(2, len(X_scaled) - 1))
                tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=RANDOM_SEED, init='pca', learning_rate='auto')
                X_tsne = tsne.fit_transform(X_scaled)

                umap_reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=min(15, max(2, len(X_scaled) - 1)),
                    min_dist=0.3,
                    random_state=RANDOM_SEED
                )
                X_umap = umap_reducer.fit_transform(X_scaled)

                plot_df = pd.DataFrame({
                    'Compound': plot_smiles,
                    'Probability': plot_input_df['Probability'].values,
                    'Cluster': all_clusters,
                    'PCA1': X_pca[:, 0],
                    'PCA2': X_pca[:, 1],
                    'TSNE1': X_tsne[:, 0],
                    'TSNE2': X_tsne[:, 1],
                    'UMAP1': X_umap[:, 0],
                    'UMAP2': X_umap[:, 1]
                })
                plot_df = add_priority_scores(plot_df)

                if used_fallback:
                    flash(
                        f'No compounds passed the probability threshold ({SCREENING_THRESHOLD:.2f}). '
                        f'Showing top {len(final_df)} highest-scoring compounds instead.',
                        'warning'
                    )

                if len(full_probability_df) > len(plot_df):
                    flash(
                        f'Graphs were generated using {len(plot_df)} molecules for speed, while screening was completed on all {len(full_probability_df)} valid molecules.',
                        'warning'
                    )

                generated_file_path = os.path.join(app.config['GENERATED_FILES_DIR'], final_compounds_filename)
                final_df.to_csv(generated_file_path, index=False)

                if n_clusters > 1 and len(X_scaled) > n_clusters:
                    silhouette_avg = silhouette_score(X_scaled, plot_df['Cluster'])
                    davies_bouldin = davies_bouldin_score(X_scaled, plot_df['Cluster'])
                    print(f"Silhouette Score: {silhouette_avg:.4f}")
                    print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")

                final_clusters_file_path = os.path.join(app.config['GENERATED_FILES_DIR'], final_clusters_filename)
                final_df_with_plot_info = final_df.merge(
                    plot_df[['Compound', 'Cluster', 'PCA1', 'PCA2', 'TSNE1', 'TSNE2', 'UMAP1', 'UMAP2']],
                    on='Compound',
                    how='left'
                )
                final_df_with_plot_info.to_csv(final_clusters_file_path, index=False)

                top_hits_for_plot = plot_df[plot_df['Compound'].isin(final_df['Compound'])].copy()

                def save_cluster_plot(df_plot, top_hits_df, x_col, y_col, title, out_path, x_label, y_label):
                    clusters = df_plot['Cluster'].values

                    plt.figure(figsize=(12, 8))
                    for cluster in sorted(np.unique(clusters)):
                        cluster_points = df_plot[df_plot['Cluster'] == cluster]
                        if len(cluster_points) == 0:
                            continue
                        plt.scatter(
                            cluster_points[x_col],
                            cluster_points[y_col],
                            alpha=0.75,
                            label=f"Cluster {cluster}"
                        )

                    if not top_hits_df.empty:
                        plt.scatter(
                            top_hits_df[x_col],
                            top_hits_df[y_col],
                            s=120,
                            facecolors='none',
                            edgecolors='black',
                            linewidths=1.5,
                            label='Top Hits'
                        )

                    if gmm is not None and x_col == 'PCA1' and y_col == 'PCA2':
                        centroids_2d = pca.transform(gmm.means_)
                        plt.scatter(
                            centroids_2d[:, 0],
                            centroids_2d[:, 1],
                            c='red',
                            marker='X',
                            s=220,
                            label='Centroids'
                        )

                    plt.xlabel(x_label)
                    plt.ylabel(y_label)
                    plt.title(title)
                    plt.grid(True, alpha=0.25)
                    plt.legend()
                    plt.savefig(out_path, bbox_inches='tight', dpi=300)
                    plt.close()

                pca_plot_file_path = os.path.join('static', pca_plot_filename)
                tsne_plot_file_path = os.path.join('static', tsne_plot_filename)
                umap_plot_file_path = os.path.join('static', umap_plot_filename)

                save_cluster_plot(
                    plot_df,
                    top_hits_for_plot,
                    'PCA1',
                    'PCA2',
                    f'Molecular Embedding PCA Plot ({pca.explained_variance_ratio_[0] * 100:.1f}% / {pca.explained_variance_ratio_[1] * 100:.1f}% variance)',
                    pca_plot_file_path,
                    f'PCA 1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)',
                    f'PCA 2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)'
                )

                save_cluster_plot(
                    plot_df,
                    top_hits_for_plot,
                    'TSNE1',
                    'TSNE2',
                    'Molecular Embedding t-SNE Plot',
                    tsne_plot_file_path,
                    't-SNE 1',
                    't-SNE 2'
                )

                save_cluster_plot(
                    plot_df,
                    top_hits_for_plot,
                    'UMAP1',
                    'UMAP2',
                    'Molecular Embedding UMAP Plot',
                    umap_plot_file_path,
                    'UMAP 1',
                    'UMAP 2'
                )

                training_curve_filename = f'training_curve_{timestamp}.png'
                prob_hist_filename = f'probability_hist_{timestamp}.png'
                top_hits_filename = f'top_hits_{timestamp}.png'
                drug_space_filename = f'drug_space_{timestamp}.png'
                cluster_dist_filename = f'cluster_dist_{timestamp}.png'
                roc_curve_filename = f'roc_curve_{timestamp}.png'
                pr_curve_filename = f'pr_curve_{timestamp}.png'
                confusion_filename = f'confusion_matrix_{timestamp}.png'

                plot_training_history(training_history, os.path.join('static', training_curve_filename), title='Virtual Screening Training History')
                plot_probability_distribution(full_probability_df, os.path.join('static', prob_hist_filename))
                plot_top_hits_bar(final_df, os.path.join('static', top_hits_filename), score_col='PriorityScore', title='Top Ranked Virtual Screening Hits')
                plot_drug_space(final_df[['Compound']], os.path.join('static', drug_space_filename), title='Virtual Screening Drug Space')
                plot_cluster_distribution(plot_df, os.path.join('static', cluster_dist_filename))
                plot_roc_curve_image(true_labels, class_1_probs, os.path.join('static', roc_curve_filename), title='Virtual Screening ROC Curve')
                plot_pr_curve_image(true_labels, class_1_probs, os.path.join('static', pr_curve_filename), title='Virtual Screening PR Curve')
                plot_confusion_matrix_image(true_labels, all_predictions, os.path.join('static', confusion_filename), title='Virtual Screening Confusion Matrix')

                metrics_rows = [
                    ('Accuracy', f'{accuracy:.4f}' if not np.isnan(accuracy) else 'NA'),
                    ('Precision', f'{precision:.4f}' if not np.isnan(precision) else 'NA'),
                    ('Recall', f'{recall:.4f}' if not np.isnan(recall) else 'NA'),
                    ('F1', f'{f1:.4f}' if not np.isnan(f1) else 'NA'),
                    ('AUC', f'{auc:.4f}' if not np.isnan(auc) else 'NA'),
                    ('MCC', f'{matthews_corrcoef(true_labels, all_predictions):.4f}' if len(true_labels) and len(np.unique(true_labels)) > 1 else 'NA'),
                    ('Valid Molecules', str(len(full_probability_df))),
                    ('Selected Hits', str(len(final_df))),
                    ('Unique Scaffolds', str(final_df['Scaffold'].nunique()) if 'Scaffold' in final_df.columns else 'NA')
                ]
                dashboard_html = build_dashboard_html(metrics_rows, [
                    ('Training Curve', training_curve_filename),
                    ('Probability Histogram', prob_hist_filename),
                    ('Top Hits', top_hits_filename),
                    ('Drug Space', drug_space_filename),
                    ('Cluster Distribution', cluster_dist_filename),
                    ('ROC Curve', roc_curve_filename if os.path.exists(os.path.join('static', roc_curve_filename)) else None),
                    ('PR Curve', pr_curve_filename if os.path.exists(os.path.join('static', pr_curve_filename)) else None),
                    ('Confusion Matrix', confusion_filename if os.path.exists(os.path.join('static', confusion_filename)) else None)
                ], heading='Virtual Screening Dashboard')

                compounds_df = pd.read_csv(generated_file_path)
                clusters_df = pd.read_csv(final_clusters_file_path)

                compounds_table = dashboard_html + compounds_df.to_html(classes='table table-striped table-bordered', index=False)
                clusters_table = clusters_df.to_html(classes='table table-striped table-bordered', index=False)

                virtual_screening = True
                return render_template(
                    'upload.html',
                    virtual_screening=virtual_screening,
                    final_clusters_filename=final_clusters_filename,
                    final_compounds_filename=final_compounds_filename,
                    compounds_table=compounds_table,
                    clusters_table=clusters_table,
                    pca_plot_file_path=pca_plot_file_path[len('static/'):],
                    tsne_plot_file_path=tsne_plot_file_path[len('static/'):],
                    umap_plot_file_path=umap_plot_file_path[len('static/'):],
                    generated_file_path=generated_file_path,
                    final_clusters_file_path=final_clusters_file_path,
                    active_tab='virtual_screening',
                    random=int(time.time())
                )
            except Exception as e:
                flash(f'Virtual screening failed: {str(e)}', 'error')
                return render_template('upload.html', active_tab='virtual_screening')

    return render_template(
        'upload.html',
        virtual_screening=virtual_screening,
        uploaded_file_path=uploaded_file_path,
        generated_file_path=generated_file_path,
        generated_molecules=generated_molecules,
        active_tab='home'
    )


@app.route('/rescreening', methods=['POST'])
def rescreening():
    if 'file' not in request.files:
        return render_template('upload.html', active_tab='virtual_screening')

    file = request.files['file']
    if not file or not allow_files(file.filename):
        return render_template('upload.html', active_tab='virtual_screening')

    try:
        filename = 'New_Library.csv'
        uploaded_file_path = os.path.join(app.config['UPLOADED_FILES_DIR'], filename)
        file.save(uploaded_file_path)

        set_seed(RANDOM_SEED)

        model = GNN(1, MODEL_HIDDEN_SIZE, 2).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(app.root_path, "best_model.pth"), map_location=DEVICE))
        model.eval()

        raw_new_data = pd.read_csv(uploaded_file_path)
        new_data = clean_smiles_only_dataframe(raw_new_data)
        new_smiles = new_data["SMILES"].tolist()
        new_dataset = MoleculeDataset(new_smiles, [0] * len(new_smiles))
        new_dataloader = DataLoader(new_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate)

        all_probabilities = []
        valid_new_smiles = []

        for i in range(len(new_smiles)):
            sample = new_dataset[i]
            if sample[0] is not None:
                valid_new_smiles.append(new_smiles[i])

        for batched_graph, batched_features, _ in new_dataloader:
            if batched_graph is None:
                continue
            batched_graph = batched_graph.to(DEVICE)
            batched_features = batched_features.to(DEVICE)
            with torch.no_grad():
                outputs = model(batched_graph, batched_features)
                probabilities = torch.softmax(outputs, dim=1)
            all_probabilities.extend(probabilities.cpu().numpy())

        all_probabilities = np.array(all_probabilities)

        if len(all_probabilities) == 0:
            flash('No valid molecules remained after filtering in re-screening.', 'warning')
            return render_template('upload.html', active_tab='virtual_screening')

        class_1_probs = all_probabilities[:, 1]
        valid_new_smiles, class_1_probs = align_smiles_and_probabilities(valid_new_smiles, class_1_probs)

        full_probability_df = pd.DataFrame({
            'Compound': valid_new_smiles,
            'Probability': class_1_probs
        })
        full_probability_df = add_priority_scores(full_probability_df)

        final_df, used_fallback = get_screening_hits(full_probability_df)
        plot_input_df = select_plot_subset(full_probability_df)

        plot_smiles = plot_input_df['Compound'].tolist()
        plot_dataset = MoleculeDataset(plot_smiles, [0] * len(plot_smiles))
        plot_dataloader = DataLoader(plot_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate)

        all_embeddings = []
        for batched_graph, batched_features, _ in plot_dataloader:
            if batched_graph is None:
                continue
            batched_graph = batched_graph.to(DEVICE)
            batched_features = batched_features.to(DEVICE)
            with torch.no_grad():
                embeddings = model.get_features(batched_graph, batched_features)
            all_embeddings.extend(embeddings.cpu().numpy())

        X = np.array(all_embeddings)

        if len(X) < 2:
            flash('Not enough valid re-screening compounds for clustering.', 'warning')
            return render_template('upload.html', active_tab='virtual_screening')

        n_clusters = 2 if len(X) >= 10 else 1

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if n_clusters == 1:
            all_clusters = np.zeros(len(X_scaled), dtype=int)
            gmm = None
        else:
            gmm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_SEED)
            all_clusters = gmm.fit_predict(X_scaled)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        tsne_perplexity = min(30, max(2, len(X_scaled) - 1))
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=RANDOM_SEED, init='pca', learning_rate='auto')
        X_tsne = tsne.fit_transform(X_scaled)

        umap_reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, max(2, len(X_scaled) - 1)),
            min_dist=0.3,
            random_state=RANDOM_SEED
        )
        X_umap = umap_reducer.fit_transform(X_scaled)

        plot_df = pd.DataFrame({
            'Compound': plot_smiles,
            'Probability': plot_input_df['Probability'].values,
            'Cluster': all_clusters,
            'PCA1': X_pca[:, 0],
            'PCA2': X_pca[:, 1],
            'TSNE1': X_tsne[:, 0],
            'TSNE2': X_tsne[:, 1],
            'UMAP1': X_umap[:, 0],
            'UMAP2': X_umap[:, 1]
        })
        plot_df = add_priority_scores(plot_df)

        if used_fallback:
            flash(
                f'No compounds passed the re-screening threshold ({SCREENING_THRESHOLD:.2f}). '
                f'Showing top {len(final_df)} highest-scoring compounds instead.',
                'warning'
            )

        if len(full_probability_df) > len(plot_df):
            flash(
                f'Graphs were generated using {len(plot_df)} molecules for speed, while re-screening was completed on all {len(full_probability_df)} valid molecules.',
                'warning'
            )

        generated_file_path = os.path.join(app.config['GENERATED_FILES_DIR'], 'new_library_predictions.csv')
        final_df.to_csv(generated_file_path, index=False)

        final_clusters_file_path = os.path.join(app.config['GENERATED_FILES_DIR'], 'new_library_clusters.csv')
        final_df_with_plot_info = final_df.merge(
            plot_df[['Compound', 'Cluster', 'PCA1', 'PCA2', 'TSNE1', 'TSNE2', 'UMAP1', 'UMAP2']],
            on='Compound',
            how='left'
        )
        final_df_with_plot_info.to_csv(final_clusters_file_path, index=False)

        top_hits_for_plot = plot_df[plot_df['Compound'].isin(final_df['Compound'])].copy()

        compounds_sdf_dir = os.path.join(app.config['GENERATED_FILES_DIR'], 'compounds_sdf')
        os.makedirs(compounds_sdf_dir, exist_ok=True)

        for index, row in final_df_with_plot_info.iterrows():
            compound_name = get_compound_name_from_pubchem(row['Compound']) or f"Compound_{index}"
            cluster_value = row['Cluster'] if pd.notna(row['Cluster']) else -1
            safe_filename = secure_filename(compound_name)
            sdf_filename = f"{safe_filename}.sdf" if safe_filename else f"Compound_{index}.sdf"
            sdf_path = os.path.join(compounds_sdf_dir, sdf_filename)
            write_3d_sdf_with_properties(row['Compound'], sdf_path, properties={'Probability': row.get('Probability', ''), 'PriorityScore': row.get('PriorityScore', ''), 'Cluster': cluster_value}, mol_name=compound_name)

        sdf_zipfile_path = os.path.join(app.config['GENERATED_FILES_DIR'], 'compounds_sdf.zip')
        with zipfile.ZipFile(sdf_zipfile_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(compounds_sdf_dir):
                for sdf_file in files:
                    file_path = os.path.join(root, sdf_file)
                    zipf.write(file_path, arcname=os.path.relpath(file_path, compounds_sdf_dir))

        shutil.rmtree(compounds_sdf_dir)

        pca_plot_filename = 'new_library_pca_plot.png'
        tsne_plot_filename = 'new_library_tsne_plot.png'
        umap_plot_filename = 'new_library_umap_plot.png'

        def save_cluster_plot(df_plot, top_hits_df, x_col, y_col, title, out_path, x_label, y_label):
            clusters = df_plot['Cluster'].values

            plt.figure(figsize=(12, 8))
            for cluster in sorted(np.unique(clusters)):
                cluster_points = df_plot[df_plot['Cluster'] == cluster]
                if len(cluster_points) == 0:
                    continue
                plt.scatter(
                    cluster_points[x_col],
                    cluster_points[y_col],
                    alpha=0.75,
                    label=f"Cluster {cluster}"
                )

            if not top_hits_df.empty:
                plt.scatter(
                    top_hits_df[x_col],
                    top_hits_df[y_col],
                    s=120,
                    facecolors='none',
                    edgecolors='black',
                    linewidths=1.5,
                    label='Top Hits'
                )

            if gmm is not None and x_col == 'PCA1' and y_col == 'PCA2':
                centroids_2d = pca.transform(gmm.means_)
                plt.scatter(
                    centroids_2d[:, 0],
                    centroids_2d[:, 1],
                    c='red',
                    marker='X',
                    s=220,
                    label='Centroids'
                )

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.savefig(out_path, bbox_inches='tight', dpi=300)
            plt.close()

        pca_plot_path = os.path.join(app.config['GENERATED_FILES_DIR'], pca_plot_filename)
        tsne_plot_path = os.path.join(app.config['GENERATED_FILES_DIR'], tsne_plot_filename)
        umap_plot_path = os.path.join(app.config['GENERATED_FILES_DIR'], umap_plot_filename)

        save_cluster_plot(
            plot_df,
            top_hits_for_plot,
            'PCA1',
            'PCA2',
            f'Re-screened Molecular PCA Plot ({pca.explained_variance_ratio_[0] * 100:.1f}% / {pca.explained_variance_ratio_[1] * 100:.1f}% variance)',
            pca_plot_path,
            f'PCA 1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)',
            f'PCA 2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)'
        )

        save_cluster_plot(
            plot_df,
            top_hits_for_plot,
            'TSNE1',
            'TSNE2',
            'Re-screened Molecular t-SNE Plot',
            tsne_plot_path,
            't-SNE 1',
            't-SNE 2'
        )

        save_cluster_plot(
            plot_df,
            top_hits_for_plot,
            'UMAP1',
            'UMAP2',
            'Re-screened Molecular UMAP Plot',
            umap_plot_path,
            'UMAP 1',
            'UMAP 2'
        )

        prob_hist_filename = 'new_library_probability_hist.png'
        top_hits_filename = 'new_library_top_hits.png'
        drug_space_filename = 'new_library_drug_space.png'
        cluster_dist_filename = 'new_library_cluster_dist.png'
        plot_probability_distribution(full_probability_df, os.path.join('static', prob_hist_filename), title='Re-screening Probability Distribution')
        plot_top_hits_bar(final_df, os.path.join('static', top_hits_filename), score_col='PriorityScore', title='Top Re-screened Hits')
        plot_drug_space(final_df[['Compound']], os.path.join('static', drug_space_filename), title='Re-screening Drug Space')
        plot_cluster_distribution(plot_df, os.path.join('static', cluster_dist_filename), title='Re-screening Cluster Distribution')
        metrics_rows = [('Re-screened valid molecules', str(len(full_probability_df))), ('Selected hits', str(len(final_df))), ('Unique scaffolds', str(final_df['Scaffold'].nunique()) if 'Scaffold' in final_df.columns else 'NA'), ('Mean probability', f"{full_probability_df['Probability'].mean():.4f}"), ('Max probability', f"{full_probability_df['Probability'].max():.4f}")]
        dashboard_html = build_dashboard_html(metrics_rows, [('Probability Histogram', prob_hist_filename), ('Top Hits', top_hits_filename), ('Drug Space', drug_space_filename), ('Cluster Distribution', cluster_dist_filename)], heading='Re-screening Dashboard')

        compound_table = dashboard_html + final_df.to_html(classes='table table-striped table-bordered', index=False)
        cluster_table = final_df_with_plot_info.to_html(classes='table table-striped table-bordered', index=False)

        return render_template(
            'upload.html',
            success=True,
            compound_table=compound_table,
            cluster_table=cluster_table,
            pca_plot_file_path=pca_plot_filename,
            tsne_plot_file_path=tsne_plot_filename,
            umap_plot_file_path=umap_plot_filename,
            sdf_zip_file='compounds_sdf.zip',
            active_tab='virtual_screening',
            random=int(time.time())
        )
    except Exception as e:
        flash(f'Re-screening failed: {str(e)}', 'error')
        return render_template('upload.html', active_tab='virtual_screening')

# =========================
# PROTEIN REFINEMENT
# =========================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdb'}


def perform_protein_refinement(protein_file_path):
    timestamp = int(time.time())
    stripped_pdb_filename = f'protein_stripped_{timestamp}.pdb'
    fixed_pdb_filename = f'fixed_output_{timestamp}.pdb'
    minimized_pdb_filename = f'minimized_protein_{timestamp}.pdb'
    ramachandran_plot_filename = f'ramachandran_plot_{timestamp}.png'
    sasa_per_residue_plot_filename = f'sasa_per_residue_plot_{timestamp}.png'

    logger.debug(f"Starting protein refinement for: {protein_file_path}")

    traj = md.load(protein_file_path)
    protein = traj.topology.select('protein')
    stripped_traj = traj.atom_slice(protein)
    stripped_traj.save(stripped_pdb_filename)

    fixer = PDBFixer(stripped_pdb_filename)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    with open(fixed_pdb_filename, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    pdb = PDBFile(fixed_pdb_filename)
    modeller = Modeller(pdb.topology, pdb.positions)
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    modeller.addHydrogens(forcefield)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1 * nanometer,
        constraints=HBonds
    )
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.004 * picoseconds)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=500)

    with open(minimized_pdb_filename, 'w') as f:
        state = simulation.context.getState(getPositions=True)
        PDBFile.writeFile(modeller.topology, state.getPositions(), f)

    traj = md.load(minimized_pdb_filename)
    phi = md.compute_phi(traj)
    psi = md.compute_psi(traj)
    phi_angles = np.rad2deg(md.compute_dihedrals(traj, phi[0]))
    psi_angles = np.rad2deg(md.compute_dihedrals(traj, psi[0]))

    plt.figure(figsize=(8, 6))
    plt.scatter(phi_angles, psi_angles, s=2, c='blue', alpha=0.5)
    plt.fill_betweenx(np.arange(-180, 50, 1), -100, -45, color='orange', alpha=0.25)
    plt.fill_betweenx(np.arange(-100, 180, 1), 45, 100, color='orange', alpha=0.25)
    plt.fill_between(np.arange(-180, 180, 1), 135, 180, color='green', alpha=0.25)
    plt.fill_between(np.arange(-180, 180, 1), -180, -135, color='green', alpha=0.25)
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel('Phi (φ) angles (degrees)')
    plt.ylabel('Psi (ψ) angles (degrees)')
    plt.title('Ramachandran Plot with Highlighted Secondary Structure Regions')
    plt.grid(True)
    plt.text(-75, 150, 'β-sheet', ha='center', va='center', color='green', alpha=0.75)
    plt.text(-60, -60, 'α-helix', ha='center', va='center', color='orange', alpha=0.75)
    plt.text(60, 60, 'α-helix', ha='center', va='center', color='orange', alpha=0.75)
    plt.text(100, -160, 'β-sheet', ha='center', va='center', color='green', alpha=0.75)
    plt.savefig(os.path.join('static', ramachandran_plot_filename))
    plt.close()

    sasa = md.shrake_rupley(traj, mode='residue')
    plt.plot(np.mean(sasa, axis=0))
    plt.title('Average Solvent Accessible Surface Area (SASA) per residue')
    plt.xlabel('Residue')
    plt.ylabel('SASA (nm²)')
    plt.savefig(os.path.join('static', sasa_per_residue_plot_filename))
    plt.close()

    return {
        'stripped_pdb': stripped_pdb_filename,
        'fixed_pdb': fixed_pdb_filename,
        'minimized_pdb': minimized_pdb_filename,
        'ramachandran_plot': f'static/{ramachandran_plot_filename}',
        'sasa_per_residue_plot': f'static/{sasa_per_residue_plot_filename}'
    }


@app.route('/protein_refinement', methods=['GET', 'POST'])
def protein_refinement():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part', 'error')
                return redirect(request.url)

            file = request.files['file']
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                protein_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(protein_file_path)

                result_files = perform_protein_refinement(protein_file_path)

                download_links = {
                    'stripped_protein': url_for('uploa', filename=result_files['stripped_pdb']),
                    'fixed_protein': url_for('uploa', filename=result_files['fixed_pdb']),
                    'minimized_protein': url_for('uploa', filename=result_files['minimized_pdb']),
                    'ramachandran_plot': url_for('static', filename=os.path.basename(result_files['ramachandran_plot'])),
                    'sasa_per_residue_plot': url_for('static', filename=os.path.basename(result_files['sasa_per_residue_plot']))
                }

                return render_template(
                    'upload.html',
                    download_links=download_links,
                    random=int(time.time()),
                    active_tab='protein_refinement'
                )
    except Exception as e:
        app.logger.error(f"An error occurred during protein refinement: {str(e)}")
        flash('An error occurred during protein refinement.', 'error')
        return redirect(request.url)

    return render_template('upload.html', active_tab='protein_refinement')


# =========================
# MOLECULAR DOCKING
# =========================

def allowed_fil(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'zip', 'pdb'}


def convert_sdf_to_pdbqt(sdf_path, output_directory):
    python_exe = resolve_python_executable(sys.executable)
    converted = []
    for root, dirs, files in os.walk(output_directory):
        for file in files:
            if file.lower().endswith('.sdf'):
                sdf_file = os.path.join(root, file)
                pdbqt_filename = os.path.splitext(file)[0] + '.pdbqt'
                pdbqt_path = os.path.join(root, pdbqt_filename)
                cmd = [python_exe, '-m', 'meeko.cli.mk_prepare_ligand', '-i', sdf_file, '-o', pdbqt_path]
                result = run_command_capture(cmd, timeout=1800, check=False)
                if result.returncode == 0 and os.path.exists(pdbqt_path):
                    converted.append(pdbqt_path)
                else:
                    print(f'An error occurred while converting {file}: {result.stderr}')
    return converted




def convert_protein(protein_pdb_path, protein_pdbqt_path):
    python_exe = resolve_python_executable(sys.executable)
    base = os.path.splitext(protein_pdbqt_path)[0]
    cmd = [python_exe, '-m', 'meeko.cli.mk_prepare_receptor', '--read_pdb', protein_pdb_path, '-o', base, '-p']
    result = run_command_capture(cmd, timeout=1800, check=False)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    if os.path.exists(protein_pdbqt_path):
        return protein_pdbqt_path
    for suffix in ['.pdbqt', '_rigid.pdbqt']:
        alt_path = base + suffix
        if os.path.exists(alt_path):
            return alt_path
    raise RuntimeError(f'Protein conversion failed for {protein_pdb_path}.')




def run_docking(protein_pdbqt, ligand_directory_path, results_directory_path):
    print('Starting the docking process...')
    os.makedirs(results_directory_path, exist_ok=True)
    vina_executable = resolve_vina_executable()

    center_x = request.form.get('center_x', type=float)
    center_y = request.form.get('center_y', type=float)
    center_z = request.form.get('center_z', type=float)
    size_x = request.form.get('size_x', type=float)
    size_y = request.form.get('size_y', type=float)
    size_z = request.form.get('size_z', type=float)
    exhaustiveness = request.form.get('exhaustiveness', type=int) or 8
    num_modes = request.form.get('num_modes', type=int) or 9
    energy_range = request.form.get('energy_range', type=int) or 3

    docking_data = []
    for ligand_file in Path(ligand_directory_path).glob('*.pdbqt'):
        ligand_pdbqt = str(ligand_file)
        result_file_path = os.path.join(results_directory_path, ligand_file.stem + '_docked.pdbqt')
        config_text = f"""receptor = {protein_pdbqt}
ligand = {ligand_pdbqt}

center_x = {center_x}
center_y = {center_y}
center_z = {center_z}
size_x = {size_x}
size_y = {size_y}
size_z = {size_z}

out = {result_file_path}
exhaustiveness = {exhaustiveness}
num_modes = {num_modes}
energy_range = {energy_range}
"""
        config_file_path = os.path.join(results_directory_path, ligand_file.stem + '_config.txt')
        with open(config_file_path, 'w', encoding='utf-8') as config_file:
            config_file.write(config_text)
        try:
            result = run_command_capture([vina_executable, '--config', config_file_path], timeout=7200, check=False)
            if result.returncode != 0:
                print(f'Error in docking: {result.stderr}')
            else:
                print(f'Docking completed for {ligand_file.stem}. Output:\n{result.stdout}')
        finally:
            if os.path.exists(config_file_path):
                os.remove(config_file_path)
        poses = parse_vina_results_from_pdbqt(result_file_path)
        for pose in poses:
            docking_data.append({'file_name': os.path.basename(result_file_path), 'ligand_name': ligand_file.stem, 'binding_affinity': pose['binding_affinity'], 'rmsd_lb': pose['rmsd_lb'], 'rmsd_ub': pose['rmsd_ub'], 'pose_rank': pose['pose_rank']})

    if docking_data:
        df = pd.DataFrame(docking_data).sort_values(['ligand_name', 'binding_affinity', 'pose_rank'], ascending=[True, True, True])
        df_best_poses = df.groupby('ligand_name', as_index=False).first().copy()
        df_best_poses['final_rmsd'] = (df_best_poses['rmsd_ub'] - df_best_poses['rmsd_lb']).round(4)
        csv_file_path = os.path.join(results_directory_path, 'docking_results.csv')
        df_best_poses.to_csv(csv_file_path, index=False)
        generate_docking_summary_plots(csv_file_path, results_directory_path, prefix='focused_docking')
        return df_best_poses
    print('No docking data to process.')
    return pd.DataFrame()




@app.route('/upload', methods=['POST'])
def upload_files():
    job_id = uuid.uuid4().hex
    job_workspace = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    job_results_dir = os.path.join(app.config['DOCKING_RESULTS_DIR'], job_id)

    clear_workspace(job_workspace)
    clear_workspace(job_results_dir)

    protein_file = request.files.get('protein_file')
    ligand_zip = request.files.get('ligand_zip')

    if protein_file and allowed_fil(protein_file.filename) and ligand_zip and allowed_fil(ligand_zip.filename):
        protein_filename = secure_filename(protein_file.filename)
        ligand_zip_filename = secure_filename(ligand_zip.filename)

        protein_file_path = os.path.join(job_workspace, protein_filename)
        ligand_zip_path = os.path.join(job_workspace, ligand_zip_filename)

        protein_file.save(protein_file_path)
        ligand_zip.save(ligand_zip_path)

        output_directory_path = os.path.join(job_workspace, 'refined_ligands')
        Path(output_directory_path).mkdir(parents=True, exist_ok=True)

        safe_extract_zip(ligand_zip_path, output_directory_path)

        convert_sdf_to_pdbqt(sdf_path=ligand_zip_path, output_directory=output_directory_path)

        protein_pdbqt_path = protein_file_path.replace('.pdb', '.pdbqt')
        protein_pdbqt_path = convert_protein(protein_file_path, protein_pdbqt_path)
        run_docking(protein_pdbqt_path, output_directory_path, job_results_dir)

        return jsonify({'job_id': job_id, 'message': 'Files uploaded, conversion started, and docking initiated!'})

    return jsonify({'error': 'Invalid file type or missing files.'}), 400


@app.route('/docking', methods=['GET'])
def docking():
    protein_file_path = request.args.get('protein_file_path', type=str)
    protein_pdbqt_path = os.path.join(app.config['UPLOADED_FILES_DIR'], protein_file_path)
    ligand_directory_path = os.path.join(app.config['GENERATED_FILES_DIR'], 'refined_ligands')
    results_directory_path = os.path.join(app.config['DOCKING_RESULTS_DIR'])

    run_docking(protein_pdbqt_path, ligand_directory_path, results_directory_path)
    return jsonify({'message': 'Docking completed!'})


@app.route('/list_docking_results')
def list_docking_results():
    results_files = Path(app.config['DOCKING_RESULTS_DIR']).glob('*_docked.pdbqt')
    results_list = [str(result) for result in results_files if result.is_file() and result.stat().st_size > 0]
    return jsonify(results_list)


@app.route('/results/<path:filename>')
def download_results(filename):
    results_directory_path = os.path.join(app.config['DOCKING_RESULTS_DIR'])
    return send_from_directory(directory=results_directory_path, path=filename, as_attachment=True)


@app.route('/analyze_results/<job_id>', methods=['GET'])
def analyze_results(job_id):
    results_directory = os.path.join(app.config['DOCKING_RESULTS_DIR'], job_id)
    filepath = os.path.join(results_directory, 'docking_results.csv')

    if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
        return send_file(filepath, as_attachment=True)
    return jsonify({'message': 'Results not ready'}), 202


@app.route('/chart_data/<job_id>')
def chart_data(job_id):
    job_results_dir = os.path.join(app.config['DOCKING_RESULTS_DIR'], job_id)
    filepath = os.path.join(job_results_dir, 'docking_results.csv')

    if os.path.isfile(filepath):
        df = pd.read_csv(filepath)
        binding_affinities = df['binding_affinity'].tolist()
        file_names = df['file_name'].tolist()
        chart_data = {
            'labels': file_names,
            'datasets': [{
                'label': 'Binding Affinity',
                'data': binding_affinities,
                'backgroundColor': 'rgba(0, 123, 255, 0.5)',
                'borderColor': 'rgba(0, 123, 255, 1)',
                'borderWidth': 1
            }]
        }
        return jsonify(chart_data)

    return jsonify({'message': f'Results not ready for job {job_id}'}), 202


@app.route('/download_complexes/<job_id>')
def download_complexes(job_id):
    job_results_dir = os.path.join(app.config['DOCKING_RESULTS_DIR'], job_id)

    if not os.path.isdir(job_results_dir):
        return abort(404, description="Job results not found.")

    zip_in_memory = create_zip_from_directory(job_results_dir)
    zip_filename = f'{job_id}_results.zip'
    return send_file(zip_in_memory, download_name=zip_filename, as_attachment=True, mimetype='application/zip')


# =========================
# ADMET
# =========================

def normalize_admet_input_dataframe(df):
    renamed_df = df.copy()
    if 'smiles' in renamed_df.columns:
        return renamed_df

    for col in renamed_df.columns:
        if str(col).strip().lower() == 'smiles':
            renamed_df = renamed_df.rename(columns={col: 'smiles'})
            return renamed_df

    raise ValueError('The uploaded CSV must contain a column named "smiles".')


def run_admet_cli(input_csv_path, output_csv_path):
    python_exe = resolve_python_executable(app.config['ADMET_PYTHON_EXE'])
    bridge_script = app.config['ADMET_BRIDGE_SCRIPT']
    if not os.path.exists(bridge_script):
        raise FileNotFoundError(f'ADMET bridge script not found: {bridge_script}')
    cmd = [python_exe, bridge_script, input_csv_path, output_csv_path]
    print('\n===== ADMET BRIDGE COMMAND START =====')
    print('Command:', ' '.join(cmd))
    result = run_command_capture(cmd, timeout=7200)
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)
    print('===== ADMET BRIDGE COMMAND END =====\n')
    if not os.path.exists(output_csv_path):
        raise RuntimeError(f'ADMET output file was not created: {output_csv_path}')
    return result




def prepare_admet_job_directory():
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(app.config['ADMET_JOBS_DIR'], job_id)
    os.makedirs(job_dir, exist_ok=True)
    return job_id, job_dir


@app.route('/admet', methods=['GET'])
def admet_page():
    return render_template('upload.html', active_tab='admet')


@app.route('/admet_predict_single', methods=['POST'])
def admet_predict_single():
    single_smiles = request.form.get('single_smiles', '').strip()

    if not single_smiles:
        flash('Please enter a valid SMILES string.', 'error')
        return render_template('upload.html', active_tab='admet')

    try:
        job_id, job_dir = prepare_admet_job_directory()
        input_csv_path = os.path.join(job_dir, 'admet_input.csv')
        output_csv_path = os.path.join(job_dir, 'admet_output.csv')

        input_df = pd.DataFrame({'smiles': [single_smiles]})
        input_df.to_csv(input_csv_path, index=False)

        run_admet_cli(input_csv_path, output_csv_path)
        result_df = pd.read_csv(output_csv_path)

        return render_template(
            'admet_results.html',
            job_id=job_id,
            total_rows=len(result_df),
            total_columns=len(result_df.columns),
            results_table=result_df.to_html(classes='table table-striped table-bordered table-hover', index=False),
            download_url=url_for('download_admet_results', job_id=job_id)
        )
    except FileNotFoundError as e:
        flash(str(e), 'error')
        return render_template('upload.html', active_tab='admet')
    except Exception as e:
        flash(f'ADMET prediction failed: {str(e)}', 'error')
        return render_template('upload.html', active_tab='admet')


@app.route('/admet_predict_csv', methods=['POST'])
def admet_predict_csv():
    uploaded_file = request.files.get('admet_file')

    if not uploaded_file or uploaded_file.filename == '':
        flash('Please upload a CSV file.', 'error')
        return render_template('upload.html', active_tab='admet')

    if not uploaded_file.filename.lower().endswith('.csv'):
        flash('Please upload a valid CSV file.', 'error')
        return render_template('upload.html', active_tab='admet')

    try:
        job_id, job_dir = prepare_admet_job_directory()
        raw_input_path = os.path.join(job_dir, secure_filename(uploaded_file.filename))
        prepared_input_path = os.path.join(job_dir, 'admet_input.csv')
        output_csv_path = os.path.join(job_dir, 'admet_output.csv')

        uploaded_file.save(raw_input_path)

        raw_df = pd.read_csv(raw_input_path)
        prepared_df = normalize_admet_input_dataframe(raw_df)
        prepared_df.to_csv(prepared_input_path, index=False)

        run_admet_cli(prepared_input_path, output_csv_path)
        result_df = pd.read_csv(output_csv_path)

        return render_template(
            'admet_results.html',
            job_id=job_id,
            total_rows=len(result_df),
            total_columns=len(result_df.columns),
            results_table=result_df.to_html(classes='table table-striped table-bordered table-hover', index=False),
            download_url=url_for('download_admet_results', job_id=job_id)
        )
    except FileNotFoundError as e:
        flash(str(e), 'error')
        return render_template('upload.html', active_tab='admet')
    except Exception as e:
        flash(f'ADMET prediction failed: {str(e)}', 'error')
        return render_template('upload.html', active_tab='admet')


@app.route('/download_admet_results/<job_id>', methods=['GET'])
def download_admet_results(job_id):
    job_dir = os.path.join(app.config['ADMET_JOBS_DIR'], job_id)
    output_csv_path = os.path.join(job_dir, 'admet_output.csv')

    if os.path.exists(output_csv_path):
        return send_file(output_csv_path, as_attachment=True, download_name=f'admet_results_{job_id}.csv')

    return abort(404, description='ADMET results file not found.')


# =========================
# QSAR
# =========================

def normalize_qsar_training_dataframe(df, target_column):
    renamed_df = df.copy()

    smiles_col = resolve_case_insensitive_column(renamed_df.columns, 'smiles')
    if smiles_col is None:
        raise ValueError('The uploaded training CSV must contain a column named "smiles".')

    if smiles_col != 'smiles':
        renamed_df = renamed_df.rename(columns={smiles_col: 'smiles'})

    target_col = resolve_case_insensitive_column(renamed_df.columns, target_column)
    if target_col is None:
        raise ValueError(f'The uploaded training CSV must contain the target column "{target_column}".')

    if target_col != target_column:
        renamed_df = renamed_df.rename(columns={target_col: target_column})

    return renamed_df


def normalize_qsar_prediction_dataframe(df):
    renamed_df = df.copy()

    smiles_col = resolve_case_insensitive_column(renamed_df.columns, 'smiles')
    if smiles_col is None:
        raise ValueError('The uploaded prediction CSV must contain a column named "smiles".')

    if smiles_col != 'smiles':
        renamed_df = renamed_df.rename(columns={smiles_col: 'smiles'})

    return renamed_df


def normalize_qsar_model_choice(model_type):
    model_type = (model_type or '').strip().lower()

    mapping = {
        'graphconv': 'graphconv',
        'mpnn': 'graphconv',
        'ecfp_rf': 'rf',
        'ecfp_svm': 'svm',
        'rf': 'rf',
        'svm': 'svm'
    }

    if model_type not in mapping:
        raise ValueError(f'Unsupported QSAR model type: {model_type}')

    return mapping[model_type]


def prepare_qsar_job_directory():
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(app.config['QSAR_JOBS_DIR'], job_id)
    model_dir = os.path.join(app.config['QSAR_MODELS_DIR'], job_id)

    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(job_dir, 'plots'), exist_ok=True)

    return job_id, job_dir, model_dir


def prepare_qsar_prediction_job_directory():
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(app.config['QSAR_JOBS_DIR'], job_id)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(os.path.join(job_dir, 'plots'), exist_ok=True)
    return job_id, job_dir


def run_qsar_training_bridge(
    input_csv_path,
    target_column,
    task_type,
    model_type,
    split_type,
    test_size,
    epochs,
    batch_size,
    random_seed,
    job_dir,
    model_dir
):
    python_exe = resolve_python_executable(app.config['QSAR_PYTHON_EXE'])
    bridge_script = app.config['QSAR_BRIDGE_SCRIPT']
    if not os.path.exists(bridge_script):
        raise FileNotFoundError(f'QSAR bridge script not found: {bridge_script}')
    cmd = [python_exe, bridge_script, 'train', '--input_csv', input_csv_path, '--target_column', target_column, '--task_type', task_type, '--model_type', model_type, '--split_type', split_type, '--test_size', str(test_size), '--epochs', str(epochs), '--batch_size', str(batch_size), '--random_seed', str(random_seed), '--job_dir', job_dir, '--model_dir', model_dir]
    print('\n===== QSAR TRAIN COMMAND START =====')
    print('Command:', ' '.join(cmd))
    result = run_command_capture(cmd, timeout=7200)
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)
    print('===== QSAR TRAIN COMMAND END =====\n')
    return result




def run_qsar_prediction_bridge(input_csv_path, job_dir, model_dir):
    python_exe = resolve_python_executable(app.config['QSAR_PYTHON_EXE'])
    bridge_script = app.config['QSAR_BRIDGE_SCRIPT']
    if not os.path.exists(bridge_script):
        raise FileNotFoundError(f'QSAR bridge script not found: {bridge_script}')
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f'Trained QSAR model directory not found: {model_dir}')
    cmd = [python_exe, bridge_script, 'predict', '--input_csv', input_csv_path, '--job_dir', job_dir, '--model_dir', model_dir]
    print('\n===== QSAR PREDICT COMMAND START =====')
    print('Command:', ' '.join(cmd))
    result = run_command_capture(cmd, timeout=7200)
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)
    print('===== QSAR PREDICT COMMAND END =====\n')
    return result




@app.route('/qsar', methods=['GET'])
def qsar_page():
    return render_template('upload.html', active_tab='qsar')


@app.route('/qsar_train', methods=['POST'])
def qsar_train():
    uploaded_file = request.files.get('qsar_train_file')
    target_column = request.form.get('qsar_target_column', '').strip()
    task_type = request.form.get('qsar_task_type', 'classification').strip()
    selected_model_type = request.form.get('qsar_model_type', 'graphconv').strip()
    split_type = request.form.get('qsar_split_type', 'random').strip()
    test_size = request.form.get('qsar_test_size', type=float) or 0.2
    epochs = request.form.get('qsar_epochs', type=int) or 20
    batch_size = request.form.get('qsar_batch_size', type=int) or 32
    random_seed = request.form.get('qsar_random_seed', type=int) or 42

    if not uploaded_file or uploaded_file.filename == '':
        flash('Please upload a QSAR training CSV file.', 'error')
        return render_template('upload.html', active_tab='qsar')

    if not uploaded_file.filename.lower().endswith('.csv'):
        flash('Please upload a valid QSAR training CSV file.', 'error')
        return render_template('upload.html', active_tab='qsar')

    if not target_column:
        flash('Please provide a valid target column name.', 'error')
        return render_template('upload.html', active_tab='qsar')

    try:
        normalized_model_type = normalize_qsar_model_choice(selected_model_type)

        job_id, job_dir, model_dir = prepare_qsar_job_directory()
        raw_input_path = os.path.join(job_dir, secure_filename(uploaded_file.filename))
        prepared_input_path = os.path.join(job_dir, 'qsar_training_input.csv')

        uploaded_file.save(raw_input_path)

        raw_df = pd.read_csv(raw_input_path)
        prepared_df = normalize_qsar_training_dataframe(raw_df, target_column)
        prepared_df.to_csv(prepared_input_path, index=False)

        run_qsar_training_bridge(
            input_csv_path=prepared_input_path,
            target_column=target_column,
            task_type=task_type,
            model_type=normalized_model_type,
            split_type=split_type,
            test_size=test_size,
            epochs=epochs,
            batch_size=batch_size,
            random_seed=random_seed,
            job_dir=job_dir,
            model_dir=model_dir
        )

        metrics_path = first_existing_path([
            os.path.join(job_dir, 'metrics.json'),
            os.path.join(job_dir, 'training_metrics.json')
        ])

        train_predictions_path = first_existing_path([
            os.path.join(job_dir, 'train_predictions.csv'),
            os.path.join(job_dir, 'training_predictions.csv')
        ])

        test_predictions_path = first_existing_path([
            os.path.join(job_dir, 'test_predictions.csv'),
            os.path.join(job_dir, 'validation_predictions.csv')
        ])

        all_results_path = first_existing_path([
            os.path.join(job_dir, 'all_predictions.csv'),
            os.path.join(job_dir, 'qsar_results.csv'),
            test_predictions_path
        ])

        performance_plot_path = first_existing_path([
            os.path.join(job_dir, 'plots', 'performance.png'),
            os.path.join(job_dir, 'plots', 'training_curve.png'),
            os.path.join(job_dir, 'plots', 'metrics_plot.png')
        ])

        secondary_plot_path = first_existing_path([
            os.path.join(job_dir, 'plots', 'prediction_scatter.png'),
            os.path.join(job_dir, 'plots', 'roc_curve.png'),
            os.path.join(job_dir, 'plots', 'confusion_matrix.png')
        ])

        metrics = {}
        if metrics_path and os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)

        results_table = safe_read_csv_to_html(all_results_path)
        train_table = safe_read_csv_to_html(train_predictions_path)
        test_table = safe_read_csv_to_html(test_predictions_path)

        performance_plot_rel = None
        secondary_plot_rel = None

        if performance_plot_path:
            performance_plot_rel = os.path.relpath(performance_plot_path, job_dir).replace("\\", "/")
        if secondary_plot_path:
            secondary_plot_rel = os.path.relpath(secondary_plot_path, job_dir).replace("\\", "/")

        total_rows = 0
        total_columns = 0
        if all_results_path and os.path.exists(all_results_path):
            result_df = pd.read_csv(all_results_path)
            total_rows = len(result_df)
            total_columns = len(result_df.columns)

        return render_template(
            'qsar_results.html',
            job_id=job_id,
            total_rows=total_rows,
            total_columns=total_columns,
            target_column=target_column,
            task_type=task_type,
            model_type=selected_model_type,
            split_type=split_type,
            metrics=metrics,
            results_table=results_table,
            train_table=train_table,
            test_table=test_table,
            performance_plot_url=url_for('download_qsar_job_file', job_id=job_id, filename=performance_plot_rel) if performance_plot_rel else None,
            secondary_plot_url=url_for('download_qsar_job_file', job_id=job_id, filename=secondary_plot_rel) if secondary_plot_rel else None,
            metrics_download_url=url_for('download_qsar_job_file', job_id=job_id, filename=os.path.basename(metrics_path)) if metrics_path else None,
            train_predictions_download_url=url_for('download_qsar_job_file', job_id=job_id, filename=os.path.basename(train_predictions_path)) if train_predictions_path else None,
            test_predictions_download_url=url_for('download_qsar_job_file', job_id=job_id, filename=os.path.basename(test_predictions_path)) if test_predictions_path else None,
            results_download_url=url_for('download_qsar_job_file', job_id=job_id, filename=os.path.basename(all_results_path)) if all_results_path else None,
            model_download_url=url_for('download_qsar_model', job_id=job_id)
        )
    except FileNotFoundError as e:
        flash(str(e), 'error')
        return render_template('upload.html', active_tab='qsar')
    except Exception as e:
        flash(f'QSAR training failed: {str(e)}', 'error')
        return render_template('upload.html', active_tab='qsar')


@app.route('/qsar_predict', methods=['POST'])
def qsar_predict():
    uploaded_file = request.files.get('qsar_prediction_file')
    model_job_id = request.form.get('qsar_model_job_id', '').strip()

    if not uploaded_file or uploaded_file.filename == '':
        flash('Please upload a QSAR prediction CSV file.', 'error')
        return render_template('upload.html', active_tab='qsar')

    if not uploaded_file.filename.lower().endswith('.csv'):
        flash('Please upload a valid QSAR prediction CSV file.', 'error')
        return render_template('upload.html', active_tab='qsar')

    if not model_job_id:
        flash('Please provide a valid trained model job ID.', 'error')
        return render_template('upload.html', active_tab='qsar')

    try:
        model_dir = os.path.join(app.config['QSAR_MODELS_DIR'], model_job_id)
        prediction_job_id, prediction_job_dir = prepare_qsar_prediction_job_directory()

        raw_input_path = os.path.join(prediction_job_dir, secure_filename(uploaded_file.filename))
        prepared_input_path = os.path.join(prediction_job_dir, 'qsar_prediction_input.csv')

        uploaded_file.save(raw_input_path)

        raw_df = pd.read_csv(raw_input_path)
        prepared_df = normalize_qsar_prediction_dataframe(raw_df)
        prepared_df.to_csv(prepared_input_path, index=False)

        run_qsar_prediction_bridge(
            input_csv_path=prepared_input_path,
            job_dir=prediction_job_dir,
            model_dir=model_dir
        )

        predictions_path = first_existing_path([
            os.path.join(prediction_job_dir, 'external_predictions.csv'),
            os.path.join(prediction_job_dir, 'prediction_results.csv'),
            os.path.join(prediction_job_dir, 'qsar_external_predictions.csv')
        ])

        prediction_plot_path = first_existing_path([
            os.path.join(prediction_job_dir, 'plots', 'prediction_plot.png'),
            os.path.join(prediction_job_dir, 'plots', 'prediction_distribution.png'),
            os.path.join(prediction_job_dir, 'plots', 'prediction_histogram.png')
        ])

        results_table = safe_read_csv_to_html(predictions_path)

        prediction_plot_rel = None
        if prediction_plot_path:
            prediction_plot_rel = os.path.relpath(prediction_plot_path, prediction_job_dir).replace("\\", "/")

        total_rows = 0
        total_columns = 0
        if predictions_path and os.path.exists(predictions_path):
            pred_df = pd.read_csv(predictions_path)
            total_rows = len(pred_df)
            total_columns = len(pred_df.columns)

        return render_template(
            'qsar_predict_results.html',
            job_id=prediction_job_id,
            model_job_id=model_job_id,
            total_rows=total_rows,
            total_columns=total_columns,
            results_table=results_table,
            prediction_plot_url=url_for('download_qsar_job_file', job_id=prediction_job_id, filename=prediction_plot_rel) if prediction_plot_rel else None,
            predictions_download_url=url_for('download_qsar_job_file', job_id=prediction_job_id, filename=os.path.basename(predictions_path)) if predictions_path else None
        )
    except FileNotFoundError as e:
        flash(str(e), 'error')
        return render_template('upload.html', active_tab='qsar')
    except Exception as e:
        flash(f'QSAR prediction failed: {str(e)}', 'error')
        return render_template('upload.html', active_tab='qsar')


@app.route('/download_qsar_job_file/<job_id>/<path:filename>', methods=['GET'])
def download_qsar_job_file(job_id, filename):
    job_dir = os.path.join(app.config['QSAR_JOBS_DIR'], job_id)
    if not os.path.isdir(job_dir):
        return abort(404, description='QSAR job directory not found.')
    return send_from_directory(directory=job_dir, path=filename, as_attachment=False)


@app.route('/download_qsar_model/<job_id>', methods=['GET'])
def download_qsar_model(job_id):
    model_dir = os.path.join(app.config['QSAR_MODELS_DIR'], job_id)

    if not os.path.isdir(model_dir):
        return abort(404, description='QSAR model directory not found.')

    zip_in_memory = BytesIO()
    with zipfile.ZipFile(zip_in_memory, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, model_dir))

    zip_in_memory.seek(0)
    zip_filename = f'qsar_model_{job_id}.zip'
    return send_file(zip_in_memory, download_name=zip_filename, as_attachment=True, mimetype='application/zip')


# =========================
# HIT-TO-LEAD OPTIMIZATION
# =========================

def normalize_hit_to_lead_input_dataframe(df):
    renamed_df = df.copy()

    smiles_col = resolve_case_insensitive_column(renamed_df.columns, 'smiles')
    if smiles_col is None:
        raise ValueError('The uploaded hit-to-lead CSV must contain a column named "smiles".')

    if smiles_col != 'smiles':
        renamed_df = renamed_df.rename(columns={smiles_col: 'smiles'})

    return renamed_df


def prepare_hit_to_lead_job_directory():
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(app.config['HIT_TO_LEAD_JOBS_DIR'], job_id)

    for sub_dir in ['input', 'output', 'plots', 'reports', 'temp']:
        os.makedirs(os.path.join(job_dir, sub_dir), exist_ok=True)

    return job_id, job_dir


def run_hit_to_lead_bridge(
    input_csv_path,
    reference_smiles,
    top_n,
    similarity_threshold,
    max_analogs,
    apply_lipinski,
    job_dir
):
    python_exe = resolve_python_executable(app.config['HIT_TO_LEAD_PYTHON_EXE'])
    bridge_script = app.config['HIT_TO_LEAD_BRIDGE_SCRIPT']
    if not os.path.exists(bridge_script):
        raise FileNotFoundError(f'Hit-to-Lead bridge script not found: {bridge_script}')
    cmd = [python_exe, bridge_script, '--input_csv', input_csv_path, '--job_dir', job_dir, '--top_n', str(top_n), '--similarity_threshold', str(similarity_threshold), '--max_analogs', str(max_analogs), '--apply_lipinski', str(apply_lipinski)]
    if reference_smiles:
        cmd.extend(['--reference_smiles', reference_smiles])
    print('\n===== HIT-TO-LEAD COMMAND START =====')
    print('Command:', ' '.join(cmd))
    result = run_command_capture(cmd, timeout=7200)
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)
    print('===== HIT-TO-LEAD COMMAND END =====\n')
    return result




@app.route('/hit_to_lead', methods=['GET'])
def hit_to_lead_page():
    return render_template('upload.html', active_tab='hit_to_lead')


@app.route('/run_hit_to_lead', methods=['POST'])
def run_hit_to_lead():
    uploaded_file = request.files.get('htl_file')
    reference_smiles = request.form.get('htl_reference_smiles', '').strip()
    top_n = request.form.get('htl_top_n', type=int) or 50
    similarity_threshold = request.form.get('htl_similarity_threshold', type=float) or 0.60
    max_analogs = request.form.get('htl_max_analogs', type=int) or 20
    apply_lipinski = request.form.get('htl_apply_lipinski', 'yes').strip().lower()

    if not uploaded_file or uploaded_file.filename == '':
        flash('Please upload a valid hit-to-lead CSV file.', 'error')
        return render_template('upload.html', active_tab='hit_to_lead')

    if not uploaded_file.filename.lower().endswith('.csv'):
        flash('Please upload a valid hit-to-lead CSV file.', 'error')
        return render_template('upload.html', active_tab='hit_to_lead')

    try:
        job_id, job_dir = prepare_hit_to_lead_job_directory()

        raw_input_path = os.path.join(job_dir, 'input', secure_filename(uploaded_file.filename))
        prepared_input_path = os.path.join(job_dir, 'input', 'hit_to_lead_input.csv')

        uploaded_file.save(raw_input_path)

        raw_df = pd.read_csv(raw_input_path)
        prepared_df = normalize_hit_to_lead_input_dataframe(raw_df)
        prepared_df.to_csv(prepared_input_path, index=False)

        run_hit_to_lead_bridge(
            input_csv_path=prepared_input_path,
            reference_smiles=reference_smiles,
            top_n=top_n,
            similarity_threshold=similarity_threshold,
            max_analogs=max_analogs,
            apply_lipinski=apply_lipinski,
            job_dir=job_dir
        )

        results_csv_path = first_existing_path([
            os.path.join(job_dir, 'output', 'ranked_analogs.csv'),
            os.path.join(job_dir, 'output', 'hit_to_lead_results.csv'),
            os.path.join(job_dir, 'output', 'optimized_hits.csv')
        ])

        summary_json_path = first_existing_path([
            os.path.join(job_dir, 'reports', 'summary.json'),
            os.path.join(job_dir, 'output', 'summary.json'),
            os.path.join(job_dir, 'summary.json')
        ])

        summary_txt_path = first_existing_path([
            os.path.join(job_dir, 'reports', 'summary.txt'),
            os.path.join(job_dir, 'reports', 'report.txt')
        ])

        ranking_plot_path = first_existing_path([
            os.path.join(job_dir, 'plots', 'ranking_plot.png'),
            os.path.join(job_dir, 'plots', 'similarity_distribution.png'),
            os.path.join(job_dir, 'plots', 'hit_to_lead_overview.png')
        ])

        property_plot_path = first_existing_path([
            os.path.join(job_dir, 'plots', 'property_distribution.png'),
            os.path.join(job_dir, 'plots', 'filter_summary.png'),
            os.path.join(job_dir, 'plots', 'lead_profile.png')
        ])

        if not results_csv_path or not os.path.exists(results_csv_path):
            raise RuntimeError('Hit-to-Lead results file was not generated.')

        fallback_ranking_plot, fallback_property_plot, fallback_summary_json = generate_hit_to_lead_fallback_artifacts(results_csv_path, job_dir)
        ranking_plot_path = ranking_plot_path or fallback_ranking_plot
        property_plot_path = property_plot_path or fallback_property_plot
        summary_json_path = summary_json_path or fallback_summary_json

        results_table = safe_read_csv_to_html(results_csv_path)
        summary_data = safe_read_json_file(summary_json_path)
        summary_text = safe_read_text_file(summary_txt_path)

        result_df = pd.read_csv(results_csv_path)
        total_rows = len(result_df)
        total_columns = len(result_df.columns)

        ranking_plot_rel = None
        property_plot_rel = None
        results_rel = os.path.relpath(results_csv_path, job_dir).replace("\\", "/")
        summary_json_rel = os.path.relpath(summary_json_path, job_dir).replace("\\", "/") if summary_json_path else None
        summary_txt_rel = os.path.relpath(summary_txt_path, job_dir).replace("\\", "/") if summary_txt_path else None

        if ranking_plot_path:
            ranking_plot_rel = os.path.relpath(ranking_plot_path, job_dir).replace("\\", "/")
        if property_plot_path:
            property_plot_rel = os.path.relpath(property_plot_path, job_dir).replace("\\", "/")

        return render_template(
            'hit_to_lead_results.html',
            job_id=job_id,
            total_rows=total_rows,
            total_columns=total_columns,
            reference_smiles=reference_smiles,
            summary_data=summary_data,
            summary_text=summary_text,
            results_table=results_table,
            ranking_plot_url=url_for('view_hit_to_lead_job_file', job_id=job_id, filename=ranking_plot_rel) if ranking_plot_rel else None,
            property_plot_url=url_for('view_hit_to_lead_job_file', job_id=job_id, filename=property_plot_rel) if property_plot_rel else None,
            ranking_plot_download_url=url_for('download_hit_to_lead_job_file', job_id=job_id, filename=ranking_plot_rel) if ranking_plot_rel else None,
            property_plot_download_url=url_for('download_hit_to_lead_job_file', job_id=job_id, filename=property_plot_rel) if property_plot_rel else None,
            results_download_url=url_for('download_hit_to_lead_job_file', job_id=job_id, filename=results_rel),
            summary_download_url=url_for('download_hit_to_lead_job_file', job_id=job_id, filename=summary_json_rel) if summary_json_rel else None,
            summary_text_download_url=url_for('download_hit_to_lead_job_file', job_id=job_id, filename=summary_txt_rel) if summary_txt_rel else None,
            job_zip_download_url=url_for('download_hit_to_lead_job_zip', job_id=job_id)
        )
    except FileNotFoundError as e:
        flash(str(e), 'error')
        return render_template('upload.html', active_tab='hit_to_lead')
    except Exception as e:
        flash(f'Hit-to-Lead optimization failed: {str(e)}', 'error')
        return render_template('upload.html', active_tab='hit_to_lead')


@app.route('/view_hit_to_lead_job_file/<job_id>/<path:filename>', methods=['GET'])
def view_hit_to_lead_job_file(job_id, filename):
    job_dir = os.path.join(app.config['HIT_TO_LEAD_JOBS_DIR'], job_id)
    if not os.path.isdir(job_dir):
        return abort(404, description='Hit-to-Lead job directory not found.')
    try:
        return safe_send_job_file(job_dir, filename, as_attachment=False)
    except FileNotFoundError as e:
        return abort(404, description=str(e))




@app.route('/download_hit_to_lead_job_file/<job_id>/<path:filename>', methods=['GET'])
def download_hit_to_lead_job_file(job_id, filename):
    job_dir = os.path.join(app.config['HIT_TO_LEAD_JOBS_DIR'], job_id)
    if not os.path.isdir(job_dir):
        return abort(404, description='Hit-to-Lead job directory not found.')
    try:
        return safe_send_job_file(job_dir, filename, as_attachment=True, download_name=os.path.basename(filename))
    except FileNotFoundError as e:
        return abort(404, description=str(e))




@app.route('/download_hit_to_lead_job_zip/<job_id>', methods=['GET'])
def download_hit_to_lead_job_zip(job_id):
    job_dir = os.path.join(app.config['HIT_TO_LEAD_JOBS_DIR'], job_id)
    if not os.path.isdir(job_dir):
        return abort(404, description='Hit-to-Lead job directory not found.')

    zip_in_memory = create_zip_from_directory(job_dir)
    zip_filename = f'hit_to_lead_{job_id}.zip'
    return send_file(zip_in_memory, download_name=zip_filename, as_attachment=True, mimetype='application/zip')


# =========================
# PROTEIN-LIGAND INTERACTION ANALYSIS
# =========================

def prepare_plip_job_directory():
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(app.config['PLIP_JOBS_DIR'], job_id)

    for sub_dir in ['input', 'output', 'plots', 'reports', 'temp']:
        os.makedirs(os.path.join(job_dir, sub_dir), exist_ok=True)

    return job_id, job_dir


def run_plip_bridge(
    complex_file_path,
    ligand_code,
    report_format,
    include_hydrophobic,
    include_pi,
    job_dir
):
    python_exe = resolve_python_executable(app.config['PLIP_PYTHON_EXE'])
    bridge_script = app.config['PLIP_BRIDGE_SCRIPT']
    if not os.path.exists(bridge_script):
        raise FileNotFoundError(f'PLIP bridge script not found: {bridge_script}')
    cmd = [python_exe, bridge_script, '--complex_file', complex_file_path, '--job_dir', job_dir, '--report_format', report_format, '--include_hydrophobic', include_hydrophobic, '--include_pi', include_pi]
    if ligand_code:
        cmd.extend(['--ligand_code', ligand_code])
    print('\n===== PLIP ANALYSIS COMMAND START =====')
    print('Command:', ' '.join(cmd))
    result = run_command_capture(cmd, timeout=7200)
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)
    print('===== PLIP ANALYSIS COMMAND END =====\n')
    return result




@app.route('/plip_analysis', methods=['GET'])
def plip_analysis_page():
    return render_template('upload.html', active_tab='plip_analysis')


@app.route('/run_plip_analysis', methods=['POST'])
def run_plip_analysis():
    complex_file = request.files.get('plip_complex_file')
    ligand_code = request.form.get('plip_ligand_code', '').strip()
    report_format = request.form.get('plip_report_format', 'standard').strip()
    include_hydrophobic = request.form.get('plip_include_hydrophobic', 'yes').strip().lower()
    include_pi = request.form.get('plip_include_pi', 'yes').strip().lower()

    if not complex_file or complex_file.filename == '':
        flash('Please upload a protein–ligand complex PDB file.', 'error')
        return render_template('upload.html', active_tab='plip_analysis')

    if not complex_file.filename.lower().endswith('.pdb'):
        flash('Please upload a valid PDB complex file.', 'error')
        return render_template('upload.html', active_tab='plip_analysis')

    try:
        job_id, job_dir = prepare_plip_job_directory()

        complex_file_path = os.path.join(job_dir, 'input', secure_filename(complex_file.filename))
        complex_file.save(complex_file_path)

        run_plip_bridge(
            complex_file_path=complex_file_path,
            ligand_code=ligand_code,
            report_format=report_format,
            include_hydrophobic=include_hydrophobic,
            include_pi=include_pi,
            job_dir=job_dir
        )

        interactions_csv_path = first_existing_path([
            os.path.join(job_dir, 'output', 'interaction_summary.csv'),
            os.path.join(job_dir, 'output', 'plip_interactions.csv'),
            os.path.join(job_dir, 'output', 'interaction_table.csv')
        ])

        counts_csv_path = first_existing_path([
            os.path.join(job_dir, 'output', 'interaction_counts.csv'),
            os.path.join(job_dir, 'output', 'interaction_type_counts.csv')
        ])

        summary_json_path = first_existing_path([
            os.path.join(job_dir, 'reports', 'summary.json'),
            os.path.join(job_dir, 'output', 'summary.json')
        ])

        report_txt_path = first_existing_path([
            os.path.join(job_dir, 'reports', 'plip_report.txt'),
            os.path.join(job_dir, 'reports', 'interaction_report.txt'),
            os.path.join(job_dir, 'reports', 'report.txt')
        ])

        interaction_plot_path = first_existing_path([
            os.path.join(job_dir, 'plots', 'interaction_barplot.png'),
            os.path.join(job_dir, 'plots', 'interaction_overview.png'),
            os.path.join(job_dir, 'plots', 'interaction_counts.png')
        ])

        secondary_plot_path = first_existing_path([
            os.path.join(job_dir, 'plots', 'binding_site_overview.png'),
            os.path.join(job_dir, 'plots', 'residue_contact_map.png')
        ])

        if not interactions_csv_path or not os.path.exists(interactions_csv_path):
            raise RuntimeError('Protein–Ligand interaction results file was not generated.')

        fallback_counts_csv, fallback_interaction_plot, fallback_secondary_plot = generate_plip_fallback_artifacts(interactions_csv_path, job_dir)
        counts_csv_path = counts_csv_path or fallback_counts_csv
        interaction_plot_path = interaction_plot_path or fallback_interaction_plot
        secondary_plot_path = secondary_plot_path or fallback_secondary_plot
        if summary_json_path is None and os.path.exists(os.path.join(job_dir, 'reports', 'summary.json')):
            summary_json_path = os.path.join(job_dir, 'reports', 'summary.json')

        interactions_table = safe_read_csv_to_html(interactions_csv_path)
        counts_table = safe_read_csv_to_html(counts_csv_path)
        summary_data = safe_read_json_file(summary_json_path)
        report_text = safe_read_text_file(report_txt_path)

        interaction_df = pd.read_csv(interactions_csv_path)
        total_rows = len(interaction_df)
        total_columns = len(interaction_df.columns)

        interaction_plot_rel = None
        secondary_plot_rel = None
        interactions_rel = os.path.relpath(interactions_csv_path, job_dir).replace("\\", "/")
        counts_rel = os.path.relpath(counts_csv_path, job_dir).replace("\\", "/") if counts_csv_path else None
        summary_json_rel = os.path.relpath(summary_json_path, job_dir).replace("\\", "/") if summary_json_path else None
        report_txt_rel = os.path.relpath(report_txt_path, job_dir).replace("\\", "/") if report_txt_path else None

        if interaction_plot_path:
            interaction_plot_rel = os.path.relpath(interaction_plot_path, job_dir).replace("\\", "/")
        if secondary_plot_path:
            secondary_plot_rel = os.path.relpath(secondary_plot_path, job_dir).replace("\\", "/")

        return render_template(
            'plip_results.html',
            job_id=job_id,
            ligand_code=ligand_code,
            total_rows=total_rows,
            total_columns=total_columns,
            summary_data=summary_data,
            report_text=report_text,
            interactions_table=interactions_table,
            counts_table=counts_table,
            interaction_plot_url=url_for('view_plip_job_file', job_id=job_id, filename=interaction_plot_rel) if interaction_plot_rel else None,
            secondary_plot_url=url_for('view_plip_job_file', job_id=job_id, filename=secondary_plot_rel) if secondary_plot_rel else None,
            interaction_plot_download_url=url_for('download_plip_job_file', job_id=job_id, filename=interaction_plot_rel) if interaction_plot_rel else None,
            secondary_plot_download_url=url_for('download_plip_job_file', job_id=job_id, filename=secondary_plot_rel) if secondary_plot_rel else None,
            interactions_download_url=url_for('download_plip_job_file', job_id=job_id, filename=interactions_rel),
            counts_download_url=url_for('download_plip_job_file', job_id=job_id, filename=counts_rel) if counts_rel else None,
            summary_download_url=url_for('download_plip_job_file', job_id=job_id, filename=summary_json_rel) if summary_json_rel else None,
            report_download_url=url_for('download_plip_job_file', job_id=job_id, filename=report_txt_rel) if report_txt_rel else None,
            job_zip_download_url=url_for('download_plip_job_zip', job_id=job_id)
        )
    except FileNotFoundError as e:
        flash(str(e), 'error')
        return render_template('upload.html', active_tab='plip_analysis')
    except Exception as e:
        flash(f'Protein–Ligand Interaction Analysis failed: {str(e)}', 'error')
        return render_template('upload.html', active_tab='plip_analysis')


@app.route('/view_plip_job_file/<job_id>/<path:filename>', methods=['GET'])
def view_plip_job_file(job_id, filename):
    job_dir = os.path.join(app.config['PLIP_JOBS_DIR'], job_id)
    if not os.path.isdir(job_dir):
        return abort(404, description='PLIP job directory not found.')
    try:
        return safe_send_job_file(job_dir, filename, as_attachment=False)
    except FileNotFoundError as e:
        return abort(404, description=str(e))




@app.route('/download_plip_job_file/<job_id>/<path:filename>', methods=['GET'])
def download_plip_job_file(job_id, filename):
    job_dir = os.path.join(app.config['PLIP_JOBS_DIR'], job_id)
    if not os.path.isdir(job_dir):
        return abort(404, description='PLIP job directory not found.')
    try:
        return safe_send_job_file(job_dir, filename, as_attachment=True, download_name=os.path.basename(filename))
    except FileNotFoundError as e:
        return abort(404, description=str(e))




@app.route('/download_plip_job_zip/<job_id>', methods=['GET'])
def download_plip_job_zip(job_id):
    job_dir = os.path.join(app.config['PLIP_JOBS_DIR'], job_id)
    if not os.path.isdir(job_dir):
        return abort(404, description='PLIP job directory not found.')

    zip_in_memory = create_zip_from_directory(job_dir)
    zip_filename = f'plip_{job_id}.zip'
    return send_file(zip_in_memory, download_name=zip_filename, as_attachment=True, mimetype='application/zip')


# =========================
# BLIND DOCKING
# =========================

def calculate_blind_box_from_pdb(pdb_file_path, padding=10.0):
    xs, ys, zs = [], [], []

    with open(pdb_file_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                except ValueError:
                    continue

    if not xs or not ys or not zs:
        raise ValueError("No valid atomic coordinates found in the PDB file.")

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_z = (min_z + max_z) / 2.0

    size_x = (max_x - min_x) + padding
    size_y = (max_y - min_y) + padding
    size_z = (max_z - min_z) + padding

    return {
        "center_x": round(center_x, 3),
        "center_y": round(center_y, 3),
        "center_z": round(center_z, 3),
        "size_x": round(size_x, 3),
        "size_y": round(size_y, 3),
        "size_z": round(size_z, 3),
        "min_x": round(min_x, 3),
        "max_x": round(max_x, 3),
        "min_y": round(min_y, 3),
        "max_y": round(max_y, 3),
        "min_z": round(min_z, 3),
        "max_z": round(max_z, 3),
    }


def generate_site_centers_from_box(box_info, site_box_size=24.0, spacing=18.0, max_sites=25):
    centers = []

    x = box_info["min_x"]
    while x <= box_info["max_x"]:
        y = box_info["min_y"]
        while y <= box_info["max_y"]:
            z = box_info["min_z"]
            while z <= box_info["max_z"]:
                centers.append({
                    "center_x": round(x, 3),
                    "center_y": round(y, 3),
                    "center_z": round(z, 3),
                    "size_x": round(site_box_size, 3),
                    "size_y": round(site_box_size, 3),
                    "size_z": round(site_box_size, 3)
                })
                z += spacing
            y += spacing
        x += spacing

    all_sites = [{
        "center_x": box_info["center_x"],
        "center_y": box_info["center_y"],
        "center_z": box_info["center_z"],
        "size_x": box_info["size_x"],
        "size_y": box_info["size_y"],
        "size_z": box_info["size_z"]
    }] + centers

    return all_sites[:max_sites]


def parse_vina_results_from_pdbqt(output_pdbqt_path):
    poses = []

    if not os.path.exists(output_pdbqt_path):
        return poses

    pose_rank = 0
    with open(output_pdbqt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("REMARK VINA RESULT:"):
                parts = line.strip().split()
                if len(parts) >= 6:
                    pose_rank += 1
                    poses.append({
                        "pose_rank": pose_rank,
                        "binding_affinity": float(parts[3]),
                        "rmsd_lb": float(parts[4]),
                        "rmsd_ub": float(parts[5])
                    })

    return poses


def run_blind_multi_site_docking(
    receptor_pdbqt_path,
    ligand_dir,
    results_dir,
    site_centers,
    exhaustiveness=8,
    num_modes=9,
    energy_range=3
):
    os.makedirs(results_dir, exist_ok=True)

    ligand_files = sorted(Path(ligand_dir).rglob("*.pdbqt"))
    records = []

    for ligand_path in ligand_files:
        ligand_name = Path(ligand_path).stem

        for idx, site in enumerate(site_centers):
            site_name = f"site_{idx}"
            out_pdbqt = os.path.join(results_dir, f"{ligand_name}_{site_name}_out.pdbqt")
            config_path = os.path.join(results_dir, f"{ligand_name}_{site_name}_config.txt")

            config_text = f"""receptor = {receptor_pdbqt_path}
ligand = {str(ligand_path)}

center_x = {site['center_x']}
center_y = {site['center_y']}
center_z = {site['center_z']}
size_x = {site['size_x']}
size_y = {site['size_y']}
size_z = {site['size_z']}

out = {out_pdbqt}
exhaustiveness = {exhaustiveness}
num_modes = {num_modes}
energy_range = {energy_range}
"""

            with open(config_path, "w") as cf:
                cf.write(config_text)

            vina_command = ['.\\vina.exe', '--config', config_path]

            try:
                result = subprocess.run(vina_command, capture_output=True, text=True)
            except Exception as e:
                result = None
                print(f"Blind docking exception for {ligand_name} at {site_name}: {e}")

            poses = parse_vina_results_from_pdbqt(out_pdbqt)

            for pose in poses:
                records.append({
                    "ligand_name": ligand_name,
                    "site_name": site_name,
                    "pose_rank": pose["pose_rank"],
                    "binding_affinity": pose["binding_affinity"],
                    "rmsd_lb": pose["rmsd_lb"],
                    "rmsd_ub": pose["rmsd_ub"],
                    "center_x": site["center_x"],
                    "center_y": site["center_y"],
                    "center_z": site["center_z"],
                    "size_x": site["size_x"],
                    "size_y": site["size_y"],
                    "size_z": site["size_z"],
                    "vina_returncode": result.returncode if result else -1,
                    "vina_stdout": result.stdout if result else "",
                    "vina_stderr": result.stderr if result else ""
                })

            if os.path.exists(config_path):
                os.remove(config_path)

    all_results_df = pd.DataFrame(records)

    if len(all_results_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_results_df = all_results_df[all_results_df["binding_affinity"].notna()].copy()

    if len(all_results_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    best_per_ligand_df = (
        all_results_df.sort_values(["binding_affinity", "pose_rank"], ascending=[True, True])
        .groupby("ligand_name", as_index=False)
        .first()
    )

    best_per_site_df = (
        all_results_df.sort_values(["binding_affinity", "pose_rank"], ascending=[True, True])
        .groupby("site_name", as_index=False)
        .first()
    )

    csv_path = os.path.join(results_dir, "blind_docking_results.csv")
    all_results_df.to_csv(csv_path, index=False)

    best_ligand_csv = os.path.join(results_dir, "best_per_ligand.csv")
    best_site_csv = os.path.join(results_dir, "best_per_site.csv")
    best_per_ligand_df.to_csv(best_ligand_csv, index=False)
    best_per_site_df.to_csv(best_site_csv, index=False)

    return all_results_df, best_per_ligand_df, best_per_site_df


def generate_blind_docking_charts(best_per_ligand_df, best_per_site_df, job_id):
    os.makedirs("static", exist_ok=True)

    ligand_chart = f"blind_ligands_{job_id}.png"
    site_chart = f"blind_sites_{job_id}.png"

    ligand_chart_path = os.path.join("static", ligand_chart)
    site_chart_path = os.path.join("static", site_chart)

    if len(best_per_ligand_df) > 0:
        lig_df = best_per_ligand_df.sort_values("binding_affinity", ascending=True)

        plt.figure(figsize=(12, 6))
        plt.bar(lig_df["ligand_name"], lig_df["binding_affinity"])
        plt.xticks(rotation=90)
        plt.ylabel("Binding Affinity (kcal/mol)")
        plt.xlabel("Ligands")
        plt.title("Best Docking Score per Ligand")
        plt.tight_layout()
        plt.savefig(ligand_chart_path)
        plt.close()

    if len(best_per_site_df) > 0:
        site_df = best_per_site_df.sort_values("binding_affinity", ascending=True)

        plt.figure(figsize=(12, 6))
        plt.bar(site_df["site_name"], site_df["binding_affinity"])
        plt.xticks(rotation=90)
        plt.ylabel("Binding Affinity (kcal/mol)")
        plt.xlabel("Predicted Binding Sites")
        plt.title("Top Predicted Binding Sites")
        plt.tight_layout()
        plt.savefig(site_chart_path)
        plt.close()

    return ligand_chart, site_chart


@app.route('/blind-docking', methods=['GET', 'POST'])
def blind_docking_page():
    if request.method == 'GET':
        return render_template('upload.html', active_tab='blind_docking')

    protein_file = request.files.get('protein_file')
    ligand_zip = request.files.get('ligand_zip')

    if not protein_file or not ligand_zip:
        return "Protein file and ligand ZIP are required.", 400

    if not protein_file.filename.lower().endswith('.pdb'):
        return "Protein file must be a PDB file.", 400

    if not ligand_zip.filename.lower().endswith('.zip'):
        return "Ligand input must be provided as a ZIP archive containing SDF files.", 400

    exhaustiveness = request.form.get('exhaustiveness', type=int) or 8
    num_modes = request.form.get('num_modes', type=int) or 9
    energy_range = request.form.get('energy_range', type=float) or 3.0
    blind_padding = request.form.get('blind_padding', type=float) or 10.0
    site_box_size = request.form.get('site_box_size', type=float) or 24.0
    site_spacing = request.form.get('site_spacing', type=float) or 18.0
    max_sites = request.form.get('max_sites', type=int) or 25

    job_id = uuid.uuid4().hex
    job_dir = os.path.join(app.config['BLIND_JOBS_DIR'], job_id)
    ligand_dir = os.path.join(job_dir, "ligands")
    results_dir = os.path.join(job_dir, "results")

    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(ligand_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    protein_pdb_path = os.path.join(job_dir, secure_filename(protein_file.filename))
    ligand_zip_path = os.path.join(job_dir, secure_filename(ligand_zip.filename))

    protein_file.save(protein_pdb_path)
    ligand_zip.save(ligand_zip_path)

    safe_extract_zip(ligand_zip_path, ligand_dir)

    receptor_pdbqt_path = os.path.join(job_dir, "receptor.pdbqt")
    receptor_pdbqt_path = convert_protein(protein_pdb_path, receptor_pdbqt_path)

    convert_sdf_to_pdbqt(None, ligand_dir)

    box_info = calculate_blind_box_from_pdb(protein_pdb_path, padding=blind_padding)

    site_centers = generate_site_centers_from_box(
        box_info,
        site_box_size=site_box_size,
        spacing=site_spacing,
        max_sites=max_sites
    )

    all_results_df, best_per_ligand_df, best_per_site_df = run_blind_multi_site_docking(
        receptor_pdbqt_path=receptor_pdbqt_path,
        ligand_dir=ligand_dir,
        results_dir=results_dir,
        site_centers=site_centers,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        energy_range=energy_range
    )

    if len(all_results_df) == 0:
        return "Blind docking completed, but no valid docking scores were generated.", 500

    ligand_chart, site_chart = generate_blind_docking_charts(
        best_per_ligand_df,
        best_per_site_df,
        job_id
    )

    top_sites_df = best_per_site_df.sort_values("binding_affinity", ascending=True).head(10)
    best_ligands_df = best_per_ligand_df.sort_values("binding_affinity", ascending=True)
    all_sorted_df = all_results_df.sort_values(["binding_affinity", "pose_rank"], ascending=[True, True])

    return render_template(
        'blind_results.html',
        job_id=job_id,
        box_info=box_info,
        ligand_chart=ligand_chart,
        site_chart=site_chart,
        top_sites_table=top_sites_df.to_html(classes='table table-striped table-bordered', index=False),
        best_ligands_table=best_ligands_df.to_html(classes='table table-striped table-bordered', index=False),
        all_results_table=all_sorted_df.to_html(classes='table table-striped table-bordered', index=False),
        job_zip_download_url=url_for('download_blind_job_zip', job_id=job_id)
    )


@app.route('/download_blind_job_zip/<job_id>', methods=['GET'])
def download_blind_job_zip(job_id):
    job_dir = os.path.join(app.config['BLIND_JOBS_DIR'], job_id)
    if not os.path.isdir(job_dir):
        return abort(404, description='Blind docking job directory not found.')
    zip_in_memory = create_zip_from_directory(job_dir)
    zip_filename = f'blind_docking_{job_id}.zip'
    return send_file(zip_in_memory, download_name=zip_filename, as_attachment=True, mimetype='application/zip')


if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    for directory in [
        RUNTIME_DIR,
        STATIC_DIR,
        MODELS_DIR,
        THIRD_PARTY_DIR,
        app.config['UPLOAD_FOLDER'],
        app.config['GENERATED_FILES_DIR'],
        app.config['DOCKING_RESULTS_DIR'],
        app.config['BLIND_JOBS_DIR'],
        app.config['ADMET_JOBS_DIR'],
        app.config['QSAR_JOBS_DIR'],
        app.config['QSAR_MODELS_DIR'],
        app.config['HIT_TO_LEAD_JOBS_DIR'],
        app.config['PLIP_JOBS_DIR'],
        app.config['AUTOGROW4_DIR'],
        app.config['PLIP_TOOL_DIR'],
        os.path.join(STATIC_DIR, 'css'),
        os.path.join(STATIC_DIR, 'js'),
        os.path.join(STATIC_DIR, 'img')
    ]:
        os.makedirs(directory, exist_ok=True)

    app.run(host="0.0.0.0", port=5001, debug=False, threaded=False)
