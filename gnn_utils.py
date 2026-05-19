"""
DrugForge AI GNN utilities - Enhanced 2D version

This module provides:
1. SMILES standardization and molecule cleaning
2. Rich 2D molecular graph construction for DGL
3. Fixed-length molecular descriptor/fingerprint generation
4. Dataset and collate utilities for model training/inference
5. Bemis-Murcko scaffold splitting helper
6. Applicability-domain helper using Tanimoto similarity

Main upgrade over old version:
- Old node feature: atomic number only
- New node feature: 31 chemically meaningful atom features
- Old edge feature: bond order only
- New edge feature: 15 chemically meaningful bond features
"""

from collections import defaultdict
import random

import dgl
import numpy as np
import torch

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, QED, RDKFingerprint

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:  # pragma: no cover
    MurckoScaffold = None

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
except Exception:  # pragma: no cover
    rdMolStandardize = None

from torch.utils.data import Dataset


# -------------------------------------------------------------------------
# Global dimensions
# -------------------------------------------------------------------------

# 6 descriptors + 512 Morgan + 167 MACCS + 512 RDK
# Kept unchanged to avoid breaking the descriptor/fingerprint branch.
FEATURE_VECTOR_DIM = 1197

# Enhanced atom feature dimension used by mol_to_graph().
# If you change atom_features(), update this value.
ATOM_FEATURE_DIM = 31

# Enhanced bond feature dimension used by mol_to_graph().
# If you change bond_features(), update this value.
BOND_FEATURE_DIM = 15


# -------------------------------------------------------------------------
# Safe numeric helpers
# -------------------------------------------------------------------------

def _safe_float(value, default=0.0):
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return float(default)
        return value
    except Exception:
        return float(default)


def _clip_and_scale(value, low, high):
    value = _safe_float(value, default=low)
    value = float(np.clip(value, low, high))
    return (value - low) / (high - low) if high > low else 0.0


def _one_hot(value, choices, include_unknown=True):
    """
    Generic one-hot encoder.

    If include_unknown=True, the last index is used for unknown/other values.
    """
    encoded = [0.0] * (len(choices) + int(include_unknown))

    try:
        idx = choices.index(value)
        encoded[idx] = 1.0
    except ValueError:
        if include_unknown:
            encoded[-1] = 1.0

    return encoded


# -------------------------------------------------------------------------
# Molecule standardization
# -------------------------------------------------------------------------

def _cleanup_with_rdkit_standardizer(mol):
    """
    Use RDKit MolStandardize when available.

    Steps:
    - Cleanup
    - Keep largest fragment
    - Uncharge molecule
    - Canonical tautomer selection
    """
    if rdMolStandardize is None:
        return mol

    try:
        mol = rdMolStandardize.Cleanup(mol)
    except Exception:
        pass

    try:
        largest_fragment_chooser = rdMolStandardize.LargestFragmentChooser()
        mol = largest_fragment_chooser.choose(mol)
    except Exception:
        pass

    try:
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
    except Exception:
        pass

    try:
        enumerator = rdMolStandardize.TautomerEnumerator()
        mol = enumerator.Canonicalize(mol)
    except Exception:
        pass

    return mol


def standardize_smiles_and_mol(smiles):
    """
    Convert raw SMILES into standardized canonical SMILES and RDKit Mol.

    Returns
    -------
    canonical_smiles : str or None
    mol : rdkit.Chem.Mol or None
    """
    if smiles is None:
        return None, None

    smiles = str(smiles).strip()
    if not smiles:
        return None, None

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception:
        return None, None

    if mol is None:
        return None, None

    # Keep largest fragment for salts / mixtures.
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    except Exception:
        return None, None

    mol = _cleanup_with_rdkit_standardizer(mol)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None, None

    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        pass

    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        pass

    try:
        canonical_smiles = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True
        )
    except Exception:
        return None, None

    if not canonical_smiles:
        return None, None

    return canonical_smiles, mol


# -------------------------------------------------------------------------
# Scaffold utilities
# -------------------------------------------------------------------------

def get_bemis_murcko_scaffold(smiles):
    """
    Return Bemis-Murcko scaffold SMILES for scaffold split.
    """
    canonical_smiles, mol = standardize_smiles_and_mol(smiles)
    if canonical_smiles is None or mol is None or MurckoScaffold is None:
        return None

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold or None
    except Exception:
        return None


def scaffold_split_indices(
    smiles_list,
    train_frac=0.70,
    val_frac=0.15,
    test_frac=0.15,
    seed=42
):
    """
    Bemis-Murcko scaffold split.

    This is stronger than random split for medicinal chemistry because
    molecules with the same core scaffold are kept in the same split.

    Returns
    -------
    train_indices, val_indices, test_indices : list[int]
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must be 1.0")

    scaffold_to_indices = defaultdict(list)

    for idx, smiles in enumerate(smiles_list):
        scaffold = get_bemis_murcko_scaffold(smiles)
        if scaffold is None:
            scaffold = f"NO_SCAFFOLD_{idx}"
        scaffold_to_indices[scaffold].append(idx)

    scaffold_groups = list(scaffold_to_indices.values())

    # Large scaffolds first; randomize only among same-size groups for stability.
    rng = random.Random(seed)
    rng.shuffle(scaffold_groups)
    scaffold_groups = sorted(scaffold_groups, key=len, reverse=True)

    n_total = len(smiles_list)
    n_train_target = int(train_frac * n_total)
    n_val_target = int(val_frac * n_total)

    train_indices = []
    val_indices = []
    test_indices = []

    for group in scaffold_groups:
        if len(train_indices) + len(group) <= n_train_target:
            train_indices.extend(group)
        elif len(val_indices) + len(group) <= n_val_target:
            val_indices.extend(group)
        else:
            test_indices.extend(group)

    # Safety fallback: if val or test becomes empty in very small datasets.
    if len(val_indices) == 0 and len(train_indices) > 1:
        val_indices.append(train_indices.pop())
    if len(test_indices) == 0 and len(train_indices) > 1:
        test_indices.append(train_indices.pop())

    return train_indices, val_indices, test_indices


# -------------------------------------------------------------------------
# Enhanced atom and bond features
# -------------------------------------------------------------------------

def _compute_gasteiger_charges(mol):
    """
    Compute Gasteiger charges safely.
    """
    try:
        mol_copy = Chem.Mol(mol)
        AllChem.ComputeGasteigerCharges(mol_copy)
        return mol_copy
    except Exception:
        return mol


def atom_features(atom):
    """
    Create a chemically meaningful atom feature vector.

    Feature layout:
    1. Atomic number scaled
    2. Atom degree one-hot
    3. Formal charge scaled
    4. Hybridization one-hot
    5. Aromaticity
    6. Ring membership
    7. Total hydrogen count one-hot
    8. Implicit valence scaled
    9. Total valence scaled
    10. Chirality one-hot
    11. Atomic mass scaled
    12. Gasteiger partial charge scaled

    Returns
    -------
    list[float] of length ATOM_FEATURE_DIM
    """
    atomic_num = _clip_and_scale(atom.GetAtomicNum(), 0, 100)

    degree = _one_hot(
        atom.GetDegree(),
        choices=[0, 1, 2, 3, 4, 5],
        include_unknown=True
    )

    formal_charge = _clip_and_scale(atom.GetFormalCharge(), -5, 5)

    hybridization = _one_hot(
        atom.GetHybridization(),
        choices=[
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ],
        include_unknown=True
    )

    aromatic = float(atom.GetIsAromatic())
    in_ring = float(atom.IsInRing())

    total_hs = _one_hot(
        atom.GetTotalNumHs(),
        choices=[0, 1, 2, 3, 4],
        include_unknown=True
    )

    try:
        implicit_valence = _clip_and_scale(atom.GetImplicitValence(), 0, 8)
    except Exception:
        implicit_valence = 0.0

    try:
        total_valence = _clip_and_scale(atom.GetTotalValence(), 0, 8)
    except Exception:
        total_valence = 0.0

    chiral_tag = _one_hot(
        atom.GetChiralTag(),
        choices=[
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
        ],
        include_unknown=True
    )

    mass = _clip_and_scale(atom.GetMass(), 0, 250)

    try:
        gasteiger_charge = atom.GetProp("_GasteigerCharge")
        gasteiger_charge = _clip_and_scale(gasteiger_charge, -2.0, 2.0)
    except Exception:
        gasteiger_charge = 0.5  # scaled neutral midpoint for [-2, 2]

    features = (
        [atomic_num]
        + degree
        + [formal_charge]
        + hybridization
        + [aromatic]
        + [in_ring]
        + total_hs
        + [implicit_valence]
        + [total_valence]
        + chiral_tag
        + [mass]
        + [gasteiger_charge]
    )

    if len(features) != ATOM_FEATURE_DIM:
        raise ValueError(
            f"Atom feature length {len(features)} does not match "
            f"ATOM_FEATURE_DIM={ATOM_FEATURE_DIM}"
        )

    return features


def bond_features(bond, is_self_loop=False):
    """
    Create a chemically meaningful bond feature vector.

    Feature layout:
    1. Bond type one-hot
    2. Conjugation
    3. Ring membership
    4. Stereo one-hot
    5. Bond order scaled
    6. Self-loop flag

    Returns
    -------
    list[float] of length BOND_FEATURE_DIM
    """
    if is_self_loop or bond is None:
        features = (
            [0.0, 0.0, 0.0, 0.0, 0.0]  # bond type placeholder
            + [0.0]                    # conjugated
            + [0.0]                    # ring
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # stereo placeholder
            + [0.0]                    # bond order
            + [1.0]                    # self-loop
        )
    else:
        bond_type = _one_hot(
            bond.GetBondType(),
            choices=[
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC
            ],
            include_unknown=True
        )

        conjugated = float(bond.GetIsConjugated())
        in_ring = float(bond.IsInRing())

        stereo = _one_hot(
            bond.GetStereo(),
            choices=[
                Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOANY,
                Chem.rdchem.BondStereo.STEREOZ,
                Chem.rdchem.BondStereo.STEREOE,
                Chem.rdchem.BondStereo.STEREOCIS,
                Chem.rdchem.BondStereo.STEREOTRANS
            ],
            include_unknown=False
        )

        bond_order = _clip_and_scale(bond.GetBondTypeAsDouble(), 0.0, 3.0)

        features = (
            bond_type
            + [conjugated]
            + [in_ring]
            + stereo
            + [bond_order]
            + [0.0]  # self-loop flag
        )

    if len(features) != BOND_FEATURE_DIM:
        raise ValueError(
            f"Bond feature length {len(features)} does not match "
            f"BOND_FEATURE_DIM={BOND_FEATURE_DIM}"
        )

    return features


def mol_to_graph(mol):
    """
    Convert an RDKit Mol into a DGL graph.

    Graph data:
    - g.ndata["h"] : atom features, shape [num_atoms, ATOM_FEATURE_DIM]
    - g.edata["e"] : bond features, shape [num_edges, BOND_FEATURE_DIM]
    """
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None

    # Compute Gasteiger charges on a copy for atom features.
    mol_with_charges = _compute_gasteiger_charges(mol)

    src = []
    dst = []
    edge_feats = []

    for bond in mol_with_charges.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bf = bond_features(bond, is_self_loop=False)

        # Add both directions because DGL graph is directed.
        src.extend([start, end])
        dst.extend([end, start])
        edge_feats.extend([bf, bf])

    # Add self-loops to stabilize message passing.
    for i in range(num_atoms):
        src.append(i)
        dst.append(i)
        edge_feats.append(bond_features(None, is_self_loop=True))

    g = dgl.graph((src, dst), num_nodes=num_atoms)

    h_feats = [atom_features(atom) for atom in mol_with_charges.GetAtoms()]

    g.ndata["h"] = torch.tensor(h_feats, dtype=torch.float32)
    g.edata["e"] = torch.tensor(edge_feats, dtype=torch.float32)

    return g


# -------------------------------------------------------------------------
# Descriptor/fingerprint vector
# -------------------------------------------------------------------------

def build_molecular_feature_vector(mol):
    """
    Build fixed-length descriptor/fingerprint vector.

    Kept as 1197 dimensions for compatibility:
    - 6 scaled descriptors
    - 512 Morgan fingerprint bits
    - 167 MACCS keys
    - 512 RDK fingerprint bits
    """
    mol_wt = _clip_and_scale(Descriptors.MolWt(mol), 0.0, 1000.0)
    tpsa = _clip_and_scale(Descriptors.TPSA(mol), 0.0, 300.0)
    h_donors = _clip_and_scale(Descriptors.NumHDonors(mol), 0.0, 20.0)
    h_acceptors = _clip_and_scale(Descriptors.NumHAcceptors(mol), 0.0, 20.0)
    logp = _clip_and_scale(Descriptors.MolLogP(mol), -5.0, 10.0)
    rot_bonds = _clip_and_scale(Descriptors.NumRotatableBonds(mol), 0.0, 30.0)

    descriptors = np.array(
        [mol_wt, tpsa, h_donors, h_acceptors, logp, rot_bonds],
        dtype=np.float32
    )

    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=512,
        useChirality=True
    )
    morgan_bits = np.zeros((512,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(morgan_fp, morgan_bits)

    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_bits = np.zeros((167,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(maccs_fp, maccs_bits)

    rdk_fp = RDKFingerprint(
        mol,
        fpSize=512,
        maxPath=7,
        branchedPaths=True,
        useHs=False
    )
    rdk_bits = np.zeros((512,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(rdk_fp, rdk_bits)

    features = np.concatenate(
        [descriptors, morgan_bits, maccs_bits, rdk_bits]
    ).astype(np.float32)

    if features.shape[0] != FEATURE_VECTOR_DIM:
        raise ValueError(
            f"Unexpected feature length {features.shape[0]}; "
            f"expected {FEATURE_VECTOR_DIM}."
        )

    return torch.tensor(features, dtype=torch.float32)


# -------------------------------------------------------------------------
# Molecule filtering and drug-like properties
# -------------------------------------------------------------------------

METAL_ATOMIC_NUMS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56,
    57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82
}


def is_bad_molecule(mol):
    """
    Filter molecules that are likely unsuitable for generic ligand QSAR/GNN.
    """
    if mol is None:
        return True

    num_heavy = mol.GetNumHeavyAtoms()
    mol_wt = _safe_float(Descriptors.MolWt(mol), default=0.0)
    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    atom_count = mol.GetNumAtoms()

    if any(atom.GetAtomicNum() in METAL_ATOMIC_NUMS for atom in mol.GetAtoms()):
        return True

    if num_heavy < 3:
        return True

    if mol_wt < 50 or mol_wt > 1000:
        return True

    if atom_count > 180:
        return True

    if abs(formal_charge) > 4:
        return True

    return False


def calculate_druglike_properties(smiles):
    """
    Calculate basic drug-like properties for output/ranking.
    """
    canonical_smiles, mol = standardize_smiles_and_mol(smiles)

    if canonical_smiles is None or mol is None:
        return {
            "CanonicalSMILES": None,
            "MW": np.nan,
            "LogP": np.nan,
            "TPSA": np.nan,
            "HBD": np.nan,
            "HBA": np.nan,
            "RotatableBonds": np.nan,
            "QED": np.nan,
            "LipinskiViolations": np.nan,
            "DrugLikeScore": np.nan,
        }

    mw = _safe_float(Descriptors.MolWt(mol))
    logp = _safe_float(Descriptors.MolLogP(mol))
    tpsa = _safe_float(Descriptors.TPSA(mol))
    hbd = int(Descriptors.NumHDonors(mol))
    hba = int(Descriptors.NumHAcceptors(mol))
    rot = int(Descriptors.NumRotatableBonds(mol))

    try:
        qed = _safe_float(QED.qed(mol))
    except Exception:
        qed = np.nan

    lipinski_violations = 0
    lipinski_violations += int(mw > 500)
    lipinski_violations += int(logp > 5)
    lipinski_violations += int(hbd > 5)
    lipinski_violations += int(hba > 10)

    # Simple bounded drug-like score.
    if np.isnan(qed):
        drug_like_score = max(0.0, 1.0 - 0.25 * lipinski_violations)
    else:
        drug_like_score = max(0.0, float(qed) - 0.10 * lipinski_violations)

    return {
        "CanonicalSMILES": canonical_smiles,
        "MW": mw,
        "LogP": logp,
        "TPSA": tpsa,
        "HBD": hbd,
        "HBA": hba,
        "RotatableBonds": rot,
        "QED": qed,
        "LipinskiViolations": lipinski_violations,
        "DrugLikeScore": drug_like_score,
    }


# -------------------------------------------------------------------------
# Applicability-domain helpers
# -------------------------------------------------------------------------

def smiles_to_morgan_fp(smiles, radius=2, n_bits=2048):
    """
    Convert SMILES to Morgan fingerprint for similarity/applicability domain.
    """
    canonical_smiles, mol = standardize_smiles_and_mol(smiles)
    if canonical_smiles is None or mol is None:
        return None

    try:
        return AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits,
            useChirality=True
        )
    except Exception:
        return None


def build_reference_fingerprints(smiles_list, radius=2, n_bits=2048):
    """
    Build reference fingerprints from training SMILES.
    """
    fps = []
    for smiles in smiles_list:
        fp = smiles_to_morgan_fp(smiles, radius=radius, n_bits=n_bits)
        if fp is not None:
            fps.append(fp)
    return fps


def max_tanimoto_similarity(smiles, reference_fps, radius=2, n_bits=2048):
    """
    Maximum Tanimoto similarity between one molecule and training set.
    """
    if not reference_fps:
        return 0.0

    fp = smiles_to_morgan_fp(smiles, radius=radius, n_bits=n_bits)
    if fp is None:
        return 0.0

    try:
        sims = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
        return float(max(sims)) if sims else 0.0
    except Exception:
        return 0.0


def applicability_domain_label(similarity, inside_threshold=0.35):
    """
    Convert Tanimoto similarity to applicability-domain label.
    """
    if similarity >= inside_threshold:
        return "Inside"
    return "Outside"


# -------------------------------------------------------------------------
# Dataset and collate utilities
# -------------------------------------------------------------------------

class MoleculeDataset(Dataset):
    """
    Dataset for 2D GNN training/inference.

    Parameters
    ----------
    smiles_list : list[str]
    labels : list[int] or None
        If None, dummy labels of zero are used. This is useful for prediction.
    """

    def __init__(self, smiles_list, labels=None):
        self.smiles_list = list(smiles_list)

        if labels is None:
            self.labels = [0] * len(self.smiles_list)
        else:
            self.labels = list(labels)

        if len(self.smiles_list) != len(self.labels):
            raise ValueError("smiles_list and labels must have the same length.")

        # Lazy caches keep memory use reasonable while preserving speed.
        self._standardized_cache = {}
        self._feature_cache = {}
        self._graph_cache = {}
        self._bad_cache = {}

    def __len__(self):
        return len(self.smiles_list)

    def _is_bad_molecule(self, mol):
        return is_bad_molecule(mol)

    def _build_feature_vector(self, mol):
        return build_molecular_feature_vector(mol)

    def _get_standardized(self, idx):
        if idx in self._standardized_cache:
            return self._standardized_cache[idx]

        smiles = self.smiles_list[idx]
        canonical_smiles, mol = standardize_smiles_and_mol(smiles)
        self._standardized_cache[idx] = (canonical_smiles, mol)
        return canonical_smiles, mol

    def __getitem__(self, idx):
        label = self.labels[idx]
        canonical_smiles, mol = self._get_standardized(idx)

        if canonical_smiles is None or mol is None:
            return None, None, None

        if idx not in self._bad_cache:
            self._bad_cache[idx] = self._is_bad_molecule(mol)

        if self._bad_cache[idx]:
            return None, None, None

        if idx not in self._graph_cache:
            self._graph_cache[idx] = mol_to_graph(mol)

        graph = self._graph_cache[idx]
        if graph is None:
            return None, None, None

        if idx not in self._feature_cache:
            self._feature_cache[idx] = self._build_feature_vector(mol)

        features = self._feature_cache[idx]

        return graph, features, int(label)


def collate(samples):
    """
    Collate function for DGL DataLoader.

    Removes invalid molecules returned as None.
    """
    valid_samples = [
        s for s in samples
        if s is not None and s[0] is not None and s[1] is not None
    ]

    if len(valid_samples) == 0:
        return None, None, None

    graphs, features, labels = map(list, zip(*valid_samples))
    batched_graph = dgl.batch(graphs)

    return (
        batched_graph,
        torch.stack(features),
        torch.tensor(labels, dtype=torch.long)
    )
