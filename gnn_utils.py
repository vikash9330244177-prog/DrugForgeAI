"""
DrugForge AI GNN utilities

This module provides:
1. SMILES standardization
2. Molecular graph construction for DGL
3. Fixed-length molecular feature generation
4. Dataset and collate utilities for model training/inference
"""

import dgl
import numpy as np
import torch

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, RDKFingerprint

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:  # pragma: no cover
    MurckoScaffold = None

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
except Exception:  # pragma: no cover
    rdMolStandardize = None

from torch.utils.data import Dataset


FEATURE_VECTOR_DIM = 1197  # 6 descriptors + 512 Morgan + 167 MACCS + 512 RDK


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


def _cleanup_with_rdkit_standardizer(mol):
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

    # Keep largest fragment for salts / mixtures
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
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None, None

    if not canonical_smiles:
        return None, None

    return canonical_smiles, mol


def get_bemis_murcko_scaffold(smiles):
    canonical_smiles, mol = standardize_smiles_and_mol(smiles)
    if canonical_smiles is None or mol is None or MurckoScaffold is None:
        return None

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold or None
    except Exception:
        return None


def mol_to_graph(mol):
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None

    src = []
    dst = []
    edge_feats = []

    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_type = float(bond.GetBondTypeAsDouble())

        src.extend([start, end])
        dst.extend([end, start])
        edge_feats.extend([bond_type, bond_type])

    # Add self-loops to stabilize message passing
    for i in range(num_atoms):
        src.append(i)
        dst.append(i)
        edge_feats.append(0.0)

    g = dgl.graph((src, dst), num_nodes=num_atoms)

    # Keep node feature dimension = 1 for compatibility with the existing model
    h_feats = [[float(atom.GetAtomicNum())] for atom in mol.GetAtoms()]
    g.ndata["h"] = torch.tensor(h_feats, dtype=torch.float32)
    g.edata["e"] = torch.tensor(edge_feats, dtype=torch.float32).unsqueeze(1)

    return g


class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, labels):
        self.smiles_list = list(smiles_list)
        self.labels = list(labels)

        self.metal_atomic_nums = {
            3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56,
            57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82
        }

        # Lazy caches keep memory use reasonable while preserving speed
        self._standardized_cache = {}
        self._feature_cache = {}
        self._graph_cache = {}
        self._bad_cache = {}

    def __len__(self):
        return len(self.smiles_list)

    def _is_bad_molecule(self, mol):
        if mol is None:
            return True

        num_heavy = mol.GetNumHeavyAtoms()
        mol_wt = _safe_float(Descriptors.MolWt(mol), default=0.0)
        formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        atom_count = mol.GetNumAtoms()

        # Reject metal-containing compounds; these often break generic GNN/QSAR workflows
        if any(atom.GetAtomicNum() in self.metal_atomic_nums for atom in mol.GetAtoms()):
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

    def _build_feature_vector(self, mol):
        # Preserve overall feature dimension for downstream compatibility
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

        features = np.concatenate([descriptors, morgan_bits, maccs_bits, rdk_bits]).astype(np.float32)

        if features.shape[0] != FEATURE_VECTOR_DIM:
            raise ValueError(
                f"Unexpected feature length {features.shape[0]}; expected {FEATURE_VECTOR_DIM}."
            )

        return torch.tensor(features, dtype=torch.float32)

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
    valid_samples = [s for s in samples if s is not None and s[0] is not None]

    if len(valid_samples) == 0:
        return None, None, None

    graphs, features, labels = map(list, zip(*valid_samples))
    batched_graph = dgl.batch(graphs)

    return (
        batched_graph,
        torch.stack(features),
        torch.tensor(labels, dtype=torch.long)
    )