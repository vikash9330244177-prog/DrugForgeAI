"""
DrugForge AI 3D GNN utilities

This module provides:
1. SMILES standardization reuse from enhanced gnn_utils.py
2. 3D conformer generation using RDKit ETKDGv3
3. MMFF/UFF geometry optimization
4. 3D molecular graph construction for DGL
5. Covalent bond edges + spatial radius edges
6. Fixed-length descriptor/fingerprint generation reuse
7. Dataset and collate utilities for 3D GNN training/inference
8. Applicability-domain helpers for final ranking

Expected project files:
- gnn_utils.py should already contain enhanced 2D atom/bond features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from gnn_utils import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    FEATURE_VECTOR_DIM,
    atom_features,
    bond_features,
    build_molecular_feature_vector,
    standardize_smiles_and_mol,
    is_bad_molecule,
    calculate_druglike_properties,
    build_reference_fingerprints,
    max_tanimoto_similarity,
    applicability_domain_label,
    scaffold_split_indices,
)


# -------------------------------------------------------------------------
# 3D graph constants
# -------------------------------------------------------------------------

# Extra 3D edge features added after enhanced 2D bond features:
# 1. scaled interatomic distance
# 2. covalent edge flag
# 3. spatial/non-bonded edge flag
# 4. self-loop flag
EDGE_3D_EXTRA_DIM = 4

# Final 3D edge feature dimension.
EDGE_FEATURE_DIM_3D = BOND_FEATURE_DIM + EDGE_3D_EXTRA_DIM

# Node features remain the same as enhanced 2D atom features.
NODE_FEATURE_DIM_3D = ATOM_FEATURE_DIM

# Default spatial radius cutoff in Angstrom.
DEFAULT_RADIUS_CUTOFF = 5.0

# Distance scaling upper bound in Angstrom.
DISTANCE_SCALE_MAX = 10.0


# -------------------------------------------------------------------------
# Basic numeric helpers
# -------------------------------------------------------------------------

def _safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _clip_and_scale(value, low: float, high: float) -> float:
    value = _safe_float(value, default=low)
    value = float(np.clip(value, low, high))
    if high <= low:
        return 0.0
    return float((value - low) / (high - low))


def _zero_bond_feature() -> List[float]:
    return [0.0] * BOND_FEATURE_DIM


def _make_3d_edge_feature(
    base_bond_feature: Sequence[float],
    distance: float,
    is_covalent: bool,
    is_spatial: bool,
    is_self_loop: bool,
) -> List[float]:
    """
    Build final 3D edge feature vector.

    Final layout:
    [enhanced 2D bond features] +
    [scaled distance, covalent flag, spatial flag, self-loop flag]
    """
    base = list(base_bond_feature)

    if len(base) < BOND_FEATURE_DIM:
        base = base + [0.0] * (BOND_FEATURE_DIM - len(base))
    elif len(base) > BOND_FEATURE_DIM:
        base = base[:BOND_FEATURE_DIM]

    dist_scaled = _clip_and_scale(distance, 0.0, DISTANCE_SCALE_MAX)

    feature = base + [
        dist_scaled,
        float(is_covalent),
        float(is_spatial),
        float(is_self_loop),
    ]

    if len(feature) != EDGE_FEATURE_DIM_3D:
        raise ValueError(
            f"3D edge feature length {len(feature)} does not match "
            f"EDGE_FEATURE_DIM_3D={EDGE_FEATURE_DIM_3D}"
        )

    return feature


# -------------------------------------------------------------------------
# 3D conformer generation
# -------------------------------------------------------------------------

@dataclass
class ConformerResult:
    """
    Container for 3D conformer generation result.
    """
    mol: Optional[Chem.Mol]
    status: str
    energy: Optional[float] = None
    canonical_smiles: Optional[str] = None


def generate_3d_conformer(
    smiles: str,
    random_seed: int = 42,
    max_attempts: int = 3,
    optimize: bool = True,
    forcefield: str = "MMFF",
    max_iters: int = 500,
) -> ConformerResult:
    """
    Generate and optimize a 3D conformer from SMILES.

    Parameters
    ----------
    smiles : str
        Input SMILES.
    random_seed : int
        Reproducible seed for conformer generation.
    max_attempts : int
        Number of embedding attempts.
    optimize : bool
        Whether to run geometry optimization.
    forcefield : str
        "MMFF" or "UFF".
    max_iters : int
        Maximum force-field optimization iterations.

    Returns
    -------
    ConformerResult
    """
    canonical_smiles, mol = standardize_smiles_and_mol(smiles)

    if canonical_smiles is None or mol is None:
        return ConformerResult(
            mol=None,
            status="invalid_smiles",
            energy=None,
            canonical_smiles=None,
        )

    if is_bad_molecule(mol):
        return ConformerResult(
            mol=None,
            status="filtered_bad_molecule",
            energy=None,
            canonical_smiles=canonical_smiles,
        )

    try:
        mol_h = Chem.AddHs(mol)
    except Exception:
        return ConformerResult(
            mol=None,
            status="add_hydrogens_failed",
            energy=None,
            canonical_smiles=canonical_smiles,
        )

    embedded = False
    last_status = None

    for attempt in range(max(1, int(max_attempts))):
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = int(random_seed + attempt)
            params.useSmallRingTorsions = True
            params.useMacrocycleTorsions = True
            params.enforceChirality = True

            status = AllChem.EmbedMolecule(mol_h, params)
            last_status = status

            if status == 0:
                embedded = True
                break

            # Fallback to random coordinates.
            status = AllChem.EmbedMolecule(
                mol_h,
                randomSeed=int(random_seed + attempt),
                useRandomCoords=True,
            )
            last_status = status

            if status == 0:
                embedded = True
                break

        except Exception:
            last_status = "exception"

    if not embedded:
        return ConformerResult(
            mol=None,
            status=f"embedding_failed_{last_status}",
            energy=None,
            canonical_smiles=canonical_smiles,
        )

    energy = None

    if optimize:
        ff = str(forcefield or "MMFF").strip().upper()

        try:
            if ff == "MMFF":
                props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
                if props is not None:
                    ff_obj = AllChem.MMFFGetMoleculeForceField(mol_h, props)
                    if ff_obj is not None:
                        ff_obj.Minimize(maxIts=int(max_iters))
                        energy = _safe_float(ff_obj.CalcEnergy(), default=np.nan)
                    else:
                        AllChem.UFFOptimizeMolecule(mol_h, maxIters=int(max_iters))
                else:
                    AllChem.UFFOptimizeMolecule(mol_h, maxIters=int(max_iters))
            else:
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=int(max_iters))
                ff_obj = AllChem.UFFGetMoleculeForceField(mol_h)
                if ff_obj is not None:
                    energy = _safe_float(ff_obj.CalcEnergy(), default=np.nan)

        except Exception:
            # Keep embedded conformer even if optimization fails.
            return ConformerResult(
                mol=mol_h,
                status="embedded_optimization_failed",
                energy=energy,
                canonical_smiles=canonical_smiles,
            )

    return ConformerResult(
        mol=mol_h,
        status="success",
        energy=energy,
        canonical_smiles=canonical_smiles,
    )


def extract_coordinates(mol: Chem.Mol) -> Optional[torch.Tensor]:
    """
    Extract 3D coordinates from RDKit Mol.

    Returns
    -------
    torch.Tensor or None
        Shape [num_atoms, 3]
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None

    try:
        conf = mol.GetConformer()
        coords = []

        for atom_idx in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(atom_idx)
            coords.append([float(pos.x), float(pos.y), float(pos.z)])

        coords = torch.tensor(coords, dtype=torch.float32)
        coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        return coords

    except Exception:
        return None


def remove_hs_keep_conformer(mol_h: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Remove hydrogens while preserving a heavy-atom conformer.

    This keeps graph size smaller and consistent with enhanced 2D GNN.
    """
    if mol_h is None:
        return None

    try:
        mol_no_h = Chem.RemoveHs(mol_h, sanitize=True)
        if mol_no_h is not None and mol_no_h.GetNumConformers() > 0:
            return mol_no_h
    except Exception:
        pass

    # If hydrogen removal fails, keep hydrogenated molecule.
    return mol_h


# -------------------------------------------------------------------------
# 3D graph construction
# -------------------------------------------------------------------------

def _bond_lookup(mol: Chem.Mol) -> Dict[Tuple[int, int], Chem.Bond]:
    """
    Dictionary for quick bond lookup.
    """
    lookup = {}

    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        lookup[(i, j)] = bond
        lookup[(j, i)] = bond

    return lookup


def _pairwise_distance(coords: torch.Tensor, i: int, j: int) -> float:
    return float(torch.norm(coords[i] - coords[j], p=2).item())


def mol_to_3d_graph(
    mol: Chem.Mol,
    radius_cutoff: float = DEFAULT_RADIUS_CUTOFF,
    include_spatial_edges: bool = True,
    include_self_loops: bool = True,
) -> Optional[dgl.DGLGraph]:
    """
    Convert 3D RDKit molecule into DGL graph.

    Graph contains:
    - covalent bond edges
    - optional non-bonded spatial edges within radius_cutoff
    - optional self-loops

    Graph fields:
    - g.ndata["h"]   : atom features [N, NODE_FEATURE_DIM_3D]
    - g.ndata["pos"] : 3D coordinates [N, 3]
    - g.edata["e"]   : 3D edge features [E, EDGE_FEATURE_DIM_3D]
    - g.edata["dist"]: interatomic distances [E, 1]
    """
    if mol is None:
        return None

    if mol.GetNumConformers() == 0:
        return None

    num_atoms = int(mol.GetNumAtoms())

    if num_atoms == 0:
        return None

    coords = extract_coordinates(mol)

    if coords is None or coords.size(0) != num_atoms:
        return None

    # Compute Gasteiger charges for atom_features().
    try:
        mol_for_features = Chem.Mol(mol)
        AllChem.ComputeGasteigerCharges(mol_for_features)
    except Exception:
        mol_for_features = mol

    src: List[int] = []
    dst: List[int] = []
    edge_feats: List[List[float]] = []
    distances: List[List[float]] = []

    added_edges = set()

    bond_map = _bond_lookup(mol_for_features)

    # 1. Covalent bond edges.
    for bond in mol_for_features.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        dist = _pairwise_distance(coords, i, j)
        base_bond = bond_features(bond, is_self_loop=False)

        for a, b in [(i, j), (j, i)]:
            src.append(a)
            dst.append(b)
            edge_feats.append(
                _make_3d_edge_feature(
                    base_bond_feature=base_bond,
                    distance=dist,
                    is_covalent=True,
                    is_spatial=False,
                    is_self_loop=False,
                )
            )
            distances.append([dist])
            added_edges.add((a, b))

    # 2. Spatial non-bonded edges.
    if include_spatial_edges:
        cutoff = float(radius_cutoff)

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if (i, j) in bond_map or (j, i) in bond_map:
                    continue

                dist = _pairwise_distance(coords, i, j)

                if dist <= cutoff:
                    for a, b in [(i, j), (j, i)]:
                        if (a, b) in added_edges:
                            continue

                        src.append(a)
                        dst.append(b)
                        edge_feats.append(
                            _make_3d_edge_feature(
                                base_bond_feature=_zero_bond_feature(),
                                distance=dist,
                                is_covalent=False,
                                is_spatial=True,
                                is_self_loop=False,
                            )
                        )
                        distances.append([dist])
                        added_edges.add((a, b))

    # 3. Self-loops.
    if include_self_loops:
        for i in range(num_atoms):
            src.append(i)
            dst.append(i)
            edge_feats.append(
                _make_3d_edge_feature(
                    base_bond_feature=bond_features(None, is_self_loop=True),
                    distance=0.0,
                    is_covalent=False,
                    is_spatial=False,
                    is_self_loop=True,
                )
            )
            distances.append([0.0])
            added_edges.add((i, i))

    if len(src) == 0:
        return None

    try:
        g = dgl.graph((src, dst), num_nodes=num_atoms)
    except Exception:
        return None

    try:
        node_features = [
            atom_features(atom)
            for atom in mol_for_features.GetAtoms()
        ]
    except Exception:
        return None

    g.ndata["h"] = torch.tensor(node_features, dtype=torch.float32)
    g.ndata["pos"] = coords.to(dtype=torch.float32)

    g.edata["e"] = torch.tensor(edge_feats, dtype=torch.float32)
    g.edata["dist"] = torch.tensor(distances, dtype=torch.float32)

    return g


def smiles_to_3d_graph(
    smiles: str,
    radius_cutoff: float = DEFAULT_RADIUS_CUTOFF,
    random_seed: int = 42,
    max_attempts: int = 3,
    optimize: bool = True,
    forcefield: str = "MMFF",
    keep_hydrogens: bool = False,
) -> Tuple[Optional[dgl.DGLGraph], Optional[torch.Tensor], Dict[str, object]]:
    """
    Convert SMILES to 3D graph + descriptor vector.

    Returns
    -------
    graph : DGLGraph or None
    features : torch.Tensor or None
        1197-dimensional descriptor/fingerprint vector.
    info : dict
        Canonical SMILES, conformer status, energy, etc.
    """
    result = generate_3d_conformer(
        smiles=smiles,
        random_seed=random_seed,
        max_attempts=max_attempts,
        optimize=optimize,
        forcefield=forcefield,
    )

    info = {
        "CanonicalSMILES": result.canonical_smiles,
        "ConformerStatus": result.status,
        "ConformerEnergy": result.energy,
    }

    if result.mol is None:
        return None, None, info

    mol_3d = result.mol if keep_hydrogens else remove_hs_keep_conformer(result.mol)

    if mol_3d is None:
        info["ConformerStatus"] = "remove_hydrogens_failed"
        return None, None, info

    graph = mol_to_3d_graph(
        mol=mol_3d,
        radius_cutoff=radius_cutoff,
        include_spatial_edges=True,
        include_self_loops=True,
    )

    if graph is None:
        info["ConformerStatus"] = "graph_construction_failed"
        return None, None, info

    # Descriptor vector should use standardized heavy-atom molecule.
    canonical_smiles, mol_2d = standardize_smiles_and_mol(smiles)

    if canonical_smiles is None or mol_2d is None:
        return None, None, info

    try:
        features = build_molecular_feature_vector(mol_2d)
    except Exception:
        info["ConformerStatus"] = "feature_vector_failed"
        return None, None, info

    info["CanonicalSMILES"] = canonical_smiles
    return graph, features, info


# -------------------------------------------------------------------------
# Dataset and collate utilities
# -------------------------------------------------------------------------

class Molecule3DDataset(Dataset):
    """
    Dataset for 3D GNN training/inference.

    Parameters
    ----------
    smiles_list : list[str]
    labels : list[int] or None
        If None, dummy labels are used.
    radius_cutoff : float
        Spatial edge cutoff in Angstrom.
    random_seed : int
        Seed for conformer generation.
    max_attempts : int
        Number of conformer embedding attempts.
    optimize : bool
        Whether to optimize conformer.
    forcefield : str
        "MMFF" or "UFF".
    keep_hydrogens : bool
        Whether to keep explicit hydrogens in 3D graph.
    """

    def __init__(
        self,
        smiles_list: Sequence[str],
        labels: Optional[Sequence[int]] = None,
        radius_cutoff: float = DEFAULT_RADIUS_CUTOFF,
        random_seed: int = 42,
        max_attempts: int = 3,
        optimize: bool = True,
        forcefield: str = "MMFF",
        keep_hydrogens: bool = False,
    ):
        self.smiles_list = list(smiles_list)

        if labels is None:
            self.labels = [0] * len(self.smiles_list)
        else:
            self.labels = list(labels)

        if len(self.smiles_list) != len(self.labels):
            raise ValueError("smiles_list and labels must have the same length.")

        self.radius_cutoff = float(radius_cutoff)
        self.random_seed = int(random_seed)
        self.max_attempts = int(max_attempts)
        self.optimize = bool(optimize)
        self.forcefield = str(forcefield or "MMFF").upper()
        self.keep_hydrogens = bool(keep_hydrogens)

        self._graph_cache: Dict[int, Optional[dgl.DGLGraph]] = {}
        self._feature_cache: Dict[int, Optional[torch.Tensor]] = {}
        self._info_cache: Dict[int, Dict[str, object]] = {}

    def __len__(self):
        return len(self.smiles_list)

    def _build_item(self, idx: int):
        if idx in self._graph_cache:
            return (
                self._graph_cache[idx],
                self._feature_cache[idx],
                self._info_cache[idx],
            )

        smiles = self.smiles_list[idx]

        graph, features, info = smiles_to_3d_graph(
            smiles=smiles,
            radius_cutoff=self.radius_cutoff,
            random_seed=self.random_seed + idx,
            max_attempts=self.max_attempts,
            optimize=self.optimize,
            forcefield=self.forcefield,
            keep_hydrogens=self.keep_hydrogens,
        )

        self._graph_cache[idx] = graph
        self._feature_cache[idx] = features
        self._info_cache[idx] = info

        return graph, features, info

    def get_info(self, idx: int) -> Dict[str, object]:
        """
        Return conformer information for a molecule.
        """
        _, _, info = self._build_item(idx)
        return info

    def valid_indices(self) -> List[int]:
        """
        Return indices whose 3D graph and features are valid.
        """
        valid = []

        for idx in range(len(self.smiles_list)):
            graph, features, _ = self._build_item(idx)
            if graph is not None and features is not None:
                valid.append(idx)

        return valid

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        graph, features, info = self._build_item(idx)

        if graph is None or features is None:
            return None, None, None, info

        return graph, features, label, info


def collate_3d(samples):
    """
    Collate function for 3D GNN DataLoader.

    Returns
    -------
    batched_graph : dgl.DGLGraph or None
    batched_features : torch.Tensor or None
    batched_labels : torch.Tensor or None
    infos : list[dict]
    """
    valid_samples = [
        s for s in samples
        if s is not None and s[0] is not None and s[1] is not None
    ]

    if len(valid_samples) == 0:
        return None, None, None, []

    graphs, features, labels, infos = map(list, zip(*valid_samples))

    batched_graph = dgl.batch(graphs)

    return (
        batched_graph,
        torch.stack(features),
        torch.tensor(labels, dtype=torch.long),
        infos,
    )


# -------------------------------------------------------------------------
# 3D result/ranking helpers
# -------------------------------------------------------------------------

def calculate_3d_priority_score(
    active_probability: float,
    uncertainty: float = 0.0,
    qed: float = 0.0,
    applicability_similarity: float = 1.0,
    lipinski_violations: int = 0,
) -> float:
    """
    Calculate final 3D priority score.

    High score is better.

    Components:
    - active probability
    - confidence from uncertainty
    - QED/drug-likeness
    - applicability-domain similarity
    - Lipinski penalty
    """
    prob = float(np.clip(_safe_float(active_probability), 0.0, 1.0))
    unc = float(np.clip(_safe_float(uncertainty), 0.0, 1.0))
    qed_val = float(np.clip(_safe_float(qed), 0.0, 1.0))
    app_sim = float(np.clip(_safe_float(applicability_similarity), 0.0, 1.0))

    try:
        lipinski_violations = int(lipinski_violations)
    except Exception:
        lipinski_violations = 4

    lipinski_bonus = float(np.clip(1.0 - min(max(lipinski_violations, 0), 4) / 4.0, 0.0, 1.0))

    score = (
        0.50 * prob +
        0.15 * (1.0 - unc) +
        0.15 * qed_val +
        0.10 * app_sim +
        0.10 * lipinski_bonus
    )

    return round(float(score), 4)


def build_3d_result_rows(
    smiles_list: Sequence[str],
    probabilities: Sequence[float],
    uncertainties: Optional[Sequence[float]] = None,
    infos: Optional[Sequence[Dict[str, object]]] = None,
    reference_fps=None,
    applicability_threshold: float = 0.35,
) -> List[Dict[str, object]]:
    """
    Build output rows for 3D GNN result table.
    """
    rows = []

    if uncertainties is None:
        uncertainties = [0.0] * len(smiles_list)

    if infos is None:
        infos = [{} for _ in smiles_list]

    n = min(len(smiles_list), len(probabilities), len(uncertainties), len(infos))

    for i in range(n):
        smi = str(smiles_list[i])
        prob = _safe_float(probabilities[i], default=0.0)
        unc = _safe_float(uncertainties[i], default=0.0)
        info = infos[i] or {}

        props = calculate_druglike_properties(smi)

        app_sim = 1.0
        if reference_fps:
            app_sim = max_tanimoto_similarity(smi, reference_fps)

        app_domain = applicability_domain_label(
            app_sim,
            inside_threshold=applicability_threshold,
        )

        priority = calculate_3d_priority_score(
            active_probability=prob,
            uncertainty=unc,
            qed=props.get("QED", 0.0),
            applicability_similarity=app_sim,
            lipinski_violations=props.get("LipinskiViolations", 0),
        )

        row = {
            "Rank": None,
            "SMILES": smi,
            "CanonicalSMILES": props.get("CanonicalSMILES") or info.get("CanonicalSMILES") or smi,
            "ActiveProbability": round(prob, 6),
            "Uncertainty": round(unc, 6),
            "ApplicabilitySimilarity": round(float(app_sim), 4),
            "ApplicabilityDomain": app_domain,
            "QED": props.get("QED"),
            "MW": props.get("MW"),
            "LogP": props.get("LogP"),
            "TPSA": props.get("TPSA"),
            "HBD": props.get("HBD"),
            "HBA": props.get("HBA"),
            "RotatableBonds": props.get("RotatableBonds"),
            "LipinskiViolations": props.get("LipinskiViolations"),
            "DrugLikeScore": props.get("DrugLikeScore"),
            "ConformerStatus": info.get("ConformerStatus", "NA"),
            "ConformerEnergy": info.get("ConformerEnergy", None),
            "Priority3DScore": priority,
        }

        rows.append(row)

    rows = sorted(
        rows,
        key=lambda r: (
            r.get("Priority3DScore", 0.0),
            r.get("ActiveProbability", 0.0),
        ),
        reverse=True,
    )

    for rank, row in enumerate(rows, start=1):
        row["Rank"] = rank

    return rows


def summarize_conformer_status(infos: Sequence[Dict[str, object]]) -> Dict[str, int]:
    """
    Count conformer generation statuses.
    """
    summary: Dict[str, int] = {}

    for info in infos:
        status = str((info or {}).get("ConformerStatus", "NA"))
        summary[status] = summary.get(status, 0) + 1

    return summary


# -------------------------------------------------------------------------
# Public re-exports for app.py convenience
# -------------------------------------------------------------------------

__all__ = [
    "ATOM_FEATURE_DIM",
    "BOND_FEATURE_DIM",
    "FEATURE_VECTOR_DIM",
    "NODE_FEATURE_DIM_3D",
    "EDGE_FEATURE_DIM_3D",
    "DEFAULT_RADIUS_CUTOFF",
    "generate_3d_conformer",
    "extract_coordinates",
    "mol_to_3d_graph",
    "smiles_to_3d_graph",
    "Molecule3DDataset",
    "collate_3d",
    "calculate_3d_priority_score",
    "build_3d_result_rows",
    "summarize_conformer_status",
    "build_reference_fingerprints",
    "max_tanimoto_similarity",
    "applicability_domain_label",
    "scaffold_split_indices",
]