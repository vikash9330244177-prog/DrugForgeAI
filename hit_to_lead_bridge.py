"""
DrugForge AI Hit-to-Lead bridge

This script:
1. Reads an input CSV containing a smiles column
2. Standardizes and ranks candidate analogs
3. Applies optional reference-similarity and Lipinski-based filtering
4. Writes ranked outputs, summaries, and plots
"""

import argparse
import json
import os
import warnings
from typing import Optional

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
    HAS_STANDARDIZER = True
except Exception:
    HAS_STANDARDIZER = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_case_insensitive_column(columns, desired_name):
    for col in columns:
        if str(col).strip().lower() == str(desired_name).strip().lower():
            return col
    return None


def smiles_to_mol(smiles: str):
    if pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        return None


def _largest_fragment(mol):
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if frags:
            return max(frags, key=lambda m: m.GetNumHeavyAtoms())
    except Exception:
        return mol
    return mol


def standardize_mol(smiles: str):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    try:
        mol = _largest_fragment(mol)
        if HAS_STANDARDIZER:
            mol = rdMolStandardize.Cleanup(mol)
            parent = rdMolStandardize.FragmentParent(mol)
            if parent is not None:
                mol = parent
            try:
                uncharger = rdMolStandardize.Uncharger()
                mol = uncharger.uncharge(mol)
            except Exception:
                pass
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        return mol
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = standardize_mol(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def morgan_fp(mol, radius=2, n_bits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_similarity(fp1, fp2):
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def count_lipinski_violations(mw, logp, hbd, hba):
    violations = 0
    if mw > 500:
        violations += 1
    if logp > 5:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1
    return violations


def bounded_score(value, target, spread):
    try:
        value = float(value)
    except Exception:
        return 0.0
    score = 1.0 - abs(value - target) / float(spread)
    return max(0.0, min(1.0, score))


def synthetic_proxy_score(mw, logp, rot_bonds, rings, heavy_atoms):
    mw_score = bounded_score(mw, 350.0, 250.0)
    logp_score = bounded_score(logp, 2.5, 3.0)
    rot_score = bounded_score(rot_bonds, 4.0, 8.0)
    ring_score = bounded_score(rings, 3.0, 3.0)
    size_score = bounded_score(heavy_atoms, 28.0, 18.0)
    return float(np.mean([mw_score, logp_score, rot_score, ring_score, size_score]))


def compute_hit_to_lead_score(
    similarity,
    qed,
    mw,
    logp,
    tpsa,
    rot_bonds,
    fsp3,
    synthetic_proxy,
    has_reference
):
    sim_component = similarity if has_reference else 0.50
    mw_score = bounded_score(mw, 380.0, 220.0)
    logp_score = bounded_score(logp, 2.5, 2.5)
    tpsa_score = bounded_score(tpsa, 75.0, 75.0)
    rot_score = bounded_score(rot_bonds, 4.0, 6.0)
    fsp3_score = bounded_score(fsp3, 0.35, 0.35)

    if has_reference:
        score = (
            0.35 * sim_component +
            0.22 * qed +
            0.12 * mw_score +
            0.08 * logp_score +
            0.06 * tpsa_score +
            0.05 * rot_score +
            0.05 * fsp3_score +
            0.07 * synthetic_proxy
        )
    else:
        score = (
            0.28 * qed +
            0.18 * mw_score +
            0.15 * logp_score +
            0.10 * tpsa_score +
            0.10 * rot_score +
            0.09 * fsp3_score +
            0.10 * synthetic_proxy
        )

    return round(float(score), 4)


def compute_priority_bucket(score):
    if score >= 0.80:
        return "High"
    if score >= 0.65:
        return "Medium"
    return "Low"


def get_bemis_murcko_scaffold(mol):
    try:
        return rdMolDescriptors.CalcMurckoScaffoldSmiles(mol)
    except Exception:
        return ""


def normalize_input_dataframe(df):
    smiles_col = resolve_case_insensitive_column(df.columns, "smiles")
    if smiles_col is None:
        raise ValueError('Input CSV must contain a column named "smiles".')

    out = df.copy()
    if smiles_col != "smiles":
        out = out.rename(columns={smiles_col: "smiles"})

    out["canonical_smiles"] = out["smiles"].apply(canonicalize_smiles)
    out = out.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    out = out.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)

    if out.empty:
        raise ValueError("No valid SMILES remain after cleaning.")

    return out


def compute_candidate_table(df, reference_smiles=None):
    reference_fp = None
    has_reference = False

    if reference_smiles:
        ref_mol = standardize_mol(reference_smiles)
        if ref_mol is not None:
            reference_fp = morgan_fp(ref_mol)
            has_reference = True

    records = []

    for idx, row in df.iterrows():
        smi = row["canonical_smiles"]
        mol = standardize_mol(smi)
        if mol is None:
            continue

        fp = morgan_fp(mol)

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)
        rings = Lipinski.RingCount(mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        qed = float(QED.qed(mol))
        fraction_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
        formal_charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
        lipinski_violations = count_lipinski_violations(mw, logp, hbd, hba)
        passed_lipinski = lipinski_violations <= 1
        synthetic_proxy = synthetic_proxy_score(mw, logp, rot_bonds, rings, heavy_atoms)
        scaffold = get_bemis_murcko_scaffold(mol)

        similarity = np.nan
        if has_reference:
            similarity = tanimoto_similarity(fp, reference_fp)

        htl_score = compute_hit_to_lead_score(
            similarity=similarity if not np.isnan(similarity) else 0.0,
            qed=qed,
            mw=mw,
            logp=logp,
            tpsa=tpsa,
            rot_bonds=rot_bonds,
            fsp3=fraction_csp3,
            synthetic_proxy=synthetic_proxy,
            has_reference=has_reference,
        )

        record = {
            "original_index": idx,
            "smiles": row["smiles"],
            "canonical_smiles": smi,
            "molecular_weight": round(float(mw), 3),
            "logp": round(float(logp), 3),
            "hbond_donors": int(hbd),
            "hbond_acceptors": int(hba),
            "tpsa": round(float(tpsa), 3),
            "rotatable_bonds": int(rot_bonds),
            "ring_count": int(rings),
            "aromatic_ring_count": int(aromatic_rings),
            "heavy_atom_count": int(heavy_atoms),
            "fraction_csp3": round(float(fraction_csp3), 4),
            "formal_charge": int(formal_charge),
            "qed": round(float(qed), 4),
            "synthetic_proxy": round(float(synthetic_proxy), 4),
            "lipinski_violations": int(lipinski_violations),
            "passed_lipinski": bool(passed_lipinski),
            "similarity_to_reference": round(float(similarity), 4) if not np.isnan(similarity) else np.nan,
            "scaffold": scaffold,
            "hit_to_lead_score": round(float(htl_score), 4),
            "priority_bucket": compute_priority_bucket(htl_score),
        }

        for col in df.columns:
            if col not in record and col != "canonical_smiles":
                record[col] = row[col]

        records.append(record)

    if not records:
        raise ValueError("No valid candidate molecules could be processed.")

    out_df = pd.DataFrame(records)
    return out_df, has_reference


def _sort_ranked_source(df):
    sort_cols = ["hit_to_lead_score", "qed"]
    ascending = [False, False]
    if "similarity_to_reference" in df.columns:
        sort_cols.append("similarity_to_reference")
        ascending.append(False)
    sort_cols += ["synthetic_proxy", "fraction_csp3"]
    ascending += [False, False]
    return df.sort_values(by=sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)


def select_diverse_followups(filtered_df, max_analogs=20, diversity_cutoff=0.85):
    if max_analogs <= 0 or filtered_df.empty:
        return filtered_df.iloc[0:0].copy()

    work_df = filtered_df.copy().reset_index(drop=True)
    fps = []
    for smi in work_df["canonical_smiles"].tolist():
        mol = standardize_mol(smi)
        fps.append(morgan_fp(mol) if mol is not None else None)

    selected_indices = []
    selected_scaffolds = set()

    for idx, row in work_df.iterrows():
        if len(selected_indices) >= int(max_analogs):
            break

        fp = fps[idx]
        if fp is None:
            continue

        scaffold = row.get("scaffold", "") or ""
        too_similar = False
        for s_idx in selected_indices:
            s_fp = fps[s_idx]
            if s_fp is None:
                continue
            if tanimoto_similarity(fp, s_fp) >= float(diversity_cutoff):
                too_similar = True
                break
        if too_similar:
            continue

        if scaffold and scaffold in selected_scaffolds and len(selected_indices) < max(3, int(max_analogs * 0.4)):
            continue

        selected_indices.append(idx)
        if scaffold:
            selected_scaffolds.add(scaffold)

    if len(selected_indices) < min(len(work_df), int(max_analogs)):
        for idx, _row in work_df.iterrows():
            if idx in selected_indices:
                continue
            if len(selected_indices) >= int(max_analogs):
                break
            selected_indices.append(idx)

    return work_df.iloc[selected_indices].copy().reset_index(drop=True)


def rank_and_filter_candidates(
    df,
    has_reference,
    similarity_threshold=0.60,
    top_n=50,
    max_analogs=20,
    apply_lipinski="yes"
):
    apply_lipinski_flag = str(apply_lipinski).strip().lower() in {"yes", "true", "1"}

    ranked_source = df.copy()

    if has_reference:
        ranked_source["similarity_pass"] = ranked_source["similarity_to_reference"] >= float(similarity_threshold)
    else:
        ranked_source["similarity_pass"] = True

    ranked_source["lipinski_pass"] = ranked_source["passed_lipinski"]
    ranked_source["basic_quality_pass"] = (
        ranked_source["formal_charge"].between(-2, 2) &
        ranked_source["molecular_weight"].between(120, 650) &
        ranked_source["rotatable_bonds"].le(12)
    )

    mask = ranked_source["similarity_pass"] & ranked_source["basic_quality_pass"]
    if apply_lipinski_flag:
        mask = mask & ranked_source["lipinski_pass"]

    filtered_df = ranked_source[mask].copy()

    fallback_used = False
    if filtered_df.empty:
        fallback_used = True
        filtered_df = ranked_source.copy()

    filtered_df = _sort_ranked_source(filtered_df)

    if top_n > 0:
        filtered_df = filtered_df.head(int(top_n)).copy()

    filtered_df["lead_rank"] = np.arange(1, len(filtered_df) + 1)
    filtered_df["selected_for_followup"] = False

    final_df = select_diverse_followups(filtered_df, max_analogs=max_analogs, diversity_cutoff=0.85)

    if len(final_df) > 0:
        filtered_df.loc[
            filtered_df["canonical_smiles"].isin(final_df["canonical_smiles"]),
            "selected_for_followup"
        ] = True
        final_df = final_df.copy().reset_index(drop=True)
        final_df["followup_rank"] = np.arange(1, len(final_df) + 1)

    return filtered_df, final_df, apply_lipinski_flag, fallback_used


def save_summary(
    job_dir,
    input_count,
    valid_count,
    ranked_df,
    final_df,
    has_reference,
    similarity_threshold,
    apply_lipinski_flag,
    fallback_used
):
    reports_dir = os.path.join(job_dir, "reports")

    top_score = None
    mean_score = None
    scaffold_count = 0
    if len(ranked_df) > 0:
        top_score = round(float(ranked_df["hit_to_lead_score"].max()), 4)
        mean_score = round(float(ranked_df["hit_to_lead_score"].mean()), 4)
        scaffold_count = int(ranked_df["scaffold"].fillna("").replace("", np.nan).nunique(dropna=True))

    summary = {
        "input_molecule_count": int(input_count),
        "valid_molecule_count": int(valid_count),
        "ranked_candidate_count": int(len(ranked_df)),
        "selected_followup_count": int(len(final_df)),
        "unique_scaffold_count": scaffold_count,
        "reference_used": bool(has_reference),
        "similarity_threshold": float(similarity_threshold) if has_reference else None,
        "lipinski_filter_applied": bool(apply_lipinski_flag),
        "fallback_used": bool(fallback_used),
        "top_hit_to_lead_score": top_score,
        "mean_hit_to_lead_score": mean_score,
    }

    with open(os.path.join(reports_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "Hit-to-Lead Optimization Summary",
        "================================",
        f"Input molecules: {summary['input_molecule_count']}",
        f"Valid molecules: {summary['valid_molecule_count']}",
        f"Ranked candidates: {summary['ranked_candidate_count']}",
        f"Selected follow-up candidates: {summary['selected_followup_count']}",
        f"Unique scaffolds: {summary['unique_scaffold_count']}",
        f"Reference used: {summary['reference_used']}",
        f"Similarity threshold: {summary['similarity_threshold']}",
        f"Lipinski filter applied: {summary['lipinski_filter_applied']}",
        f"Fallback used: {summary['fallback_used']}",
        f"Top score: {summary['top_hit_to_lead_score']}",
        f"Mean score: {summary['mean_hit_to_lead_score']}",
    ]

    with open(os.path.join(reports_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return summary


def save_output_tables(job_dir, ranked_df, final_df):
    output_dir = os.path.join(job_dir, "output")

    ranked_path = os.path.join(output_dir, "ranked_analogs.csv")
    final_path = os.path.join(output_dir, "optimized_hits.csv")
    followup_summary_path = os.path.join(output_dir, "followup_summary.csv")

    ranked_df.to_csv(ranked_path, index=False)
    final_df.to_csv(final_path, index=False)

    followup_cols = [
        c for c in [
            "followup_rank", "lead_rank", "canonical_smiles", "hit_to_lead_score", "qed",
            "similarity_to_reference", "molecular_weight", "logp", "tpsa",
            "rotatable_bonds", "fraction_csp3", "priority_bucket", "scaffold"
        ] if c in final_df.columns
    ]
    final_df[followup_cols].to_csv(followup_summary_path, index=False)

    return ranked_path, final_path, followup_summary_path


def save_plots(job_dir, ranked_df):
    plots_dir = os.path.join(job_dir, "plots")

    ranking_plot_path = os.path.join(plots_dir, "ranking_plot.png")
    property_plot_path = os.path.join(plots_dir, "property_distribution.png")
    score_hist_path = os.path.join(plots_dir, "score_histogram.png")
    sim_hist_path = os.path.join(plots_dir, "similarity_histogram.png")
    qed_sim_path = os.path.join(plots_dir, "qed_vs_similarity.png")

    top_plot_df = ranked_df.head(20).copy()

    if len(top_plot_df) > 0:
        plt.figure(figsize=(12, 6))
        labels = [f"Lead_{i}" for i in top_plot_df["lead_rank"].tolist()]
        colors = [
            "tab:red" if x == "High"
            else "tab:orange" if x == "Medium"
            else "tab:blue"
            for x in top_plot_df["priority_bucket"]
        ]
        plt.bar(labels, top_plot_df["hit_to_lead_score"].tolist(), color=colors)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Hit-to-Lead Score")
        plt.xlabel("Top Ranked Candidates")
        plt.title("Top Hit-to-Lead Ranked Candidates")
        plt.tight_layout()
        plt.savefig(ranking_plot_path, dpi=300)
        plt.close()

    if len(ranked_df) > 0:
        plt.figure(figsize=(8, 6))
        plt.scatter(
            ranked_df["molecular_weight"].astype(float).tolist(),
            ranked_df["logp"].astype(float).tolist(),
            alpha=0.7
        )
        plt.axvline(500, linestyle="--", linewidth=1)
        plt.axhline(5, linestyle="--", linewidth=1)
        plt.xlabel("Molecular Weight")
        plt.ylabel("logP")
        plt.title("Candidate Property Distribution")
        plt.tight_layout()
        plt.savefig(property_plot_path, dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.hist(ranked_df["hit_to_lead_score"].astype(float), bins=20)
        plt.xlabel("Hit-to-Lead Score")
        plt.ylabel("Count")
        plt.title("Score Distribution")
        plt.tight_layout()
        plt.savefig(score_hist_path, dpi=300)
        plt.close()

        if ranked_df["similarity_to_reference"].notna().any():
            plt.figure(figsize=(8, 5))
            plt.hist(ranked_df["similarity_to_reference"].dropna().astype(float), bins=20)
            plt.xlabel("Similarity to Reference")
            plt.ylabel("Count")
            plt.title("Similarity Distribution")
            plt.tight_layout()
            plt.savefig(sim_hist_path, dpi=300)
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.scatter(
                ranked_df["similarity_to_reference"].astype(float),
                ranked_df["qed"].astype(float),
                alpha=0.7
            )
            plt.xlabel("Similarity to Reference")
            plt.ylabel("QED")
            plt.title("QED vs Similarity")
            plt.tight_layout()
            plt.savefig(qed_sim_path, dpi=300)
            plt.close()

    return {
        "ranking_plot": ranking_plot_path,
        "property_distribution": property_plot_path,
        "score_histogram": score_hist_path,
        "similarity_histogram": sim_hist_path,
        "qed_vs_similarity": qed_sim_path,
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--job_dir", required=True)
    parser.add_argument("--reference_smiles", default="")
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--similarity_threshold", type=float, default=0.60)
    parser.add_argument("--max_analogs", type=int, default=20)
    parser.add_argument("--apply_lipinski", default="yes")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    ensure_dir(args.job_dir)
    ensure_dir(os.path.join(args.job_dir, "input"))
    ensure_dir(os.path.join(args.job_dir, "output"))
    ensure_dir(os.path.join(args.job_dir, "plots"))
    ensure_dir(os.path.join(args.job_dir, "reports"))
    ensure_dir(os.path.join(args.job_dir, "temp"))

    input_df = pd.read_csv(args.input_csv)
    input_count = len(input_df)

    normalized_df = normalize_input_dataframe(input_df)
    valid_count = len(normalized_df)

    candidate_df, has_reference = compute_candidate_table(
        normalized_df,
        reference_smiles=args.reference_smiles.strip()
    )

    ranked_df, final_df, apply_lipinski_flag, fallback_used = rank_and_filter_candidates(
        candidate_df,
        has_reference=has_reference,
        similarity_threshold=args.similarity_threshold,
        top_n=args.top_n,
        max_analogs=args.max_analogs,
        apply_lipinski=args.apply_lipinski
    )

    save_output_tables(args.job_dir, ranked_df, final_df)
    save_summary(
        job_dir=args.job_dir,
        input_count=input_count,
        valid_count=valid_count,
        ranked_df=ranked_df,
        final_df=final_df,
        has_reference=has_reference,
        similarity_threshold=args.similarity_threshold,
        apply_lipinski_flag=apply_lipinski_flag,
        fallback_used=fallback_used
    )
    save_plots(args.job_dir, ranked_df)

    print("Hit-to-Lead optimization completed successfully.")


if __name__ == "__main__":
    main()