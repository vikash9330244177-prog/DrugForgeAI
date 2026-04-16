"""
DrugForge AI protein-ligand interaction analysis bridge

This script:
1. Reads a protein-ligand complex PDB file
2. Detects ligand-like residues
3. Computes interaction proxies such as hydrogen bonds, hydrophobic contacts,
   ionic contacts, halogen bonds, pi-stacking, and close contacts
4. Writes CSV outputs, summary reports, and plots
"""

import argparse
import json
import os
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from Bio.PDB import PDBParser, is_aa


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL"
}

AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
POSITIVE_RESIDUES = {"LYS", "ARG", "HIS"}
NEGATIVE_RESIDUES = {"ASP", "GLU"}
WATER_AND_COMMON_SOLVENTS = {
    "HOH", "WAT", "DOD", "SO4", "PO4", "GOL", "EDO", "DMS",
    "PEG", "ACT", "ACE", "FMT", "CL", "NA", "K", "MG", "CA",
    "ZN", "MN", "IOD", "BR", "NO3"
}
HALOGENS = {"F", "CL", "BR", "I"}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_chain_id(chain):
    cid = str(chain.id).strip()
    return cid if cid else "-"


def residue_number(residue):
    return int(residue.id[1])


def residue_label(chain, residue):
    return f"{safe_chain_id(chain)}:{residue.resname}:{residue_number(residue)}"


def get_element(atom):
    el = str(getattr(atom, "element", "")).strip().upper()
    if el:
        return el

    atom_name = str(atom.get_name()).strip().upper()
    if atom_name:
        if atom_name[0].isdigit() and len(atom_name) > 1:
            return atom_name[1]
        if len(atom_name) >= 2 and atom_name[:2] in {"CL", "BR", "NA", "MG", "CA", "ZN", "FE", "MN", "CU"}:
            return atom_name[:2]
        return atom_name[0]
    return ""


def is_standard_protein_residue(residue):
    return residue.resname.strip().upper() in STANDARD_AA or is_aa(residue, standard=True)


def is_candidate_ligand_residue(residue):
    resname = residue.resname.strip().upper()
    hetflag = str(residue.id[0]).strip()

    if resname in WATER_AND_COMMON_SOLVENTS:
        return False
    if is_standard_protein_residue(residue):
        return False
    return hetflag != " "


def select_ligand_residues(structure, ligand_code=None):
    ligand_code = (ligand_code or "").strip().upper()
    candidate_residues = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if is_candidate_ligand_residue(residue):
                    candidate_residues.append((chain, residue))

    if not candidate_residues:
        raise ValueError("No ligand-like HETATM residue was found in the uploaded complex.")

    if ligand_code:
        matched = [
            (chain, residue)
            for chain, residue in candidate_residues
            if residue.resname.strip().upper() == ligand_code
        ]
        if not matched:
            raise ValueError(f'Ligand code "{ligand_code}" was not found in the uploaded complex.')
        return matched

    grouped = defaultdict(list)
    for chain, residue in candidate_residues:
        grouped[residue.resname.strip().upper()].append((chain, residue))

    best_resname = None
    best_score = -1
    for resname, residues in grouped.items():
        atom_count = sum(len(list(residue.get_atoms())) for _, residue in residues)
        if atom_count > best_score:
            best_score = atom_count
            best_resname = resname

    return grouped[best_resname]


def collect_protein_residues(structure):
    protein = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_standard_protein_residue(residue):
                    protein.append((chain, residue))

    if not protein:
        raise ValueError("No protein residues were found in the uploaded PDB file.")

    return protein


def ligand_has_pi_potential(ligand_residue):
    carbon_count = 0
    aromatic_atom_count = 0

    for atom in ligand_residue.get_atoms():
        el = get_element(atom)
        if el == "C":
            carbon_count += 1
        try:
            if atom.GetIsAromatic():
                aromatic_atom_count += 1
        except Exception:
            continue

    return carbon_count >= 6 or aromatic_atom_count >= 4


def shortest_atom_distance(residue_a, residue_b):
    min_dist = None
    min_pair = None

    for atom_a in residue_a.get_atoms():
        for atom_b in residue_b.get_atoms():
            try:
                dist = float(atom_a - atom_b)
            except Exception:
                continue

            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_pair = (atom_a, atom_b)

    return min_dist, min_pair


def atom_charge_hint(element):
    if element in {"N"}:
        return 1
    if element in {"O"}:
        return -1
    return 0


def analyze_interactions(
    protein_residues,
    ligand_residues,
    include_hydrophobic=True,
    include_pi=True
):
    interaction_rows = []
    dedupe = set()

    for p_chain, p_res in protein_residues:
        p_resname = p_res.resname.strip().upper()

        for l_chain, l_res in ligand_residues:
            min_dist, min_pair = shortest_atom_distance(p_res, l_res)
            if min_dist is None or min_pair is None:
                continue

            p_atom_short, l_atom_short = min_pair
            p_label = residue_label(p_chain, p_res)
            l_label = residue_label(l_chain, l_res)

            # Hydrogen-bond proxy
            for p_atom in p_res.get_atoms():
                p_el = get_element(p_atom)
                if p_el not in {"N", "O", "S"}:
                    continue

                for l_atom in l_res.get_atoms():
                    l_el = get_element(l_atom)
                    if l_el not in {"N", "O", "S"}:
                        continue

                    try:
                        dist = float(p_atom - l_atom)
                    except Exception:
                        continue

                    if dist <= 3.5:
                        key = (
                            "hydrogen_bond",
                            p_label,
                            str(p_atom.get_name()),
                            l_label,
                            str(l_atom.get_name()),
                        )
                        if key not in dedupe:
                            dedupe.add(key)
                            interaction_rows.append({
                                "interaction_type": "hydrogen_bond",
                                "protein_chain": safe_chain_id(p_chain),
                                "protein_residue_name": p_res.resname.strip(),
                                "protein_residue_id": residue_number(p_res),
                                "protein_atom": str(p_atom.get_name()),
                                "ligand_residue_name": l_res.resname.strip(),
                                "ligand_chain": safe_chain_id(l_chain),
                                "ligand_residue_id": residue_number(l_res),
                                "ligand_atom": str(l_atom.get_name()),
                                "distance_angstrom": round(dist, 3),
                            })

            # Ionic / salt bridge proxy
            if p_resname in POSITIVE_RESIDUES | NEGATIVE_RESIDUES:
                residue_sign = 1 if p_resname in POSITIVE_RESIDUES else -1

                for p_atom in p_res.get_atoms():
                    p_el = get_element(p_atom)
                    if p_el not in {"N", "O"}:
                        continue

                    for l_atom in l_res.get_atoms():
                        l_el = get_element(l_atom)
                        if l_el not in {"N", "O"}:
                            continue

                        try:
                            dist = float(p_atom - l_atom)
                        except Exception:
                            continue

                        ligand_sign = atom_charge_hint(l_el)
                        if residue_sign * ligand_sign < 0 and dist <= 4.0:
                            key = (
                                "ionic",
                                p_label,
                                str(p_atom.get_name()),
                                l_label,
                                str(l_atom.get_name()),
                            )
                            if key not in dedupe:
                                dedupe.add(key)
                                interaction_rows.append({
                                    "interaction_type": "ionic",
                                    "protein_chain": safe_chain_id(p_chain),
                                    "protein_residue_name": p_res.resname.strip(),
                                    "protein_residue_id": residue_number(p_res),
                                    "protein_atom": str(p_atom.get_name()),
                                    "ligand_residue_name": l_res.resname.strip(),
                                    "ligand_chain": safe_chain_id(l_chain),
                                    "ligand_residue_id": residue_number(l_res),
                                    "ligand_atom": str(l_atom.get_name()),
                                    "distance_angstrom": round(dist, 3),
                                })

            # Halogen bond proxy
            for l_atom in l_res.get_atoms():
                l_el = get_element(l_atom)
                if l_el not in HALOGENS:
                    continue

                for p_atom in p_res.get_atoms():
                    p_el = get_element(p_atom)
                    if p_el not in {"O", "N", "S"}:
                        continue

                    try:
                        dist = float(p_atom - l_atom)
                    except Exception:
                        continue

                    if dist <= 4.0:
                        key = (
                            "halogen_bond",
                            p_label,
                            str(p_atom.get_name()),
                            l_label,
                            str(l_atom.get_name()),
                        )
                        if key not in dedupe:
                            dedupe.add(key)
                            interaction_rows.append({
                                "interaction_type": "halogen_bond",
                                "protein_chain": safe_chain_id(p_chain),
                                "protein_residue_name": p_res.resname.strip(),
                                "protein_residue_id": residue_number(p_res),
                                "protein_atom": str(p_atom.get_name()),
                                "ligand_residue_name": l_res.resname.strip(),
                                "ligand_chain": safe_chain_id(l_chain),
                                "ligand_residue_id": residue_number(l_res),
                                "ligand_atom": str(l_atom.get_name()),
                                "distance_angstrom": round(dist, 3),
                            })

            # Hydrophobic contacts proxy
            if include_hydrophobic:
                for p_atom in p_res.get_atoms():
                    if get_element(p_atom) != "C":
                        continue

                    for l_atom in l_res.get_atoms():
                        if get_element(l_atom) != "C":
                            continue

                        try:
                            dist = float(p_atom - l_atom)
                        except Exception:
                            continue

                        if dist <= 4.5:
                            key = (
                                "hydrophobic",
                                p_label,
                                str(p_atom.get_name()),
                                l_label,
                                str(l_atom.get_name()),
                            )
                            if key not in dedupe:
                                dedupe.add(key)
                                interaction_rows.append({
                                    "interaction_type": "hydrophobic",
                                    "protein_chain": safe_chain_id(p_chain),
                                    "protein_residue_name": p_res.resname.strip(),
                                    "protein_residue_id": residue_number(p_res),
                                    "protein_atom": str(p_atom.get_name()),
                                    "ligand_residue_name": l_res.resname.strip(),
                                    "ligand_chain": safe_chain_id(l_chain),
                                    "ligand_residue_id": residue_number(l_res),
                                    "ligand_atom": str(l_atom.get_name()),
                                    "distance_angstrom": round(dist, 3),
                                })

            # Pi-stacking proxy
            if include_pi and p_resname in AROMATIC_RESIDUES and ligand_has_pi_potential(l_res):
                if min_dist <= 5.5:
                    key = ("pi_stacking", p_label, l_label)
                    if key not in dedupe:
                        dedupe.add(key)
                        interaction_rows.append({
                            "interaction_type": "pi_stacking",
                            "protein_chain": safe_chain_id(p_chain),
                            "protein_residue_name": p_res.resname.strip(),
                            "protein_residue_id": residue_number(p_res),
                            "protein_atom": "aromatic_residue",
                            "ligand_residue_name": l_res.resname.strip(),
                            "ligand_chain": safe_chain_id(l_chain),
                            "ligand_residue_id": residue_number(l_res),
                            "ligand_atom": "aromatic_region",
                            "distance_angstrom": round(min_dist, 3),
                        })

            # General close-contact fallback
            if min_dist <= 4.0:
                key = (
                    "close_contact",
                    p_label,
                    str(p_atom_short.get_name()),
                    l_label,
                    str(l_atom_short.get_name()),
                )
                if key not in dedupe:
                    dedupe.add(key)
                    interaction_rows.append({
                        "interaction_type": "close_contact",
                        "protein_chain": safe_chain_id(p_chain),
                        "protein_residue_name": p_res.resname.strip(),
                        "protein_residue_id": residue_number(p_res),
                        "protein_atom": str(p_atom_short.get_name()),
                        "ligand_residue_name": l_res.resname.strip(),
                        "ligand_chain": safe_chain_id(l_chain),
                        "ligand_residue_id": residue_number(l_res),
                        "ligand_atom": str(l_atom_short.get_name()),
                        "distance_angstrom": round(min_dist, 3),
                    })

    if not interaction_rows:
        raise ValueError("No protein–ligand interactions were detected from the uploaded complex.")

    interactions_df = pd.DataFrame(interaction_rows)
    interactions_df = interactions_df.sort_values(
        by=["interaction_type", "distance_angstrom", "protein_chain", "protein_residue_id"]
    ).reset_index(drop=True)

    return interactions_df


def build_counts_table(interactions_df):
    counts = interactions_df["interaction_type"].value_counts().reset_index()
    counts.columns = ["interaction_type", "count"]
    return counts


def build_residue_summary_table(interactions_df):
    residue_labels = interactions_df.apply(
        lambda row: f"{row['protein_chain']}:{row['protein_residue_name']}:{row['protein_residue_id']}",
        axis=1,
    )
    residue_counts = residue_labels.value_counts().reset_index()
    residue_counts.columns = ["protein_residue", "interaction_count"]
    return residue_counts


def build_distance_summary(interactions_df):
    if interactions_df.empty:
        return pd.DataFrame(columns=["interaction_type", "count", "mean_distance", "min_distance", "max_distance"])

    out = interactions_df.groupby("interaction_type")["distance_angstrom"].agg(["count", "mean", "min", "max"]).reset_index()
    out.columns = ["interaction_type", "count", "mean_distance", "min_distance", "max_distance"]
    return out


def build_summary(ligand_residues, interactions_df):
    ligand_names = sorted({residue.resname.strip() for _, residue in ligand_residues})
    ligand_chains = sorted({safe_chain_id(chain) for chain, _ in ligand_residues})
    ligand_ids = sorted({residue_number(residue) for _, residue in ligand_residues})

    counts = interactions_df["interaction_type"].value_counts().to_dict()

    residue_counter = Counter(
        [
            f"{row['protein_chain']}:{row['protein_residue_name']}:{row['protein_residue_id']}"
            for _, row in interactions_df.iterrows()
        ]
    )

    top_residues = [
        {"residue": residue, "count": count}
        for residue, count in residue_counter.most_common(15)
    ]

    return {
        "ligand_residue_names": ligand_names,
        "ligand_chains": ligand_chains,
        "ligand_residue_ids": ligand_ids,
        "total_interactions": int(len(interactions_df)),
        "interaction_type_counts": {k: int(v) for k, v in counts.items()},
        "unique_protein_residues_involved": int(len(set(residue_counter.keys()))),
        "mean_interaction_distance": round(float(interactions_df["distance_angstrom"].mean()), 3),
        "min_interaction_distance": round(float(interactions_df["distance_angstrom"].min()), 3),
        "top_contact_residues": top_residues,
    }


def save_reports(job_dir, summary, interactions_df, report_format="standard"):
    reports_dir = os.path.join(job_dir, "reports")
    ensure_dir(reports_dir)

    summary_path = os.path.join(reports_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_lines = [
        "Protein–Ligand Interaction Analysis Report",
        "=========================================",
        f"Ligand residue names: {', '.join(summary['ligand_residue_names'])}",
        f"Ligand chains: {', '.join(summary['ligand_chains'])}",
        f"Ligand residue IDs: {', '.join(str(x) for x in summary['ligand_residue_ids'])}",
        f"Total interactions: {summary['total_interactions']}",
        f"Unique protein residues involved: {summary['unique_protein_residues_involved']}",
        f"Mean interaction distance: {summary['mean_interaction_distance']} Å",
        f"Minimum interaction distance: {summary['min_interaction_distance']} Å",
        "",
        "Interaction counts:",
    ]

    for k, v in summary["interaction_type_counts"].items():
        report_lines.append(f"  - {k}: {v}")

    report_lines.append("")
    report_lines.append("Top contact residues:")

    for item in summary["top_contact_residues"][:10]:
        report_lines.append(f"  - {item['residue']}: {item['count']}")

    if str(report_format).strip().lower() == "detailed":
        report_lines.append("")
        report_lines.append("Detailed interaction preview:")
        preview_df = interactions_df.head(100)
        for _, row in preview_df.iterrows():
            report_lines.append(
                f"  - {row['interaction_type']} | "
                f"{row['protein_chain']}:{row['protein_residue_name']}:{row['protein_residue_id']}:{row['protein_atom']} -> "
                f"{row['ligand_chain']}:{row['ligand_residue_name']}:{row['ligand_residue_id']}:{row['ligand_atom']} | "
                f"{row['distance_angstrom']} Å"
            )

    report_path = os.path.join(reports_dir, "plip_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return summary_path, report_path


def save_output_tables(job_dir, interactions_df, counts_df, residue_df, distance_df):
    output_dir = os.path.join(job_dir, "output")
    ensure_dir(output_dir)

    interactions_path = os.path.join(output_dir, "plip_interactions.csv")
    counts_path = os.path.join(output_dir, "interaction_counts.csv")
    residue_path = os.path.join(output_dir, "residue_summary.csv")
    distance_path = os.path.join(output_dir, "distance_summary.csv")

    interactions_df.to_csv(interactions_path, index=False)
    counts_df.to_csv(counts_path, index=False)
    residue_df.to_csv(residue_path, index=False)
    distance_df.to_csv(distance_path, index=False)

    return interactions_path, counts_path, residue_path, distance_path


def save_plots(job_dir, interactions_df, counts_df, residue_df):
    plots_dir = os.path.join(job_dir, "plots")
    ensure_dir(plots_dir)

    barplot_path = os.path.join(plots_dir, "interaction_barplot.png")
    residue_plot_path = os.path.join(plots_dir, "residue_contact_map.png")
    distance_hist_path = os.path.join(plots_dir, "distance_histogram.png")
    interaction_distance_box_path = os.path.join(plots_dir, "interaction_distance_boxplot.png")

    if len(counts_df) > 0:
        plt.figure(figsize=(8, 5))
        plt.bar(counts_df["interaction_type"], counts_df["count"])
        plt.xlabel("Interaction Type")
        plt.ylabel("Count")
        plt.title("Interaction Type Distribution")
        plt.tight_layout()
        plt.savefig(barplot_path, dpi=300)
        plt.close()

    top_residues = residue_df.head(15)
    if len(top_residues) > 0:
        plt.figure(figsize=(12, 6))
        plt.bar(top_residues["protein_residue"], top_residues["interaction_count"])
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Protein Residues")
        plt.ylabel("Interaction Count")
        plt.title("Top Protein Residues Involved in Binding")
        plt.tight_layout()
        plt.savefig(residue_plot_path, dpi=300)
        plt.close()

    if len(interactions_df) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(interactions_df["distance_angstrom"].astype(float).values, bins=20)
        plt.xlabel("Distance (Å)")
        plt.ylabel("Count")
        plt.title("Interaction Distance Distribution")
        plt.tight_layout()
        plt.savefig(distance_hist_path, dpi=300)
        plt.close()

        types = interactions_df["interaction_type"].dropna().unique().tolist()
        data = [
            interactions_df.loc[interactions_df["interaction_type"] == t, "distance_angstrom"].astype(float).values
            for t in types
        ]
        data = [d for d in data if len(d) > 0]
        labels = [t for t in types if len(interactions_df.loc[interactions_df["interaction_type"] == t]) > 0]

        if data:
            plt.figure(figsize=(10, 6))
            plt.boxplot(data, labels=labels, vert=True)
            plt.xticks(rotation=30, ha="right")
            plt.ylabel("Distance (Å)")
            plt.title("Distance by Interaction Type")
            plt.tight_layout()
            plt.savefig(interaction_distance_box_path, dpi=300)
            plt.close()

    return barplot_path, residue_plot_path, distance_hist_path, interaction_distance_box_path


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--complex_file", required=True)
    parser.add_argument("--job_dir", required=True)
    parser.add_argument("--ligand_code", default="")
    parser.add_argument("--report_format", default="standard")
    parser.add_argument("--include_hydrophobic", default="yes")
    parser.add_argument("--include_pi", default="yes")
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

    if not os.path.exists(args.complex_file):
        raise FileNotFoundError(f"Complex file not found: {args.complex_file}")

    include_hydrophobic = str(args.include_hydrophobic).strip().lower() in {"yes", "true", "1"}
    include_pi = str(args.include_pi).strip().lower() in {"yes", "true", "1"}

    parser_obj = PDBParser(QUIET=True)
    structure = parser_obj.get_structure("complex", args.complex_file)

    ligand_residues = select_ligand_residues(structure, args.ligand_code)
    protein_residues = collect_protein_residues(structure)

    interactions_df = analyze_interactions(
        protein_residues=protein_residues,
        ligand_residues=ligand_residues,
        include_hydrophobic=include_hydrophobic,
        include_pi=include_pi,
    )

    counts_df = build_counts_table(interactions_df)
    residue_df = build_residue_summary_table(interactions_df)
    distance_df = build_distance_summary(interactions_df)
    summary = build_summary(ligand_residues, interactions_df)

    save_output_tables(args.job_dir, interactions_df, counts_df, residue_df, distance_df)
    save_reports(args.job_dir, summary, interactions_df, report_format=args.report_format)
    save_plots(args.job_dir, interactions_df, counts_df, residue_df)

    print("Protein–Ligand Interaction Analysis completed successfully.")


if __name__ == "__main__":
    main()