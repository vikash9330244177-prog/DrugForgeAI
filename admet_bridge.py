"""
DrugForge AI ADMET bridge

This script reads an input CSV containing a smiles column, standardizes valid
SMILES, runs ADMET prediction, and writes:
1. main output CSV
2. summary JSON
3. top-hits CSV
"""

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from rdkit import Chem
except Exception:
    Chem = None

from admet_ai import ADMETModel


def find_smiles_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if str(col).strip().lower() == "smiles":
            return col
    raise ValueError('Input CSV must contain a "smiles" column.')


def standardize_smiles(smiles: str) -> Optional[str]:
    if smiles is None:
        return None

    s = str(smiles).strip()
    if not s:
        return None

    if Chem is None:
        return s

    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None

        # Keep largest fragment for salts / multi-fragment entries
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def normalize_predictions(preds) -> pd.DataFrame:
    """
    Convert ADMETModel output to a DataFrame while preserving row order.
    """
    if isinstance(preds, pd.DataFrame):
        preds_df = preds.copy()
    elif isinstance(preds, dict):
        try:
            preds_df = pd.DataFrame(preds)
        except Exception:
            preds_df = pd.DataFrame([preds])
    elif isinstance(preds, list):
        preds_df = pd.DataFrame(preds)
    else:
        raise TypeError(f"Unsupported prediction output type: {type(preds)}")

    preds_df = preds_df.reset_index(drop=False)

    # Handle common shapes from admet_ai
    if "index" in preds_df.columns and "smiles" not in preds_df.columns:
        idx_series = preds_df["index"]
        if idx_series.dtype == object:
            preds_df = preds_df.rename(columns={"index": "smiles"})
        else:
            preds_df = preds_df.drop(columns=["index"])

    return preds_df.reset_index(drop=True)


def maybe_add_summary_scores(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lightweight aggregate ADMET indicators without assuming exact schema.
    This stays conservative so it does not break when column names differ.
    """
    df = result_df.copy()
    lower_map = {c: str(c).strip().lower() for c in df.columns}

    favorable_cols = []
    unfavorable_cols = []

    for col, low in lower_map.items():
        if any(
            k in low
            for k in ["qed", "bbb", "hia", "f20", "f30", "oral", "bioavailability", "solubility"]
        ):
            favorable_cols.append(col)

        if any(
            k in low
            for k in ["tox", "herg", "ames", "dili", "cyp", "inhib", "clearance", "risk"]
        ):
            unfavorable_cols.append(col)

    favorable_num = []
    for col in favorable_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 0:
            favorable_num.append((col, s))

    unfavorable_num = []
    for col in unfavorable_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 0:
            unfavorable_num.append((col, s))

    if favorable_num:
        fav_stack = pd.concat([s.rename(col) for col, s in favorable_num], axis=1)
        df["admet_favorable_mean"] = fav_stack.mean(axis=1, skipna=True)

    if unfavorable_num:
        unfav_stack = pd.concat([s.rename(col) for col, s in unfavorable_num], axis=1)
        df["admet_risk_mean"] = unfav_stack.mean(axis=1, skipna=True)

    if "admet_favorable_mean" in df.columns and "admet_risk_mean" in df.columns:
        df["admet_priority_score"] = df["admet_favorable_mean"] - df["admet_risk_mean"]
    elif "admet_favorable_mean" in df.columns:
        df["admet_priority_score"] = df["admet_favorable_mean"]
    elif "admet_risk_mean" in df.columns:
        df["admet_priority_score"] = -df["admet_risk_mean"]

    return df


def build_summary(input_df: pd.DataFrame, valid_df: pd.DataFrame, result_df: pd.DataFrame) -> dict:
    summary = {
        "input_row_count": int(len(input_df)),
        "valid_smiles_count": int(len(valid_df)),
        "invalid_or_empty_smiles_count": int(len(input_df) - len(valid_df)),
        "output_row_count": int(len(result_df)),
        "prediction_column_count": int(max(0, len(result_df.columns) - len(valid_df.columns) + 1)),
        "columns": list(result_df.columns),
    }

    if "admet_priority_score" in result_df.columns:
        score = pd.to_numeric(result_df["admet_priority_score"], errors="coerce")
        if score.notna().sum() > 0:
            summary["admet_priority_score_mean"] = float(score.mean())
            summary["admet_priority_score_max"] = float(score.max())
            summary["admet_priority_score_min"] = float(score.min())

    return summary


def write_sidecar_files(output_csv: str, result_df: pd.DataFrame, summary: dict):
    out_path = Path(output_csv)
    summary_json = out_path.with_name(out_path.stem + "_summary.json")
    top_hits_csv = out_path.with_name(out_path.stem + "_top_hits.csv")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if "admet_priority_score" in result_df.columns:
        ranked = result_df.sort_values(
            "admet_priority_score",
            ascending=False,
            na_position="last"
        ).head(50)
    else:
        ranked = result_df.head(50)

    ranked.to_csv(top_hits_csv, index=False)
    return str(summary_json), str(top_hits_csv)


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python admet_bridge.py <input_csv> <output_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    smiles_col = find_smiles_column(df)

    base_df = df.copy()
    base_df.rename(columns={smiles_col: "smiles"}, inplace=True)
    base_df["smiles"] = base_df["smiles"].astype(str).str.strip()
    base_df["standardized_smiles"] = base_df["smiles"].apply(standardize_smiles)

    valid_df = base_df.dropna(subset=["standardized_smiles"]).copy().reset_index(drop=True)
    if valid_df.empty:
        raise ValueError("No valid SMILES values found in the input CSV after cleaning.")

    model = ADMETModel()
    preds = model.predict(smiles=valid_df["standardized_smiles"].tolist())
    preds_df = normalize_predictions(preds)

    preds_df = preds_df.drop(columns=["smiles", "standardized_smiles"], errors="ignore").reset_index(drop=True)

    if len(preds_df) != len(valid_df):
        common_n = min(len(preds_df), len(valid_df))
        valid_df = valid_df.iloc[:common_n].reset_index(drop=True)
        preds_df = preds_df.iloc[:common_n].reset_index(drop=True)

    result_df = pd.concat([valid_df.reset_index(drop=True), preds_df], axis=1)
    result_df = maybe_add_summary_scores(result_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    summary = build_summary(base_df, valid_df, result_df)
    summary_json, top_hits_csv = write_sidecar_files(str(output_path), result_df, summary)

    print(f"ADMET prediction completed successfully: {output_csv}")
    print(f"Summary: {summary_json}")
    print(f"Top hits: {top_hits_csv}")


if __name__ == "__main__":
    main()