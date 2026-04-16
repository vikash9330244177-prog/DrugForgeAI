"""
DrugForge AI QSAR bridge

This script supports:
1. QSAR model training
2. External QSAR prediction using a saved model

Supported model types:
- rf
- svm
- graphconv
- mpnn
- ecfp_rf
- ecfp_svm
"""

import argparse
import json
import math
import os
import random
import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_case_insensitive_column(columns, desired_name):
    for col in columns:
        if str(col).strip().lower() == str(desired_name).strip().lower():
            return col
    return None


def load_csv(path):
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Input CSV is empty.")
    return df


def smiles_to_mol(smiles):
    if pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        return None


def standardize_smiles(smiles):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    except Exception:
        return None

    try:
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        return None

    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def canonicalize_smiles(smiles):
    return standardize_smiles(smiles)


def safe_descriptor(mol, fn, default=np.nan):
    try:
        return fn(mol)
    except Exception:
        return default


def get_scaffold(smiles):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold if scaffold else None
    except Exception:
        return None


def add_basic_descriptors(df):
    out = df.copy()
    mols = [smiles_to_mol(s) for s in out["smiles"].tolist()]
    out["molecular_weight"] = [safe_descriptor(m, Descriptors.MolWt) if m is not None else np.nan for m in mols]
    out["logp"] = [safe_descriptor(m, Descriptors.MolLogP) if m is not None else np.nan for m in mols]
    out["tpsa"] = [safe_descriptor(m, Descriptors.TPSA) if m is not None else np.nan for m in mols]
    out["num_h_donors"] = [safe_descriptor(m, Descriptors.NumHDonors) if m is not None else np.nan for m in mols]
    out["num_h_acceptors"] = [safe_descriptor(m, Descriptors.NumHAcceptors) if m is not None else np.nan for m in mols]
    out["scaffold"] = [get_scaffold(s) for s in out["smiles"].tolist()]
    return out


def ecfp_features_from_smiles(smiles_list, radius=2, n_bits=2048):
    features = []
    valid_indices = []
    fps = []

    for idx, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        features.append(arr)
        valid_indices.append(idx)
        fps.append(fp)

    if not features:
        raise ValueError("No valid SMILES could be featurized.")

    return np.array(features, dtype=np.float32), valid_indices, fps


def scaffold_split(df, test_size=0.2):
    scaffold_to_indices = {}

    for idx, smi in enumerate(df["smiles"].tolist()):
        scaffold = get_scaffold(smi)
        if scaffold is None:
            scaffold = f"no_scaffold_{idx}"
        scaffold_to_indices.setdefault(scaffold, []).append(idx)

    scaffold_sets = sorted(
        scaffold_to_indices.values(),
        key=lambda x: (len(x), x[0]),
        reverse=True
    )

    n_total = len(df)
    n_test_target = max(1, int(round(n_total * test_size)))

    test_indices = []
    train_indices = []

    for group in scaffold_sets:
        if len(test_indices) + len(group) <= n_test_target:
            test_indices.extend(group)
        else:
            train_indices.extend(group)

    if len(test_indices) == 0:
        test_indices = train_indices[:n_test_target]
        train_indices = train_indices[n_test_target:]

    if len(train_indices) == 0 and len(test_indices) > 1:
        train_indices = test_indices[:-1]
        test_indices = test_indices[-1:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    return train_df, test_df


def split_dataframe(df, split_type="random", test_size=0.2, random_seed=42, task_type="classification"):
    if split_type == "scaffold":
        return scaffold_split(df, test_size=test_size)

    if task_type == "classification":
        y = df["_target_encoded"]
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed,
            stratify=y if len(pd.Series(y).unique()) > 1 else None,
        )

    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
    )


def normalize_training_dataframe(df, target_column, task_type):
    smiles_col = resolve_case_insensitive_column(df.columns, "smiles")
    if smiles_col is None:
        raise ValueError('Training CSV must contain a column named "smiles".')

    target_col = resolve_case_insensitive_column(df.columns, target_column)
    if target_col is None:
        raise ValueError(f'Training CSV must contain target column "{target_column}".')

    out = df.copy()
    if smiles_col != "smiles":
        out = out.rename(columns={smiles_col: "smiles"})
    if target_col != target_column:
        out = out.rename(columns={target_col: target_column})

    out["smiles"] = out["smiles"].apply(canonicalize_smiles)
    out = out.dropna(subset=["smiles", target_column]).reset_index(drop=True)
    out = out.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    if out.empty:
        raise ValueError("No valid rows remain after SMILES cleaning.")

    out = add_basic_descriptors(out)

    if task_type == "classification":
        classes = sorted(out[target_column].astype(str).unique().tolist())
        if len(classes) < 2:
            raise ValueError("Classification requires at least two target classes.")
        class_to_int = {c: i for i, c in enumerate(classes)}
        out["_target_encoded"] = out[target_column].astype(str).map(class_to_int).astype(int)
        metadata = {"class_mapping": class_to_int}
    else:
        out["_target_encoded"] = pd.to_numeric(out[target_column], errors="coerce")
        out = out.dropna(subset=["_target_encoded"]).reset_index(drop=True)
        if out.empty:
            raise ValueError("Regression target column could not be converted to numeric values.")
        metadata = {"class_mapping": None}

    return out, metadata


def normalize_prediction_dataframe(df):
    smiles_col = resolve_case_insensitive_column(df.columns, "smiles")
    if smiles_col is None:
        raise ValueError('Prediction CSV must contain a column named "smiles".')

    out = df.copy()
    if smiles_col != "smiles":
        out = out.rename(columns={smiles_col: "smiles"})

    out["smiles"] = out["smiles"].apply(canonicalize_smiles)
    out = out.dropna(subset=["smiles"]).reset_index(drop=True)
    out = out.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    if out.empty:
        raise ValueError("No valid SMILES remain in prediction CSV.")

    out = add_basic_descriptors(out)
    return out


def canonicalize_model_type(model_type):
    model_type = str(model_type).strip().lower()

    mapping = {
        "rf": "rf",
        "svm": "svm",
        "graphconv": "graphconv",
        "mpnn": "mpnn",
        "ecfp_rf": "rf",
        "ecfp_svm": "svm",
    }

    if model_type not in mapping:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return mapping[model_type]


def save_training_smiles(model_dir, smiles_series):
    path = os.path.join(model_dir, "training_smiles.csv")
    pd.DataFrame({"smiles": smiles_series}).to_csv(path, index=False)
    return path


def compute_applicability_for_smiles(
    smiles_list,
    training_smiles,
    radius=2,
    n_bits=2048,
    similarity_threshold=0.35
):
    if len(training_smiles) == 0:
        return pd.DataFrame({
            "max_similarity_to_training": [np.nan] * len(smiles_list),
            "applicability_domain": ["unknown"] * len(smiles_list),
        })

    train_fps = []
    for smi in training_smiles:
        mol = smiles_to_mol(smi)
        if mol is None:
            continue
        train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))

    max_sims = []
    domain = []
    for smi in smiles_list:
        mol = smiles_to_mol(smi)
        if mol is None or len(train_fps) == 0:
            max_sims.append(np.nan)
            domain.append("unknown")
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        max_sim = max(sims) if sims else np.nan
        max_sims.append(float(max_sim))
        domain.append("inside" if max_sim >= similarity_threshold else "outside")

    return pd.DataFrame({
        "max_similarity_to_training": max_sims,
        "applicability_domain": domain,
    })


def train_sklearn_model(train_df, test_df, model_type, task_type, model_dir, random_seed=42):
    X_train, valid_train_idx, _ = ecfp_features_from_smiles(train_df["smiles"].tolist())
    train_df = train_df.iloc[valid_train_idx].reset_index(drop=True)

    X_test, valid_test_idx, _ = ecfp_features_from_smiles(test_df["smiles"].tolist())
    test_df = test_df.iloc[valid_test_idx].reset_index(drop=True)

    y_train = train_df["_target_encoded"].values
    y_test = test_df["_target_encoded"].values

    if task_type == "classification":
        if model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=400,
                random_state=random_seed,
                n_jobs=-1,
                class_weight="balanced",
            )
        elif model_type == "svm":
            model = SVC(
                probability=True,
                random_state=random_seed,
                class_weight="balanced",
            )
        else:
            raise ValueError(f"Unsupported sklearn classification model type: {model_type}")
    else:
        if model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=400,
                random_state=random_seed,
                n_jobs=-1,
            )
        elif model_type == "svm":
            model = SVR()
        else:
            raise ValueError(f"Unsupported sklearn regression model type: {model_type}")

    model.fit(X_train, y_train)

    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)

    if task_type == "classification":
        train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") and len(np.unique(y_train)) == 2 else None
        test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") and len(np.unique(y_train)) == 2 else None

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["predicted_class"] = train_pred
        test_df["predicted_class"] = test_pred

        if train_proba is not None:
            train_df["predicted_probability"] = train_proba
        if test_proba is not None:
            test_df["predicted_probability"] = test_proba

        return model, train_df, test_df

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["predicted_value"] = train_pred
    test_df["predicted_value"] = test_pred

    return model, train_df, test_df


def train_graphconv_model(train_df, test_df, task_type, model_dir, epochs=20, batch_size=32):
    try:
        import deepchem as dc
    except ImportError:
        raise RuntimeError("DeepChem is not installed in the QSAR environment.")

    featurizer = dc.feat.ConvMolFeaturizer()

    X_train = featurizer.featurize(train_df["smiles"].tolist())
    X_test = featurizer.featurize(test_df["smiles"].tolist())

    y_train = train_df["_target_encoded"].values.reshape(-1, 1)
    y_test = test_df["_target_encoded"].values.reshape(-1, 1)

    train_dataset = dc.data.NumpyDataset(X=X_train, y=y_train, ids=train_df["smiles"].tolist())
    test_dataset = dc.data.NumpyDataset(X=X_test, y=y_test, ids=test_df["smiles"].tolist())

    if task_type == "classification":
        model = dc.models.GraphConvModel(
            n_tasks=1,
            mode="classification",
            batch_size=batch_size,
            model_dir=model_dir,
        )
    else:
        model = dc.models.GraphConvModel(
            n_tasks=1,
            mode="regression",
            batch_size=batch_size,
            model_dir=model_dir,
        )

    model.fit(train_dataset, nb_epoch=epochs)
    model.save_checkpoint()

    if task_type == "classification":
        train_pred_raw = model.predict(train_dataset)
        test_pred_raw = model.predict(test_dataset)

        train_proba = train_pred_raw[:, 0, 1]
        test_proba = test_pred_raw[:, 0, 1]

        train_pred = (train_proba >= 0.5).astype(int)
        test_pred = (test_proba >= 0.5).astype(int)

        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["predicted_class"] = train_pred
        test_df["predicted_class"] = test_pred
        train_df["predicted_probability"] = train_proba
        test_df["predicted_probability"] = test_proba

        return model, train_df, test_df

    train_pred = model.predict(train_dataset).reshape(-1)
    test_pred = model.predict(test_dataset).reshape(-1)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["predicted_value"] = train_pred
    test_df["predicted_value"] = test_pred

    return model, train_df, test_df


def train_mpnn_model(train_df, test_df, task_type, model_dir, epochs=20, batch_size=32):
    try:
        import deepchem as dc
        from deepchem.models.torch_models import MPNNModel
    except ImportError:
        raise RuntimeError("DeepChem or MPNN dependencies are not installed in the QSAR environment.")

    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    X_train = featurizer.featurize(train_df["smiles"].tolist())
    X_test = featurizer.featurize(test_df["smiles"].tolist())

    y_train = train_df["_target_encoded"].values.reshape(-1, 1)
    y_test = test_df["_target_encoded"].values.reshape(-1, 1)

    train_dataset = dc.data.NumpyDataset(X=X_train, y=y_train, ids=train_df["smiles"].tolist())
    test_dataset = dc.data.NumpyDataset(X=X_test, y=y_test, ids=test_df["smiles"].tolist())

    if task_type == "classification":
        model = MPNNModel(
            n_tasks=1,
            mode="classification",
            batch_size=batch_size,
            model_dir=model_dir,
        )
    else:
        model = MPNNModel(
            n_tasks=1,
            mode="regression",
            batch_size=batch_size,
            model_dir=model_dir,
        )

    model.fit(train_dataset, nb_epoch=epochs)

    if task_type == "classification":
        train_pred_raw = model.predict(train_dataset)
        test_pred_raw = model.predict(test_dataset)

        train_proba = train_pred_raw[:, 0, 1]
        test_proba = test_pred_raw[:, 0, 1]

        train_pred = (train_proba >= 0.5).astype(int)
        test_pred = (test_proba >= 0.5).astype(int)

        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["predicted_class"] = train_pred
        test_df["predicted_class"] = test_pred
        train_df["predicted_probability"] = train_proba
        test_df["predicted_probability"] = test_proba

        return model, train_df, test_df

    train_pred = model.predict(train_dataset).reshape(-1)
    test_pred = model.predict(test_dataset).reshape(-1)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["predicted_value"] = train_pred
    test_df["predicted_value"] = test_pred

    return model, train_df, test_df


def classification_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    unique_classes = np.unique(y_true)
    if y_prob is not None and len(unique_classes) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = None
        try:
            metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["average_precision"] = None
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return {
        "r2_score": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
    }


def save_metrics_plot(metrics_dict, out_path, title):
    keys = []
    values = []

    for k, v in metrics_dict.items():
        if v is None:
            continue
        keys.append(k)
        values.append(v)

    if not keys:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(keys, values)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_confusion_matrix_plot(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_roc_curve_plot(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_pr_curve_plot(y_true, y_prob, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_regression_scatter_plot(y_true, y_pred, out_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_feature_importance_plot(model, out_path, top_n=20):
    if not hasattr(model, "feature_importances_"):
        return

    importances = np.asarray(model.feature_importances_)
    if importances.ndim != 1 or len(importances) == 0:
        return

    order = np.argsort(importances)[::-1][:top_n]
    vals = importances[order]
    labels = [f"FP_{i}" for i in order]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("Top Fingerprint Feature Importances")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_metadata(model_dir, metadata):
    with open(os.path.join(model_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(model_dir):
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model metadata file not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def invert_class_mapping(class_mapping):
    if not class_mapping:
        return None
    return {int(v): k for k, v in class_mapping.items()}


def save_training_outputs(job_dir, train_df, test_df):
    train_out = train_df.copy()
    test_out = test_df.copy()

    if "_target_encoded" in train_out.columns:
        train_out = train_out.rename(columns={"_target_encoded": "target_encoded"})
    if "_target_encoded" in test_out.columns:
        test_out = test_out.rename(columns={"_target_encoded": "target_encoded"})

    train_path = os.path.join(job_dir, "train_predictions.csv")
    test_path = os.path.join(job_dir, "test_predictions.csv")
    all_path = os.path.join(job_dir, "all_predictions.csv")

    train_out["split"] = "train"
    test_out["split"] = "test"

    combined = pd.concat([train_out, test_out], ignore_index=True)

    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)
    combined.to_csv(all_path, index=False)

    return train_path, test_path, all_path


def build_training_summary(train_df, test_df, metadata, metrics):
    return {
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "unique_train_scaffolds": int(pd.Series(train_df.get("scaffold")).dropna().nunique()) if "scaffold" in train_df.columns else None,
        "unique_test_scaffolds": int(pd.Series(test_df.get("scaffold")).dropna().nunique()) if "scaffold" in test_df.columns else None,
        "task_type": metadata["task_type"],
        "model_type": metadata["model_type"],
        "split_type": metadata["split_type"],
        "metrics": metrics,
    }


def run_train(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    ensure_dir(args.job_dir)
    ensure_dir(args.model_dir)
    ensure_dir(os.path.join(args.job_dir, "plots"))

    df = load_csv(args.input_csv)
    df, extra_meta = normalize_training_dataframe(df, args.target_column, args.task_type)

    train_df, test_df = split_dataframe(
        df=df,
        split_type=args.split_type,
        test_size=args.test_size,
        random_seed=args.random_seed,
        task_type=args.task_type,
    )

    canonical_model = canonicalize_model_type(args.model_type)

    trained_model = None
    if canonical_model == "graphconv":
        trained_model, train_pred_df, test_pred_df = train_graphconv_model(
            train_df=train_df,
            test_df=test_df,
            task_type=args.task_type,
            model_dir=args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif canonical_model == "mpnn":
        trained_model, train_pred_df, test_pred_df = train_mpnn_model(
            train_df=train_df,
            test_df=test_df,
            task_type=args.task_type,
            model_dir=args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif canonical_model in {"rf", "svm"}:
        trained_model, train_pred_df, test_pred_df = train_sklearn_model(
            train_df=train_df,
            test_df=test_df,
            model_type=canonical_model,
            task_type=args.task_type,
            model_dir=args.model_dir,
            random_seed=args.random_seed,
        )
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    save_training_smiles(args.model_dir, train_df["smiles"].tolist())

    metadata = {
        "target_column": args.target_column,
        "task_type": args.task_type,
        "model_type": canonical_model,
        "original_model_type": args.model_type,
        "split_type": args.split_type,
        "test_size": args.test_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "random_seed": args.random_seed,
        "class_mapping": extra_meta.get("class_mapping"),
    }
    save_metadata(args.model_dir, metadata)

    if args.task_type == "classification":
        y_true = test_pred_df["_target_encoded"].values
        y_pred = test_pred_df["predicted_class"].values
        y_prob = test_pred_df["predicted_probability"].values if "predicted_probability" in test_pred_df.columns else None

        metrics = classification_metrics(y_true, y_pred, y_prob)

        save_metrics_plot(
            metrics,
            os.path.join(args.job_dir, "plots", "performance.png"),
            "Classification Metrics",
        )
        save_confusion_matrix_plot(
            y_true,
            y_pred,
            os.path.join(args.job_dir, "plots", "confusion_matrix.png"),
        )

        if y_prob is not None and len(np.unique(y_true)) == 2:
            save_roc_curve_plot(
                y_true,
                y_prob,
                os.path.join(args.job_dir, "plots", "roc_curve.png"),
            )
            save_pr_curve_plot(
                y_true,
                y_prob,
                os.path.join(args.job_dir, "plots", "pr_curve.png"),
            )

        if canonical_model == "rf" and trained_model is not None:
            save_feature_importance_plot(
                trained_model,
                os.path.join(args.job_dir, "plots", "feature_importance.png"),
            )
    else:
        y_true = test_pred_df["_target_encoded"].values
        y_pred = test_pred_df["predicted_value"].values

        metrics = regression_metrics(y_true, y_pred)

        save_metrics_plot(
            metrics,
            os.path.join(args.job_dir, "plots", "performance.png"),
            "Regression Metrics",
        )
        save_regression_scatter_plot(
            y_true,
            y_pred,
            os.path.join(args.job_dir, "plots", "prediction_scatter.png"),
        )

        if canonical_model == "rf" and trained_model is not None:
            save_feature_importance_plot(
                trained_model,
                os.path.join(args.job_dir, "plots", "feature_importance.png"),
            )

    with open(os.path.join(args.job_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary = build_training_summary(train_pred_df, test_pred_df, metadata, metrics)
    with open(os.path.join(args.job_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_training_outputs(args.job_dir, train_pred_df, test_pred_df)
    print("QSAR training completed successfully.")


def predict_graphconv(df, model_dir, task_type):
    try:
        import deepchem as dc
    except ImportError:
        raise RuntimeError("DeepChem is not installed in the QSAR environment.")

    featurizer = dc.feat.ConvMolFeaturizer()
    X = featurizer.featurize(df["smiles"].tolist())
    dataset = dc.data.NumpyDataset(X=X, ids=df["smiles"].tolist())

    if task_type == "classification":
        model = dc.models.GraphConvModel(
            n_tasks=1,
            mode="classification",
            model_dir=model_dir,
        )
        model.restore()
        pred_raw = model.predict(dataset)
        prob = pred_raw[:, 0, 1]
        pred_class = (prob >= 0.5).astype(int)

        out_df = df.copy()
        out_df["predicted_class"] = pred_class
        out_df["predicted_probability"] = prob
        return out_df

    model = dc.models.GraphConvModel(
        n_tasks=1,
        mode="regression",
        model_dir=model_dir,
    )
    model.restore()
    pred = model.predict(dataset).reshape(-1)

    out_df = df.copy()
    out_df["predicted_value"] = pred
    return out_df


def predict_mpnn(df, model_dir, task_type):
    try:
        import deepchem as dc
        from deepchem.models.torch_models import MPNNModel
    except ImportError:
        raise RuntimeError("DeepChem or MPNN dependencies are not installed in the QSAR environment.")

    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    X = featurizer.featurize(df["smiles"].tolist())
    dataset = dc.data.NumpyDataset(X=X, ids=df["smiles"].tolist())

    if task_type == "classification":
        model = MPNNModel(
            n_tasks=1,
            mode="classification",
            model_dir=model_dir,
        )
        model.restore()
        pred_raw = model.predict(dataset)
        prob = pred_raw[:, 0, 1]
        pred_class = (prob >= 0.5).astype(int)

        out_df = df.copy()
        out_df["predicted_class"] = pred_class
        out_df["predicted_probability"] = prob
        return out_df

    model = MPNNModel(
        n_tasks=1,
        mode="regression",
        model_dir=model_dir,
    )
    model.restore()
    pred = model.predict(dataset).reshape(-1)

    out_df = df.copy()
    out_df["predicted_value"] = pred
    return out_df


def predict_sklearn(df, model_dir, task_type):
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved sklearn model not found: {model_path}")

    model = joblib.load(model_path)
    X, valid_idx, _ = ecfp_features_from_smiles(df["smiles"].tolist())
    valid_df = df.iloc[valid_idx].reset_index(drop=True)

    if task_type == "classification":
        pred_class = model.predict(X)
        out_df = valid_df.copy()
        out_df["predicted_class"] = pred_class
        if hasattr(model, "predict_proba") and len(model.classes_) == 2:
            proba = model.predict_proba(X)[:, 1]
            out_df["predicted_probability"] = proba
            out_df["prediction_confidence"] = np.maximum(proba, 1.0 - proba)
        return out_df

    pred = model.predict(X)
    out_df = valid_df.copy()
    out_df["predicted_value"] = pred
    return out_df


def run_predict(args):
    ensure_dir(args.job_dir)
    ensure_dir(os.path.join(args.job_dir, "plots"))

    metadata = load_metadata(args.model_dir)
    task_type = metadata["task_type"]
    model_type = metadata["model_type"]
    class_mapping = metadata.get("class_mapping")

    df = load_csv(args.input_csv)
    df = normalize_prediction_dataframe(df)

    if model_type == "graphconv":
        pred_df = predict_graphconv(df, args.model_dir, task_type)
    elif model_type == "mpnn":
        pred_df = predict_mpnn(df, args.model_dir, task_type)
    elif model_type in {"rf", "svm"}:
        pred_df = predict_sklearn(df, args.model_dir, task_type)
    else:
        raise ValueError(f"Unsupported model_type in metadata: {model_type}")

    train_smiles_path = os.path.join(args.model_dir, "training_smiles.csv")
    if os.path.exists(train_smiles_path):
        train_smiles = pd.read_csv(train_smiles_path)["smiles"].dropna().astype(str).tolist()
        ad_df = compute_applicability_for_smiles(pred_df["smiles"].tolist(), train_smiles)
        pred_df = pd.concat([pred_df.reset_index(drop=True), ad_df.reset_index(drop=True)], axis=1)

    if task_type == "classification" and class_mapping:
        inv_map = invert_class_mapping(class_mapping)
        pred_df["predicted_label"] = pred_df["predicted_class"].map(inv_map)

        if "predicted_probability" in pred_df.columns:
            plt.figure(figsize=(7, 5))
            plt.hist(pred_df["predicted_probability"], bins=20)
            plt.xlabel("Predicted Probability")
            plt.ylabel("Count")
            plt.title("Prediction Probability Distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(args.job_dir, "plots", "prediction_distribution.png"))
            plt.close()
    else:
        if "predicted_value" in pred_df.columns:
            plt.figure(figsize=(7, 5))
            plt.hist(pred_df["predicted_value"], bins=20)
            plt.xlabel("Predicted Value")
            plt.ylabel("Count")
            plt.title("Prediction Value Distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(args.job_dir, "plots", "prediction_histogram.png"))
            plt.close()

    if "max_similarity_to_training" in pred_df.columns:
        plt.figure(figsize=(7, 5))
        plt.hist(pred_df["max_similarity_to_training"].dropna(), bins=20)
        plt.xlabel("Maximum Tanimoto Similarity to Training Set")
        plt.ylabel("Count")
        plt.title("Applicability Domain Similarity")
        plt.tight_layout()
        plt.savefig(os.path.join(args.job_dir, "plots", "applicability_similarity.png"))
        plt.close()

    out_path = os.path.join(args.job_dir, "external_predictions.csv")
    pred_df.to_csv(out_path, index=False)

    top_hits_path = os.path.join(args.job_dir, "top_predictions.csv")
    sort_col = "predicted_probability" if "predicted_probability" in pred_df.columns else "predicted_value"
    pred_df.sort_values(sort_col, ascending=False, na_position="last").head(50).to_csv(top_hits_path, index=False)

    with open(os.path.join(args.job_dir, "prediction_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "row_count": int(len(pred_df)),
            "task_type": task_type,
            "model_type": model_type,
            "prediction_columns": list(pred_df.columns),
        }, f, indent=2)

    print("QSAR external prediction completed successfully.")


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--input_csv", required=True)
    train_parser.add_argument("--target_column", required=True)
    train_parser.add_argument("--task_type", choices=["classification", "regression"], required=True)
    train_parser.add_argument(
        "--model_type",
        choices=["rf", "svm", "graphconv", "mpnn", "ecfp_rf", "ecfp_svm"],
        required=True,
    )
    train_parser.add_argument("--split_type", choices=["random", "scaffold"], required=True)
    train_parser.add_argument("--test_size", type=float, required=True)
    train_parser.add_argument("--epochs", type=int, required=True)
    train_parser.add_argument("--batch_size", type=int, required=True)
    train_parser.add_argument("--random_seed", type=int, required=True)
    train_parser.add_argument("--job_dir", required=True)
    train_parser.add_argument("--model_dir", required=True)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--input_csv", required=True)
    predict_parser.add_argument("--job_dir", required=True)
    predict_parser.add_argument("--model_dir", required=True)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)