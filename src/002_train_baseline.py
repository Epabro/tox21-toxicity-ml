import os
import time
import numpy as np
import pandas as pd

from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, average_precision_score


def find_smiles_col(df: pd.DataFrame) -> str:
    for c in ["Drug", "drug", "SMILES", "smiles", "Molecule", "molecule"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise ValueError(f"Could not find SMILES column. Columns: {list(df.columns)}")


def find_label_col(df: pd.DataFrame) -> str:
    for c in ["Y", "y", "label", "Label"]:
        if c in df.columns:
            return c
    return df.columns[-1]


def smiles_to_morgan(smiles: str, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def make_xy(df: pd.DataFrame, smiles_col: str, y_col: str, n_bits=2048):
    X_list, y_list = [], []
    for smi, y in zip(df[smiles_col].astype(str), df[y_col]):
        if pd.isna(y):
            continue
        fp = smiles_to_morgan(smi, n_bits=n_bits)
        if fp is None:
            continue
        X_list.append(fp)
        y_list.append(int(y))

    if len(y_list) == 0:
        return np.zeros((0, n_bits), dtype=np.float32), np.array([], dtype=np.int64)

    X = np.vstack(X_list).astype(np.float32)  # float32 speeds things up
    y = np.array(y_list, dtype=np.int64)
    return X, y


def safe_metrics(model, X, y):
    out = {"n": int(len(y)), "pos_rate": float(y.mean()) if len(y) else np.nan}
    if len(y) == 0 or len(np.unique(y)) < 2:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
        return out
    proba = model.predict_proba(X)[:, 1]
    out["roc_auc"] = float(roc_auc_score(y, proba))
    out["pr_auc"] = float(average_precision_score(y, proba))
    return out


def get_split(data, seed: int):
    split = data.get_split(method="scaffold", seed=seed, frac=[0.8, 0.1, 0.1])
    train_df = split["train"]

    # SAFE: no "or" with DataFrames
    val_df = split.get("valid", None)
    if val_df is None:
        val_df = split.get("val", None)

    test_df = split["test"]

    if val_df is None:
        raise KeyError(f"No validation split found. Split keys: {list(split.keys())}")

    return train_df, val_df, test_df


def main():
    os.makedirs("reports", exist_ok=True)

    seed = 42
    grid = {"C": [0.1, 1.0, 3.0, 10.0]}
    endpoints = retrieve_label_name_list("Tox21")

    print(f"Found {len(endpoints)} Tox21 endpoints:\n{endpoints}")

    rows = []

    for endpoint in endpoints:
        print(f"\n=== Endpoint: {endpoint} ===")
        t0 = time.time()

        data = Tox(name="Tox21", label_name=endpoint)
        df = data.get_data()

        smiles_col = find_smiles_col(df)
        y_col = find_label_col(df)

        train_df, val_df, test_df = get_split(data, seed=seed)

        X_train, y_train = make_xy(train_df, smiles_col, y_col)
        X_val, y_val = make_xy(val_df, smiles_col, y_col)
        X_test, y_test = make_xy(test_df, smiles_col, y_col)

        print(f"Train: n={len(y_train)} pos={y_train.mean() if len(y_train) else np.nan:.3f}")
        print(f"Valid: n={len(y_val)} pos={y_val.mean() if len(y_val) else np.nan:.3f}")
        print(f"Test:  n={len(y_test)} pos={y_test.mean() if len(y_test) else np.nan:.3f}")

        # Skip if training has only one class
        if len(y_train) == 0 or len(np.unique(y_train)) < 2:
            print("Skipping: training set has <2 classes.")
            rows.append({
                "endpoint": endpoint,
                "status": "skipped_train_one_class",
                "best_C": np.nan,
                "val_pr_auc": np.nan,
                "val_roc_auc": np.nan,
                "test_pr_auc": np.nan,
                "test_roc_auc": np.nan,
                "train_n": len(y_train),
                "val_n": len(y_val),
                "test_n": len(y_test),
                "test_pos_rate": float(y_test.mean()) if len(y_test) else np.nan,
            })
            continue

        best = None
        best_model = None

        print("Training + tuning (C grid)...")
        for params in ParameterGrid(grid):
            model = LogisticRegression(
                solver="liblinear",          # faster/stable for this setup
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
                C=params["C"],
            )
            model.fit(X_train, y_train)
            m_val = safe_metrics(model, X_val, y_val)

            if best is None or (not np.isnan(m_val["pr_auc"]) and m_val["pr_auc"] > best["val_pr_auc"]):
                best = {
                    "best_C": params["C"],
                    "val_pr_auc": m_val["pr_auc"],
                    "val_roc_auc": m_val["roc_auc"],
                }
                best_model = model

        m_test = safe_metrics(best_model, X_test, y_test)

        print(f"Best C: {best['best_C']} | VAL PR-AUC: {best['val_pr_auc']:.3f} | VAL ROC-AUC: {best['val_roc_auc']:.3f}")
        print(f"TEST PR-AUC: {m_test['pr_auc']:.3f} | TEST ROC-AUC: {m_test['roc_auc']:.3f}")
        print(f"Done in {time.time() - t0:.1f}s")

        rows.append({
            "endpoint": endpoint,
            "status": "ok",
            "best_C": best["best_C"],
            "val_pr_auc": best["val_pr_auc"],
            "val_roc_auc": best["val_roc_auc"],
            "test_pr_auc": m_test["pr_auc"],
            "test_roc_auc": m_test["roc_auc"],
            "train_n": len(y_train),
            "val_n": len(y_val),
            "test_n": len(y_test),
            "test_pos_rate": m_test["pos_rate"],
        })

        # incremental save (so you don’t lose progress if something crashes)
        pd.DataFrame(rows).to_csv("reports/baseline_all_endpoints.csv", index=False)

    print("\nSaved: reports/baseline_all_endpoints.csv")
    res = pd.DataFrame(rows)
    ok = res[res["status"] == "ok"].sort_values("test_pr_auc", ascending=False)
    print("\nTop endpoints by TEST PR-AUC:")
    print(ok[["endpoint", "test_pr_auc", "test_roc_auc", "best_C", "test_pos_rate"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
