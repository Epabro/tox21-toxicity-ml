import os
import time
import json
import numpy as np
import pandas as pd

import xgboost as xgb

from tdc.single_pred import Tox
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score, average_precision_score


def smiles_to_morgan_bits(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    bitstr = fp.ToBitString()
    arr = np.frombuffer(bitstr.encode("ascii"), dtype=np.uint8) - ord("0")
    return arr


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


def make_xy(df: pd.DataFrame, smiles_col: str, y_col: str, n_bits: int = 2048):
    X_list, y_list = [], []
    for smi, y in zip(df[smiles_col].astype(str), df[y_col]):
        if pd.isna(y):
            continue
        fp = smiles_to_morgan_bits(smi, n_bits=n_bits)
        if fp is None:
            continue
        X_list.append(fp)
        y_list.append(int(y))

    if len(y_list) == 0:
        return np.zeros((0, n_bits), dtype=np.float32), np.array([], dtype=np.int64)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def safe_metrics_from_proba(y, proba):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "n": int(len(y)), "pos_rate": float(np.mean(y)) if len(y) else np.nan}
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "n": int(len(y)),
        "pos_rate": float(np.mean(y)),
    }


def get_split(data, seed: int):
    split = data.get_split(method="scaffold", seed=seed, frac=[0.8, 0.1, 0.1])
    train_df = split["train"]

    val_df = split.get("valid", None)
    if val_df is None:
        val_df = split.get("val", None)

    test_df = split["test"]

    if val_df is None:
        raise KeyError(f"No validation split found. Split keys: {list(split.keys())}")

    return train_df, val_df, test_df



def main():
    os.makedirs("reports", exist_ok=True)

    endpoint = "NR-AR-LBD"
    seed = 42
    n_bits = 2048

    print(f"Endpoint: {endpoint}")
    t0 = time.time()

    data = Tox(name="Tox21", label_name=endpoint)
    df = data.get_data()
    smiles_col = find_smiles_col(df)
    y_col = find_label_col(df)

    train_df, val_df, test_df = get_split(data, seed=seed)

    X_train, y_train = make_xy(train_df, smiles_col, y_col, n_bits=n_bits)
    X_val, y_val = make_xy(val_df, smiles_col, y_col, n_bits=n_bits)
    X_test, y_test = make_xy(test_df, smiles_col, y_col, n_bits=n_bits)

    print(f"Train: n={len(y_train)} pos={y_train.mean():.3f}")
    print(f"Valid: n={len(y_val)} pos={y_val.mean():.3f}")
    print(f"Test:  n={len(y_test)} pos={y_test.mean():.3f}")

    # scale_pos_weight
    pos = float(y_train.sum())
    neg = float((y_train == 0).sum())
    spw = (neg / pos) if pos > 0 else 1.0
    print(f"scale_pos_weight: {spw:.3f}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "seed": seed,
        "max_depth": 3,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "min_child_weight": 1.0,
        "scale_pos_weight": spw,
        "tree_method": "hist",
    }
    num_boost_round = 5000
    early_stopping_rounds = 50

    print(f"Params: {params}")
    print(f"num_boost_round={num_boost_round}, early_stopping_rounds={early_stopping_rounds}")

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )

    best_iter = booster.best_iteration
    best_score = booster.best_score
    print(f"best_iteration: {best_iter}")
    print(f"best_score (valid aucpr): {best_score}")

    # Predict probabilities
    val_proba = booster.predict(dvalid)
    test_proba = booster.predict(dtest)

    val_m = safe_metrics_from_proba(y_val, val_proba)
    test_m = safe_metrics_from_proba(y_test, test_proba)

    print(f"VAL  PR-AUC: {val_m['pr_auc']:.3f} | ROC-AUC: {val_m['roc_auc']:.3f}")
    print(f"TEST PR-AUC: {test_m['pr_auc']:.3f} | ROC-AUC: {test_m['roc_auc']:.3f}")
    print(f"Done in {time.time() - t0:.1f}s")

    row = {
        "endpoint": endpoint,
        "seed": seed,
        "best_iteration": int(best_iter) if best_iter is not None else None,
        "best_score_valid_aucpr": float(best_score) if best_score is not None else None,
        "val_pr_auc": val_m["pr_auc"],
        "val_roc_auc": val_m["roc_auc"],
        "test_pr_auc": test_m["pr_auc"],
        "test_roc_auc": test_m["roc_auc"],
        "test_pos_rate": test_m["pos_rate"],
        "scale_pos_weight": spw,
        "params": json.dumps(params),
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
    }

    out_csv = "reports/006_xgb_native_earlystop_single_endpoint.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
