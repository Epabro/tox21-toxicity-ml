import os
import time
import json
import numpy as np
import pandas as pd

from tdc.single_pred import Tox

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.metrics import roc_auc_score, average_precision_score

from xgboost import XGBClassifier


# --- Fingerprints (robust across RDKit builds) ---
def smiles_to_morgan_bits(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

    # Robust conversion: bitstring -> numpy 0/1 vector
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
    val_df = split.get("valid", None)
    if val_df is None:
        val_df = split.get("val", None)
    test_df = split["test"]
    if val_df is None:
        raise KeyError(f"No validation split found. Split keys: {list(split.keys())}")
    return train_df, val_df, test_df


def fit_with_best_effort_early_stopping(model, X_train, y_train, X_val, y_val, rounds=50):
    """
    Tries:
      1) early_stopping_rounds=...
      2) callbacks=[EarlyStopping(...)]
      3) fallback: no early stopping
    Returns: (fit_mode: str)
    """
    # 1) sklearn-style early_stopping_rounds
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=rounds,
        )
        return "early_stopping_rounds"
    except TypeError:
        pass

    # 2) callback-style early stopping
    try:
        from xgboost.callback import EarlyStopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            callbacks=[EarlyStopping(rounds=rounds, save_best=True)],
        )
        return "callbacks"
    except TypeError:
        pass
    except Exception:
        pass

    # 3) fallback
    model.fit(X_train, y_train)
    return "none"


def main():
    os.makedirs("reports", exist_ok=True)

    endpoint = "NR-AR-LBD"
    seed = 42
    n_bits = 2048

    # “big” n_estimators + early stopping = usually best practice
    params = dict(
        max_depth=3,
        n_estimators=5000,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
    )

    print(f"Endpoint: {endpoint}")
    print(f"Params: {params}")

    t0 = time.time()

    data = Tox(name="Tox21", label_name=endpoint)
    df = data.get_data()

    smiles_col = find_smiles_col(df)
    y_col = find_label_col(df)

    train_df, val_df, test_df = get_split(data, seed=seed)

    X_train, y_train = make_xy(train_df, smiles_col, y_col, n_bits=n_bits)
    X_val, y_val = make_xy(val_df, smiles_col, y_col, n_bits=n_bits)
    X_test, y_test = make_xy(test_df, smiles_col, y_col, n_bits=n_bits)

    print(f"Train: n={len(y_train)} pos={y_train.mean() if len(y_train) else np.nan:.3f}")
    print(f"Valid: n={len(y_val)} pos={y_val.mean() if len(y_val) else np.nan:.3f}")
    print(f"Test:  n={len(y_test)} pos={y_test.mean() if len(y_test) else np.nan:.3f}")

    # Guard: early stopping / metrics require both classes
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        raise RuntimeError("Train/Valid must contain both classes (0 and 1) for early stopping + PR-AUC/ROC-AUC.")

    # class imbalance handling
    pos = float(y_train.sum())
    neg = float((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    print(f"scale_pos_weight: {scale_pos_weight:.3f}")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",  # good match for imbalanced problems
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        **params,
    )

    fit_mode = fit_with_best_effort_early_stopping(model, X_train, y_train, X_val, y_val, rounds=50)
    print(f"Fit mode: {fit_mode}")

    # Best iteration info (only present if early stopping happened)
    best_iter = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    if best_iter is not None:
        print(f"best_iteration: {best_iter}")
    if best_score is not None:
        print(f"best_score: {best_score}")

    val_m = safe_metrics(model, X_val, y_val)
    test_m = safe_metrics(model, X_test, y_test)

    print(f"VAL  PR-AUC: {val_m['pr_auc']:.3f} | ROC-AUC: {val_m['roc_auc']:.3f}")
    print(f"TEST PR-AUC: {test_m['pr_auc']:.3f} | ROC-AUC: {test_m['roc_auc']:.3f}")
    print(f"Done in {time.time() - t0:.1f}s")

    row = {
        "endpoint": endpoint,
        "seed": seed,
        "fit_mode": fit_mode,
        "best_iteration": best_iter,
        "best_score": best_score,
        "val_pr_auc": val_m["pr_auc"],
        "val_roc_auc": val_m["roc_auc"],
        "test_pr_auc": test_m["pr_auc"],
        "test_roc_auc": test_m["roc_auc"],
        "test_pos_rate": test_m["pos_rate"],
        "scale_pos_weight": scale_pos_weight,
        **params,
    }

    out_csv = "reports/005_xgb_earlystop_single_endpoint.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    out_json = "reports/005_xgb_earlystop_single_endpoint.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
