import os
import time
import json
import numpy as np
import pandas as pd

import xgboost as xgb

from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import ParameterGrid


# --- Robust fingerprints (RDKit -> bitstring -> numpy) ---
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


def metrics_from_proba(y, proba):
    out = {"n": int(len(y)), "pos_rate": float(np.mean(y)) if len(y) else np.nan}
    if len(y) == 0 or len(np.unique(y)) < 2:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
        return out
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


def main():
    os.makedirs("reports", exist_ok=True)

    seed = 42
    n_bits = 2048
    endpoints = retrieve_label_name_list("Tox21")

    # Training controls
    num_boost_round = 5000
    early_stopping_rounds = 50
    verbose_eval = False  # set True for debugging

    # Base params (kept stable)
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "seed": seed,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "tree_method": "hist",
    }

    # Small grid (reasonable runtime)
    grid = {
        "max_depth": [2, 3, 4],
        "min_child_weight": [1.0, 5.0],
        "gamma": [0.0, 1.0],
    }
    grid_list = list(ParameterGrid(grid))
    print(f"Grid size per endpoint: {len(grid_list)} configs")

    rows = []
    print(f"Found {len(endpoints)} endpoints.")

    for endpoint in endpoints:
        print(f"\n=== Endpoint: {endpoint} ===")
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

        # Need both classes in train/val
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            print("Skipping: train/valid has <2 classes.")
            rows.append({"endpoint": endpoint, "status": "skipped_one_class"})
            pd.DataFrame(rows).to_csv("reports/009_xgb_native_es_small_grid_all_endpoints.csv", index=False)
            continue

        # class imbalance per endpoint
        pos = float(y_train.sum())
        neg = float((y_train == 0).sum())
        spw = (neg / pos) if pos > 0 else 1.0

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        best = None  # dict with scores + params
        best_booster = None

        for i, cfg in enumerate(grid_list, start=1):
            params = dict(base_params)
            params.update(cfg)
            params["scale_pos_weight"] = spw

            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
            )

            # model.best_score is validation aucpr (because eval_metric=aucpr)
            score = float(booster.best_score) if booster.best_score is not None else float("nan")

            if best is None or (not np.isnan(score) and score > best["best_score_valid_aucpr"]):
                best = {
                    "best_score_valid_aucpr": score,
                    "best_iteration": int(booster.best_iteration) if booster.best_iteration is not None else None,
                    **cfg,
                }
                best_booster = booster

            if i == 1 or i == len(grid_list) or i % 4 == 0:
                print(f"  [{i}/{len(grid_list)}] cfg={cfg} best_valid_aucpr={score:.4f}")

        # Evaluate best model on val/test with sklearn metrics
        val_proba = best_booster.predict(dvalid)
        test_proba = best_booster.predict(dtest)

        val_m = metrics_from_proba(y_val, val_proba)
        test_m = metrics_from_proba(y_test, test_proba)

        print(f"BEST cfg: { {k: best[k] for k in ['max_depth','min_child_weight','gamma']} }")
        print(f"best_iteration: {best['best_iteration']} | best_score(valid aucpr): {best['best_score_valid_aucpr']:.4f}")
        print(f"VAL  PR-AUC: {val_m['pr_auc']:.3f} | ROC-AUC: {val_m['roc_auc']:.3f}")
        print(f"TEST PR-AUC: {test_m['pr_auc']:.3f} | ROC-AUC: {test_m['roc_auc']:.3f}")
        print(f"Done in {time.time() - t0:.1f}s")

        rows.append({
            "endpoint": endpoint,
            "status": "ok",
            "seed": seed,
            "val_pr_auc": val_m["pr_auc"],
            "val_roc_auc": val_m["roc_auc"],
            "test_pr_auc": test_m["pr_auc"],
            "test_roc_auc": test_m["roc_auc"],
            "test_pos_rate": test_m["pos_rate"],
            "scale_pos_weight": spw,
            "best_iteration": best["best_iteration"],
            "best_score_valid_aucpr": best["best_score_valid_aucpr"],
            "max_depth": best["max_depth"],
            "min_child_weight": best["min_child_weight"],
            "gamma": best["gamma"],
            "params_base": json.dumps(base_params),
            "num_boost_round": num_boost_round,
            "early_stopping_rounds": early_stopping_rounds,
        })

        pd.DataFrame(rows).to_csv("reports/009_xgb_native_es_small_grid_all_endpoints.csv", index=False)

    print("\nSaved: reports/009_xgb_native_es_small_grid_all_endpoints.csv")
    res = pd.DataFrame(rows)
    ok = res[res["status"] == "ok"].sort_values("test_pr_auc", ascending=False)
    print("\nTop endpoints by TEST PR-AUC (small grid + early stop):")
    if len(ok):
        print(ok[["endpoint", "test_pr_auc", "test_roc_auc", "best_iteration", "max_depth", "min_child_weight", "gamma"]]
              .head(12).to_string(index=False))


if __name__ == "__main__":
    main()
