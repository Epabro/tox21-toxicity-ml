import os
import pandas as pd


def load_lr(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df.get("status", "ok") == "ok"].copy()
    return df[["endpoint", "test_pr_auc", "test_roc_auc"]].rename(
        columns={"test_pr_auc": "lr_test_pr_auc", "test_roc_auc": "lr_test_roc_auc"}
    )


def load_xgb_grid(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df.get("status", "ok") == "ok"].copy()
    return df[["endpoint", "test_pr_auc", "test_roc_auc"]].rename(
        columns={
            "test_pr_auc": "xgb_grid_test_pr_auc",
            "test_roc_auc": "xgb_grid_test_roc_auc",
        }
    )


def load_xgb_es(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df.get("status", "ok") == "ok"].copy()
    keep = ["endpoint", "test_pr_auc", "test_roc_auc", "best_iteration", "test_pos_rate"]
    out = df[keep].rename(
        columns={
            "test_pr_auc": "xgb_es_test_pr_auc",
            "test_roc_auc": "xgb_es_test_roc_auc",
            "best_iteration": "xgb_es_best_iteration",
        }
    )
    return out


def load_xgb_es_grid(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df.get("status", "ok") == "ok"].copy()
    keep = [
        "endpoint",
        "test_pr_auc",
        "test_roc_auc",
        "best_iteration",
        "max_depth",
        "min_child_weight",
        "gamma",
    ]
    out = df[keep].rename(
        columns={
            "test_pr_auc": "xgb_es_grid_test_pr_auc",
            "test_roc_auc": "xgb_es_grid_test_roc_auc",
            "best_iteration": "xgb_es_grid_best_iteration",
            "max_depth": "xgb_es_grid_max_depth",
            "min_child_weight": "xgb_es_grid_min_child_weight",
            "gamma": "xgb_es_grid_gamma",
        }
    )
    return out


def pick_best_model_row(row):
    candidates = {
        "LR": row.get("lr_test_pr_auc"),
        "XGB-grid": row.get("xgb_grid_test_pr_auc"),
        "XGB-ES": row.get("xgb_es_test_pr_auc"),
        "XGB-ES+Grid": row.get("xgb_es_grid_test_pr_auc"),
    }
    # Drop NaNs
    candidates = {k: v for k, v in candidates.items() if pd.notna(v)}
    if not candidates:
        return pd.Series({"best_model": None, "best_test_pr_auc": None})
    best_model = max(candidates, key=candidates.get)
    return pd.Series({"best_model": best_model, "best_test_pr_auc": candidates[best_model]})


def main():
    os.makedirs("reports", exist_ok=True)

    lr_path = "reports/baseline_all_endpoints.csv"
    xgb_grid_path = "reports/xgb_all_endpoints.csv"
    xgb_es_path = "reports/007_xgb_native_earlystop_all_endpoints.csv"
    xgb_es_grid_path = "reports/009_xgb_native_es_small_grid_all_endpoints.csv"

    lr = load_lr(lr_path)
    grid = load_xgb_grid(xgb_grid_path)
    es = load_xgb_es(xgb_es_path)
    esg = load_xgb_es_grid(xgb_es_grid_path)

    merged = (
        lr.merge(grid, on="endpoint", how="outer")
          .merge(es, on="endpoint", how="outer")
          .merge(esg, on="endpoint", how="outer")
    )

    # Deltas vs LR
    merged["delta_grid_vs_lr_pr_auc"] = merged["xgb_grid_test_pr_auc"] - merged["lr_test_pr_auc"]
    merged["delta_es_vs_lr_pr_auc"] = merged["xgb_es_test_pr_auc"] - merged["lr_test_pr_auc"]
    merged["delta_es_grid_vs_lr_pr_auc"] = merged["xgb_es_grid_test_pr_auc"] - merged["lr_test_pr_auc"]

    # Best model per endpoint
    best = merged.apply(pick_best_model_row, axis=1)
    merged = pd.concat([merged, best], axis=1)

    out_csv = "reports/compare_4_models.csv"
    merged.sort_values("best_test_pr_auc", ascending=False).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    print("\nBest model per endpoint (by TEST PR-AUC):")
    show = merged.sort_values("best_test_pr_auc", ascending=False)[
        [
            "endpoint",
            "best_model",
            "best_test_pr_auc",
            "lr_test_pr_auc",
            "xgb_grid_test_pr_auc",
            "xgb_es_test_pr_auc",
            "xgb_es_grid_test_pr_auc",
        ]
    ]
    print(show.to_string(index=False))

    print("\nTop endpoints by XGB-ES+Grid TEST PR-AUC:")
    show2 = merged.sort_values("xgb_es_grid_test_pr_auc", ascending=False)[
        [
            "endpoint",
            "xgb_es_grid_test_pr_auc",
            "xgb_es_grid_test_roc_auc",
            "xgb_es_grid_best_iteration",
            "xgb_es_grid_max_depth",
            "xgb_es_grid_min_child_weight",
            "xgb_es_grid_gamma",
        ]
    ].head(12)
    print(show2.to_string(index=False))


if __name__ == "__main__":
    main()
