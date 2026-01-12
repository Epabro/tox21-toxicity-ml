import os
import pandas as pd


def load_lr(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    return df[["endpoint", "test_pr_auc", "test_roc_auc"]].rename(
        columns={"test_pr_auc": "lr_test_pr_auc", "test_roc_auc": "lr_test_roc_auc"}
    )


def load_xgb_grid(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    return df[["endpoint", "test_pr_auc", "test_roc_auc"]].rename(
        columns={"test_pr_auc": "xgb_grid_test_pr_auc", "test_roc_auc": "xgb_grid_test_roc_auc"}
    )


def load_xgb_es(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    return df[["endpoint", "test_pr_auc", "test_roc_auc", "best_iteration", "test_pos_rate"]].rename(
        columns={"test_pr_auc": "xgb_es_test_pr_auc", "test_roc_auc": "xgb_es_test_roc_auc"}
    )


def main():
    os.makedirs("reports", exist_ok=True)

    lr_path = "reports/baseline_all_endpoints.csv"
    xgb_grid_path = "reports/xgb_all_endpoints.csv"
    xgb_es_path = "reports/007_xgb_native_earlystop_all_endpoints.csv"

    lr = load_lr(lr_path)
    grid = load_xgb_grid(xgb_grid_path)
    es = load_xgb_es(xgb_es_path)

    merged = lr.merge(grid, on="endpoint", how="outer").merge(es, on="endpoint", how="outer")

    merged["delta_grid_vs_lr_pr_auc"] = merged["xgb_grid_test_pr_auc"] - merged["lr_test_pr_auc"]
    merged["delta_es_vs_lr_pr_auc"] = merged["xgb_es_test_pr_auc"] - merged["lr_test_pr_auc"]
    merged["delta_es_vs_grid_pr_auc"] = merged["xgb_es_test_pr_auc"] - merged["xgb_grid_test_pr_auc"]

    out_csv = "reports/compare_lr_vs_xgbgrid_vs_xgbes.csv"
    merged.sort_values("xgb_es_test_pr_auc", ascending=False).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    top = merged.sort_values("xgb_es_test_pr_auc", ascending=False).head(12)
    print("\nTop endpoints by XGB+EarlyStop TEST PR-AUC:")
    print(
        top[[
            "endpoint",
            "lr_test_pr_auc",
            "xgb_grid_test_pr_auc",
            "xgb_es_test_pr_auc",
            "delta_es_vs_grid_pr_auc",
            "best_iteration",
            "test_pos_rate",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
