import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    os.makedirs("reports", exist_ok=True)

    # Inputs (from your previous runs)
    baseline_path = "reports/baseline_all_endpoints.csv"
    xgb_path = "reports/xgb_all_endpoints.csv"

    base = pd.read_csv(baseline_path)
    xgb = pd.read_csv(xgb_path)

    # Keep only the columns we need
    base = base[base["status"] == "ok"][["endpoint", "test_pr_auc", "test_roc_auc"]].rename(
        columns={"test_pr_auc": "lr_test_pr_auc", "test_roc_auc": "lr_test_roc_auc"}
    )
    xgb = xgb[xgb["status"] == "ok"][["endpoint", "test_pr_auc", "test_roc_auc", "test_pos_rate"]].rename(
        columns={"test_pr_auc": "xgb_test_pr_auc", "test_roc_auc": "xgb_test_roc_auc"}
    )

    merged = base.merge(xgb, on="endpoint", how="outer")
    merged["delta_pr_auc"] = merged["xgb_test_pr_auc"] - merged["lr_test_pr_auc"]
    merged["delta_roc_auc"] = merged["xgb_test_roc_auc"] - merged["lr_test_roc_auc"]

    # Save comparison table
    merged.sort_values("xgb_test_pr_auc", ascending=False).to_csv("reports/compare_lr_vs_xgb.csv", index=False)
    print("Saved: reports/compare_lr_vs_xgb.csv")

    # Plot leaderboard (XGBoost TEST PR-AUC)
    plot_df = merged.sort_values("xgb_test_pr_auc", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["endpoint"], plot_df["xgb_test_pr_auc"])
    plt.xlabel("TEST PR-AUC (XGBoost)")
    plt.title("Tox21 Leaderboard (Scaffold Split) — XGBoost")
    plt.tight_layout()
    out_png = "reports/leaderboard_xgb_test_pr_auc.png"
    plt.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")

    # Plot improvement (delta PR-AUC)
    plot_df2 = merged.sort_values("delta_pr_auc", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df2["endpoint"], plot_df2["delta_pr_auc"])
    plt.xlabel("Δ PR-AUC (XGB − Logistic Regression)")
    plt.title("Improvement from Logistic Regression to XGBoost (TEST PR-AUC)")
    plt.tight_layout()
    out_png2 = "reports/delta_pr_auc_xgb_minus_lr.png"
    plt.savefig(out_png2, dpi=200)
    print(f"Saved: {out_png2}")


if __name__ == "__main__":
    main()
