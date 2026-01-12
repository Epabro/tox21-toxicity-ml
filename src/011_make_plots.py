import os
import pandas as pd
import matplotlib.pyplot as plt


COMPARE_CSV = "reports/compare_4_models.csv"
FIG_DIR = "reports/figures"


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)


def load_compare(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Run src/010_compare_four_models.py first."
        )
    df = pd.read_csv(path)

    required = [
        "endpoint",
        "best_model",
        "best_test_pr_auc",
        "lr_test_pr_auc",
        "xgb_grid_test_pr_auc",
        "xgb_es_test_pr_auc",
        "xgb_es_grid_test_pr_auc",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    # Sort endpoints by best PR-AUC (descending)
    df = df.sort_values("best_test_pr_auc", ascending=False).reset_index(drop=True)
    return df


def plot_best_per_endpoint(df: pd.DataFrame):
    endpoints = df["endpoint"].tolist()
    y = df["best_test_pr_auc"].tolist()

    plt.figure(figsize=(14, 6))
    plt.bar(endpoints, y)
    plt.ylabel("TEST PR-AUC")
    plt.title("Best model per endpoint (TEST PR-AUC)")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, max(y) * 1.1 if len(y) else 1.0)
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "best_model_test_pr_auc.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def plot_lr_vs_best_xgb(df: pd.DataFrame):
    """
    Plot LR vs best XGB variant and annotate delta = (best_xgb - lr) for each endpoint.
    """
    df = df.copy()
    df["best_xgb_test_pr_auc"] = df[
        ["xgb_grid_test_pr_auc", "xgb_es_test_pr_auc", "xgb_es_grid_test_pr_auc"]
    ].max(axis=1)

    # Compute deltas and store a small artifact for the report
    df["delta_bestxgb_minus_lr"] = df["best_xgb_test_pr_auc"] - df["lr_test_pr_auc"]
    delta_csv = os.path.join(FIG_DIR, "delta_lr_vs_best_xgb.csv")
    df[["endpoint", "lr_test_pr_auc", "best_xgb_test_pr_auc", "delta_bestxgb_minus_lr"]].to_csv(
        delta_csv, index=False
    )
    print(f"Saved: {delta_csv}")

    # Print summary: top gains and regressions
    top_gain = df.sort_values("delta_bestxgb_minus_lr", ascending=False).head(3)
    top_drop = df.sort_values("delta_bestxgb_minus_lr", ascending=True).head(3)

    print("\nTop 3 improvements (BestXGB - LR) by TEST PR-AUC:")
    print(top_gain[["endpoint", "lr_test_pr_auc", "best_xgb_test_pr_auc", "delta_bestxgb_minus_lr"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nTop 3 regressions (BestXGB - LR) by TEST PR-AUC:")
    print(top_drop[["endpoint", "lr_test_pr_auc", "best_xgb_test_pr_auc", "delta_bestxgb_minus_lr"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    endpoints = df["endpoint"].tolist()
    lr = df["lr_test_pr_auc"].tolist()
    best_xgb = df["best_xgb_test_pr_auc"].tolist()
    deltas = df["delta_bestxgb_minus_lr"].tolist()

    x = list(range(len(endpoints)))
    width = 0.4

    plt.figure(figsize=(14, 6))
    plt.bar([i - width / 2 for i in x], lr, width=width, label="LR")
    plt.bar([i + width / 2 for i in x], best_xgb, width=width, label="Best XGB variant")

    # Add delta labels above each pair
    ymax = max(max(lr), max(best_xgb)) if len(endpoints) else 1.0
    pad = ymax * 0.02  # vertical padding for text

    for i in range(len(endpoints)):
        pair_top = max(lr[i], best_xgb[i])
        d = deltas[i]
        sign = "+" if d >= 0 else ""
        plt.text(
            i, pair_top + pad,
            f"{sign}{d:.3f}",
            ha="center", va="bottom",
            fontsize=9,
        )

    plt.ylabel("TEST PR-AUC")
    plt.title("LR vs Best XGB variant (TEST PR-AUC) — annotated with Δ(PR-AUC)")
    plt.xticks(x, endpoints, rotation=45, ha="right")
    plt.legend()
    plt.ylim(0, ymax * 1.15)
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "lr_vs_best_xgb_test_pr_auc.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def main():
    ensure_dirs()
    df = load_compare(COMPARE_CSV)

    # Save a small “plot-ready” table too
    plot_df = df[
        [
            "endpoint",
            "best_model",
            "best_test_pr_auc",
            "lr_test_pr_auc",
            "xgb_grid_test_pr_auc",
            "xgb_es_test_pr_auc",
            "xgb_es_grid_test_pr_auc",
        ]
    ].copy()
    out_table = os.path.join(FIG_DIR, "plot_table_compare_4_models.csv")
    plot_df.to_csv(out_table, index=False)
    print(f"Saved: {out_table}")

    plot_best_per_endpoint(df)
    plot_lr_vs_best_xgb(df)


if __name__ == "__main__":
    main()
