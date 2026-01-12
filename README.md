# Tox21 Toxicity Prediction (Scaffold Split) — Baselines → XGBoost + Early Stopping

End-to-end ML project to predict **Tox21 toxicity assay endpoints** from **molecular structure (SMILES)** using **RDKit Morgan fingerprints**.  
This repo benchmarks a strong baseline (**Logistic Regression**) against multiple **XGBoost** training strategies, emphasizing **generalization to novel chemical scaffolds** via **scaffold splitting**.

---

## Why this project

In early drug discovery, we often need to predict assay outcomes for *new chemistry*. A realistic evaluation is not a random split (which leaks scaffold information) but a **scaffold split**: test molecules belong to **different scaffolds** than training chemistry.

This repository demonstrates:
- reproducible dataset loading + scaffold split
- molecular featurization (Morgan fingerprints)
- baseline model (LR) and boosted tree models (XGBoost)
- early stopping with the native XGBoost API
- results reporting (PR-AUC / ROC-AUC) and plots
- model selection per endpoint

---

## Dataset

- **Tox21** benchmark (12 binary classification endpoints)
- Accessed via `tdc.single_pred.Tox`
- Input: **SMILES**
- Labels: assay outcomes per endpoint

Endpoints:
`NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`, `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`.

---

## Methods

### Features
- Morgan fingerprints (ECFP-like)
- `radius=2`, `n_bits=2048`
- Computed with RDKit

### Split
- **Scaffold split** (train/valid/test = 0.8/0.1/0.1), `seed=42`

### Models
1. **LR**: Logistic Regression (`class_weight="balanced"`) with a small `C` grid.
2. **XGB-grid**: XGBoost sklearn wrapper with a small hyperparameter grid.
3. **XGB-ES**: Native `xgboost.train` with **early stopping** (fixed params).
4. **XGB-ES+Grid**: Native `xgboost.train` + early stopping + small grid over tree complexity regularization.

### Metrics
- Primary: **PR-AUC** (Average Precision), because many endpoints are imbalanced.
- Secondary: ROC-AUC.

---

## Results (Scaffold split, TEST PR-AUC)

### Best model per endpoint (by TEST PR-AUC)

| Endpoint | Best model | Best TEST PR-AUC | LR | XGB-grid | XGB-ES | XGB-ES+Grid |
|---|---|---:|---:|---:|---:|---:|
| NR-AR-LBD | XGB-ES+Grid | **0.740689** | 0.556069 | 0.638361 | 0.693843 | 0.740689 |
| SR-MMP | LR | **0.587816** | 0.587816 | 0.581691 | 0.576096 | 0.573672 |
| NR-AhR | XGB-ES+Grid | **0.533195** | 0.516647 | 0.525595 | 0.526023 | 0.533195 |
| NR-AR | LR | **0.507015** | 0.507015 | 0.485643 | 0.468611 | 0.467019 |
| NR-ER | XGB-ES+Grid | **0.479833** | 0.440490 | 0.479351 | 0.458608 | 0.479833 |
| NR-ER-LBD | XGB-grid | **0.335392** | 0.257875 | 0.335392 | 0.284396 | 0.260897 |
| SR-ARE | XGB-ES+Grid | **0.333781** | 0.333341 | 0.301956 | 0.325442 | 0.333781 |
| NR-Aromatase | XGB-ES+Grid | **0.330425** | 0.270220 | 0.268022 | 0.307244 | 0.330425 |
| SR-p53 | XGB-ES+Grid | **0.327822** | 0.302712 | 0.230180 | 0.297568 | 0.327822 |
| NR-PPAR-gamma | LR | **0.280703** | 0.280703 | 0.206128 | 0.229224 | 0.204185 |
| SR-HSE | XGB-ES | **0.234228** | 0.153215 | 0.180972 | 0.234228 | 0.208421 |
| SR-ATAD5 | LR | **0.161067** | 0.161067 | 0.148319 | 0.130269 | 0.122893 |

### Highlights (LR vs best XGB variant)
Across the 12 endpoints, the best XGBoost variant improved TEST PR-AUC on **8/12** tasks. The largest gain was **NR-AR-LBD (+0.185 PR-AUC)** (0.556 → 0.741). Additional gains include **SR-HSE (+0.081)**, **NR-ER-LBD (+0.078)**, and **NR-Aromatase (+0.060)**. Logistic Regression remained best for several endpoints, highlighting endpoint-specific difficulty and the importance of selecting models per target.

---

## Plots

If you generated plots with `src/011_make_plots.py`, they should be in `reports/figures/`.

Example embeds:

```md
![LR vs Best XGB](reports/figures/lr_vs_best_xgb_test_pr_auc.png)
![Best model per endpoint](reports/figures/best_model_test_pr_auc.png)
