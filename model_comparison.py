"""
Model family comparison for the Return-Risk Scorer.

Compares the current baseline (Random Forest) against several gradient
boosting variants, using the SAME chronological split and the same
roc_auc-based tuning approach established in model_training.py. Then
tests whether adding a small set of engineered interaction terms helps
the best-performing family.

Honesty check built in: this does NOT assume a fancier model will win.
Given the underlying signal is genuinely weak (an ~8-point spread in
return rate across Discount_Applied, confirmed in the debugging trail),
it's entirely possible every model family lands in a similar range --
that would be a legitimate result to report, not a failure of this script.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from data_preparation import datapreparation, chronological_split

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def add_interaction_features(x):
    """
    Adds a small set of engineered interaction terms on top of the
    existing leakage-safe, redundancy-pruned feature set:
      - Discount_Applied x each Product_Category dummy (does the
        discount-return relationship differ by category?)
      - Discount_Applied x user_prior_return_rate (does discount matter
        more for customers who already return often?)
    Tree models can approximate interactions via sequential splits on
    their own, but explicit terms can still help a shallow, regularized
    tree find them with less data to work with.
    """
    x = x.copy()
    category_cols = [c for c in x.columns if c.startswith("Product_Category_")]
    for col in category_cols:
        x[f"Discount_x_{col}"] = x["Discount_Applied"] * x[col]

    if "user_prior_return_rate" in x.columns:
        x["Discount_x_prior_return_rate"] = x["Discount_Applied"] * x["user_prior_return_rate"]

    return x


def evaluate_model(name, model, param_grid, x_train, y_train, x_test, y_test, cv=3):
    grid = GridSearchCV(model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(x_train, y_train)
    best = grid.best_estimator_
    probs = best.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"{name:28s} best_cv_auc={grid.best_score_:.4f}  held_out_auc={auc:.4f}  best_params={grid.best_params_}")
    return best, auc


def build_grids():
    """Kept small deliberately -- this is a family comparison, not an
    exhaustive search. Once a winning family is identified, it's worth
    widening its grid specifically."""
    return {
        "RandomForest": (RandomForestClassifier(random_state=42), {
            "max_depth": [4, 6, None],
            "min_samples_leaf": [1, 15, 30],
            "n_estimators": [300],
        }),
        "HistGradientBoosting": (HistGradientBoostingClassifier(random_state=42), {
            "max_depth": [3, 5, None],
            "learning_rate": [0.03, 0.1],
            "max_iter": [150],
        }),
        "GradientBoosting": (GradientBoostingClassifier(random_state=42), {
            "max_depth": [2, 3],
            "learning_rate": [0.03, 0.1],
            "n_estimators": [150],
        }),
    }


if __name__ == "__main__":
    df, cost_reference = datapreparation("returns_sustainability_dataset.csv")
    x_train, x_test, y_train, y_test, cost_test, cutoff = chronological_split(df, cost_reference)

    print(f"Train: {x_train.shape}, Test: {x_test.shape}\n")
    print("=== Step 1: Model family comparison (no interactions) ===\n")

    grids = build_grids()
    if HAS_XGBOOST:
        grids["XGBoost"] = (XGBClassifier(random_state=42, eval_metric="logloss"), {
            "max_depth": [3, 5],
            "learning_rate": [0.03, 0.1],
            "n_estimators": [150],
        })
    else:
        print("xgboost not installed -- skipping (pip install xgboost --break-system-packages)")

    if HAS_LIGHTGBM:
        grids["LightGBM"] = (LGBMClassifier(random_state=42, verbosity=-1), {
            "max_depth": [3, 5, -1],
            "learning_rate": [0.03, 0.1],
            "n_estimators": [150],
        })
    else:
        print("lightgbm not installed -- skipping (pip install lightgbm --break-system-packages)")

    results = {}
    fitted = {}
    for name, (model, grid) in grids.items():
        best, auc = evaluate_model(name, model, grid, x_train, y_train, x_test, y_test)
        results[name] = auc
        fitted[name] = best

    print("\n=== Summary (held-out AUC, no interactions) ===")
    for name, auc in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:25s} {auc:.4f}")

    best_name = max(results, key=results.get)
    print(f"\nBest family so far: {best_name} (AUC={results[best_name]:.4f})")

    print("\n=== Step 2: Does adding feature interactions help the best family? ===\n")
    x_train_interact = add_interaction_features(x_train)
    x_test_interact = add_interaction_features(x_test)
    print(f"Features with interactions: {x_train_interact.shape[1]} (was {x_train.shape[1]})")

    best_model_template, best_grid = grids[best_name]
    _, auc_with_interactions = evaluate_model(
        f"{best_name} + interactions", best_model_template, best_grid,
        x_train_interact, y_train, x_test_interact, y_test
    )

    print(f"\n{best_name} without interactions: AUC={results[best_name]:.4f}")
    print(f"{best_name} with interactions:    AUC={auc_with_interactions:.4f}")
    if auc_with_interactions > results[best_name] + 0.01:
        print(">>> Interactions provided a meaningful lift -- consider adopting them.")
    else:
        print(">>> Interactions did not provide a meaningful lift beyond noise -- "
              "stick with the simpler feature set; it's easier to explain via SHAP too.")
