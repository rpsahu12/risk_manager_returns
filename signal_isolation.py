"""
Signal isolation diagnostic.

Question: is the near-random AUC because (a) there's genuinely little
predictive signal in this dataset, or (b) real-but-weak signal (mainly
Discount_Applied, secondarily Product_Category) is being diluted by
~30 other near-noise features, especially with max_features=None
(every split considers all features, which invites overfitting to
spurious splits in a high-dimensional, mostly-noisy feature set)?

This trains three variants on the SAME chronological split and compares
held-out ROC-AUC:
  1. Full feature set, same tuning as model_training.py (baseline --
     should reproduce the ~0.52 AUC already seen)
  2. Minimal feature set: just Discount_Applied + Product_Category
     dummies + user_prior_return_rate -- the columns that showed real
     marginal signal in the diagnostics.py checks
  3. Full feature set, but with a more constrained Random Forest
     (bounded depth / larger min_samples_leaf) to reduce overfitting to
     noisy splits, without removing any columns
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from data_preparation import datapreparation, chronological_split


def report(name, model, x_test, y_test):
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probabilities)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    print(f"{name:45s} AUC={auc:.4f}  precision={precision:.3f}  recall={recall:.3f}")
    return auc


if __name__ == "__main__":
    df, cost_reference = datapreparation("returns_sustainability_dataset.csv")
    x_train, x_test, y_train, y_test, cost_test, cutoff = chronological_split(df, cost_reference)

    print(f"Cutoff: {cutoff}, Train: {x_train.shape}, Test: {x_test.shape}\n")

    # --- Variant 1: full feature set, same tuning as model_training.py ---
    rf_full = RandomForestClassifier(random_state=42, max_features=None, n_estimators=300)
    rf_full.fit(x_train, y_train)
    report("1. Full feature set (baseline)", rf_full, x_test, y_test)

    # --- Variant 2: minimal feature set, only columns with real marginal signal ---
    signal_cols = [c for c in x_train.columns
                   if c == "Discount_Applied"
                   or c.startswith("Product_Category_")
                   or c == "user_prior_return_rate"]
    print(f"\nMinimal feature set ({len(signal_cols)} cols): {signal_cols}")

    x_train_min = x_train[signal_cols]
    x_test_min = x_test[signal_cols]

    rf_min = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=5, min_samples_leaf=30)
    rf_min.fit(x_train_min, y_train)
    report("2. Minimal feature set (signal-only)", rf_min, x_test_min, y_test)

    # --- Variant 3: full feature set, but constrained to reduce overfitting ---
    rf_constrained = RandomForestClassifier(
        random_state=42, n_estimators=300, max_depth=5, min_samples_leaf=30, max_features="sqrt"
    )
    rf_constrained.fit(x_train, y_train)
    report("3. Full feature set, constrained depth", rf_constrained, x_test, y_test)

    print("\nIf (2) or (3) clearly beats (1), the full model was being diluted by noisy")
    print("features -- fix by feature-pruning and/or regularizing, not by giving up on the data.")
    print("If none beat ~0.52-0.55, the dataset's signal ceiling is genuinely low regardless")
    print("of model choice, and it's time to consider the semi-synthetic label approach.")
