"""
Explainability for the Return-Risk Scorer (AI Risk Manager track).

Reuses the churn project's SHAP TreeExplainer approach, reframed from
"why is this customer churning" to "why was this order flagged as
return-risk." Produces:
  1. A global summary plot -- the overall risk drivers across a batch,
     useful for the pitch/narrative.
  2. Per-order waterfall plots -- for any single order, shows exactly
     which features pushed its risk score up or down. This is the
     differentiator: most teams will show a flag, not a reason.

Uses the final tuned hyperparameters found via GridSearchCV in
model_training.py (max_depth=4, max_features=None, min_samples_leaf=1,
n_estimators=300) rather than re-running the full 216-fit grid search
every time explanations are needed.
"""

import matplotlib
matplotlib.use("Agg")  # no display needed -- saving straight to files
import matplotlib.pyplot as plt
import shap

from data_preparation import datapreparation, chronological_split
from model_utils import FINAL_PARAMS, fit_final_model, OPERATING_THRESHOLD


def build_explainer(model):
    return shap.TreeExplainer(model)


def global_summary_plot(explainer, x_test, save_path="shap_global_summary.png", max_display=15):
    """
    Bar chart of mean |SHAP value| per feature across the whole test
    batch -- shows the overall risk drivers, e.g. for a pitch slide.
    """
    shap_values = explainer.shap_values(x_test)
    # For a binary RandomForestClassifier, shap_values may come back as a
    # list [class_0_values, class_1_values] or a single 3D array depending
    # on the shap version -- normalize to the "flagged" (class 1) values.
    values_for_class_1 = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]

    plt.figure()
    shap.summary_plot(values_for_class_1, x_test, plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved global summary plot to {save_path}")


def explain_single_order(explainer, x_row, save_path="shap_order_explanation.png", max_display=10):
    """
    Waterfall plot for ONE order -- "why was this order flagged."
    x_row should be a single-row DataFrame (e.g. x_test.iloc[[i]]).
    """
    shap_values = explainer(x_row)

    # shap's Explanation object has a class dimension for classifiers;
    # select class 1 ("Returned") for a single-row Explanation.
    if shap_values.values.ndim == 3:
        single = shap_values[0, :, 1]
    else:
        single = shap_values[0]

    plt.figure()
    shap.plots.waterfall(single, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved order explanation to {save_path}")


def explain_flagged_batch(model, explainer, x_test, threshold=OPERATING_THRESHOLD,
                           output_dir=".", max_orders=5):
    """
    Saves individual waterfall explanations for the HIGHEST-RISK flagged
    orders (not just the first ones encountered). Near-threshold orders
    (e.g. risk=0.31 against a threshold of 0.3) produce small, unconvincing
    SHAP bars -- the highest-confidence flags make a far more compelling
    "why was this order flagged" showcase for a pitch.
    """
    probabilities = model.predict_proba(x_test)[:, 1]
    flagged_idx = [i for i, p in enumerate(probabilities) if p >= threshold]
    total_flagged = len(flagged_idx)

    # Sort flagged orders by risk score, descending -- highest confidence first
    flagged_idx_sorted = sorted(flagged_idx, key=lambda i: probabilities[i], reverse=True)
    top_idx = flagged_idx_sorted[:max_orders]

    print(f"\n{len(top_idx)} highest-risk flagged orders (of {total_flagged} "
          f"total flagged at threshold={threshold}):")

    for count, i in enumerate(top_idx):
        row = x_test.iloc[[i]]
        prob = probabilities[i]
        print(f"  Order test-index {i}: predicted risk = {prob:.3f}")
        explain_single_order(
            explainer, row,
            save_path=f"{output_dir}/shap_flagged_order_{count+1}.png",
        )


if __name__ == "__main__":
    df, cost_reference = datapreparation("returns_sustainability_dataset.csv")
    x_train, x_test, y_train, y_test, cost_test, cutoff = chronological_split(df, cost_reference)

    model = fit_final_model(x_train, y_train)
    explainer = build_explainer(model)

    global_summary_plot(explainer, x_test)
    explain_flagged_batch(model, explainer, x_test, threshold=OPERATING_THRESHOLD, max_orders=3)