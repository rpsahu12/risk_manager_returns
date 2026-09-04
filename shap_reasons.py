"""
Lightweight SHAP value extraction for the batch-scoring API.

Deliberately separate from shap_explainability.py: this module imports
ONLY shap (needed to compute numeric contributions), never matplotlib --
the API needs numbers for its JSON response, never a rendered plot.

Honesty note: importing shap still carries a real, one-time startup
cost (its numba/llvmlite dependency chain is slow to import, especially
on Windows). There's no way around that if the API is going to return
genuine per-order explanations -- this module just avoids adding an
unnecessary matplotlib import on top of that unavoidable cost.
"""

from typing import List, Dict, Tuple

import pandas as pd
import shap


def build_explainer(model):
    return shap.TreeExplainer(model)


def top_reasons_for_rows(explainer, features: pd.DataFrame, row_positions: List[int],
                          top_n: int = 5) -> Dict[int, List[Tuple[str, float, object]]]:
    """
    Returns {row_position: [(feature_name, contribution, feature_value), ...]}
    for the given row positions only (e.g. just the flagged orders in a
    batch), sorted by absolute contribution, largest first. Positive
    contribution = pushed risk up; negative = pushed risk down.
    """
    if not row_positions:
        return {}

    subset = features.iloc[row_positions]
    shap_output = explainer(subset)

    values = shap_output.values
    if values.ndim == 3:
        # (rows, features, classes) -- keep the "flagged" / positive class
        values = values[:, :, 1]

    reasons = {}
    for local_i, row_pos in enumerate(row_positions):
        row_feature_values = subset.iloc[local_i]
        contributions = list(zip(features.columns, values[local_i], row_feature_values.values))
        contributions.sort(key=lambda t: abs(t[1]), reverse=True)
        reasons[row_pos] = contributions[:top_n]

    return reasons
