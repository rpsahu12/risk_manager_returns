"""
Final model configuration, kept deliberately dependency-light.

This module intentionally imports ONLY scikit-learn. fit_final_model()
is needed by both the SHAP explainability script AND the batch-scoring
API -- but the API should never have to pay the cost of importing shap
(and its numba/llvmlite dependency chain, which is notoriously slow to
import, especially on Windows) just to reuse this one function. Keep
this file free of shap/matplotlib imports.
"""

from sklearn.ensemble import RandomForestClassifier

# Final hyperparameters found via GridSearchCV in model_training.py
# (see the debugging trail, Phase 9).
FINAL_PARAMS = dict(
    max_depth=4,
    max_features=None,
    min_samples_leaf=1,
    n_estimators=300,
    random_state=42,
)

# Operating threshold chosen and justified in the debugging trail, Phase 10.
OPERATING_THRESHOLD = 0.3


def fit_final_model(x_train, y_train):
    model = RandomForestClassifier(**FINAL_PARAMS)
    model.fit(x_train, y_train)
    return model
