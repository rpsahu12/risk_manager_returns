"""
Deployment utilities for the Return-Risk Scorer batch-scoring endpoint.

The key correctness problem: NEW orders being scored have no known return
outcome, so user_prior_return_rate can't be recomputed the way it was
during training (which relied on knowing each row's own Return_Status to
build up cumulative history). Instead, a customer's historical risk
profile is precomputed ONCE from the training period and looked up for
each new order -- exactly like a real production system would maintain a
running customer-risk table rather than recompute history from scratch
on every scoring request.

New orders also need their categorical encoding aligned EXACTLY to what
the model was trained on: same top-N location bucketing (not recomputed
from a small new batch, which would produce meaningless buckets), same
one-hot columns in the same order (via reindex), so predict_proba never
sees a mismatched feature vector.
"""

import numpy as np
import pandas as pd

from data_preparation import LEAKY_COLUMNS, CATEGORICAL_COLUMNS, TOP_N_LOCATIONS

OPERATING_THRESHOLD = 0.3  # chosen and justified in the debugging trail (Phase 10)
BORDERLINE_BAND = 0.05     # predictions within +/- this of the threshold are flagged "borderline"


def build_reference_artifacts(train_raw_df: pd.DataFrame, model_feature_columns: list,
                               top_n_locations: int = TOP_N_LOCATIONS) -> dict:
    """
    Computes everything needed to score NEW orders consistently with how
    the model was trained, using only the raw training-period data (the
    same rows the model actually learned from -- never anything from the
    test period, to avoid leaking future information into the reference).
    """
    df = train_raw_df.copy()
    df["is_returned"] = (df["Return_Status"] == "Returned").astype(int)

    user_history = df.groupby("User_ID").agg(
        total_orders=("is_returned", "count"),
        total_returns=("is_returned", "sum"),
    )
    user_history["return_rate"] = user_history["total_returns"] / user_history["total_orders"]

    location_top_n = df["User_Location"].value_counts().nlargest(top_n_locations).index.tolist()
    global_return_rate = float(df["is_returned"].mean())

    return {
        "user_history": user_history,
        "location_top_n": location_top_n,
        "global_return_rate": global_return_rate,
        "model_feature_columns": list(model_feature_columns),
    }


def prepare_batch_for_scoring(raw_batch_df: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """
    Transforms a batch of NEW raw orders into the exact feature format
    the model expects, using the reference artifacts instead of
    recomputing anything from the batch itself.
    """
    df = raw_batch_df.copy()
    history = reference["user_history"]

    def lookup(user_id, field, default):
        if user_id in history.index:
            return history.loc[user_id, field]
        return default

    df["user_prior_orders"] = df["User_ID"].map(lambda u: lookup(u, "total_orders", 0))
    df["user_prior_return_rate"] = df["User_ID"].map(
        lambda u: lookup(u, "return_rate", reference["global_return_rate"])
    )
    df["is_first_order"] = (df["user_prior_orders"] == 0).astype(int)

    discount_frac = df["Discount_Applied"] / 100.0
    df["Order_Value"] = df["Product_Price"] * df["Order_Quantity"] * (1 - discount_frac)
    df = df.drop(columns=["Product_Price"])

    df["User_Gender"] = df["User_Gender"].map({"Male": 0, "Female": 1})

    df["User_Location"] = df["User_Location"].where(
        df["User_Location"].isin(reference["location_top_n"]), other="Other"
    )

    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS + ["User_Location"], drop_first=True)

    drop_cols = ["Order_ID", "Product_ID", "User_ID", "Order_Date", "Return_Status"] + LEAKY_COLUMNS
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Align to the exact columns/order the model was trained on: any
    # category missing from this batch is filled with 0 (correct -- no
    # row has that category), any category this batch has that training
    # never saw is safely dropped by reindex.
    df = df.reindex(columns=reference["model_feature_columns"], fill_value=0)

    return df


def score_batch(model, raw_batch_df: pd.DataFrame, reference: dict,
                 threshold: float = OPERATING_THRESHOLD):
    """
    Returns (per_order_results_df, summary_dict) for a batch of new orders.
    """
    features = prepare_batch_for_scoring(raw_batch_df, reference)
    probabilities = model.predict_proba(features)[:, 1]
    flagged = probabilities >= threshold

    if "Order_ID" in raw_batch_df.columns:
        results = raw_batch_df[["Order_ID"]].reset_index(drop=True).copy()
    else:
        results = pd.DataFrame(index=range(len(raw_batch_df)))

    results["predicted_risk"] = np.round(probabilities, 4)
    results["flagged"] = flagged
    results["confidence"] = np.where(
        np.abs(probabilities - threshold) < BORDERLINE_BAND, "borderline", "clear"
    )

    summary = {
        "total_orders": int(len(raw_batch_df)),
        "flagged_count": int(flagged.sum()),
        "flag_rate": round(float(flagged.mean()), 4),
        "avg_predicted_risk": round(float(probabilities.mean()), 4),
        "borderline_count": int((results["confidence"] == "borderline").sum()),
        "threshold_used": threshold,
    }

    return results, summary
