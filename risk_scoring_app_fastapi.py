"""
Batch-scoring API for the Return-Risk Scorer (FastAPI version).

POST /score_batch with a CSV of new orders (raw schema -- same columns
as the training data, MINUS anything only known after the fact:
Return_Status, Return_Date, Return_Reason, Days_to_Return, Return_Cost,
Profit_Loss) returns a risk score and flag decision for each order, plus
a batch summary. Flagged orders also come back with their top SHAP
contributing features -- "why was this order flagged," not just a score.

At startup (via the lifespan handler), trains the final model once using
the hyperparameters found via GridSearchCV in model_training.py, builds
the reference artifacts needed to score future orders consistently (see
deployment_utils.py for why that matters), and builds a SHAP explainer.

HONESTY NOTE ON STARTUP TIME: building the SHAP explainer requires
importing the shap library, which is genuinely slow to import (its
numba/llvmlite dependency chain), especially on Windows. This is a real,
unavoidable cost of returning genuine per-order explanations -- it is
not a regression of the earlier fast-startup fix, which was about
removing an *unnecessary* shap import when explanations weren't part of
the response at all. Set EXPLAIN_BY_DEFAULT = False below, or pass
?explain=false per request, to skip explanation computation if startup
time or per-request latency becomes a problem during a live demo.

Run with: uvicorn risk_scoring_app_fastapi:app --host 0.0.0.0 --port 7861
Interactive docs then available at http://localhost:7861/docs
"""

import io
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel

from data_preparation import datapreparation, chronological_split
from deployment_utils import build_reference_artifacts, prepare_batch_for_scoring, score_batch, OPERATING_THRESHOLD
from model_utils import fit_final_model
from shap_reasons import build_explainer, top_reasons_for_rows

RAW_DATA_PATH = "returns_sustainability_dataset.csv"
TOP_N_REASONS = 5
EXPLAIN_BY_DEFAULT = True

REQUIRED_COLUMNS = [
    "User_ID", "Product_Category", "Product_Price", "Order_Quantity",
    "Discount_Applied", "User_Age", "User_Gender", "User_Location",
    "Payment_Method", "Shipping_Method", "CO2_Emissions", "Packaging_Waste",
]

# Populated once at startup by the lifespan handler below, read by the
# endpoints. Kept as a plain dict (rather than module-level globals) so
# tests can spin the app up/down cleanly via TestClient's context manager.
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading data and training final model...")
    df_prepared, cost_reference = datapreparation(RAW_DATA_PATH)
    x_train, x_test, y_train, y_test, cost_test, cutoff = chronological_split(
        df_prepared, cost_reference
    )
    model = fit_final_model(x_train, y_train)
    model_feature_columns = list(x_train.columns)

    raw_all = pd.read_csv(RAW_DATA_PATH, parse_dates=["Order_Date"])
    raw_train = raw_all[raw_all["Order_Date"] < cutoff]
    reference = build_reference_artifacts(raw_train, model_feature_columns)

    print("Building SHAP explainer (this is the slow step, see module docstring)...")
    explainer = build_explainer(model)

    state["model"] = model
    state["reference"] = reference
    state["explainer"] = explainer
    state["trained_on_orders"] = len(x_train)
    state["cutoff_date"] = str(cutoff)
    print(f"Model ready. Trained on {len(x_train)} orders. Reference built from "
          f"{len(raw_train)} historical orders (cutoff={cutoff}).")

    yield

    state.clear()


app = FastAPI(
    title="Return-Risk Scorer API",
    description="Batch scoring for e-commerce order return risk (Razorpay Buildathon, AI Risk Manager track)",
    version="1.0.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str
    trained_on_orders: int
    cutoff_date: str
    operating_threshold: float


class FeatureContribution(BaseModel):
    feature: str
    contribution: float  # positive = pushed risk up, negative = pushed risk down
    order_value: str      # this order's actual value for that feature, as a string for display


class OrderResult(BaseModel):
    Order_ID: Optional[str] = None
    predicted_risk: float
    flagged: bool
    confidence: str
    top_reasons: Optional[List[FeatureContribution]] = None


class BatchSummary(BaseModel):
    total_orders: int
    flagged_count: int
    flag_rate: float
    avg_predicted_risk: float
    borderline_count: int
    threshold_used: float


class ScoreBatchResponse(BaseModel):
    summary: BatchSummary
    orders: List[OrderResult]


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        trained_on_orders=state["trained_on_orders"],
        cutoff_date=state["cutoff_date"],
        operating_threshold=OPERATING_THRESHOLD,
    )


@app.post("/score_batch", response_model=ScoreBatchResponse)
def score_batch_endpoint(
    file: UploadFile = File(..., description="CSV of new orders, raw schema minus outcome-only columns"),
    threshold: float = Query(OPERATING_THRESHOLD, ge=0.0, le=1.0,
                              description="Risk score cutoff for flagging an order"),
    explain: bool = Query(EXPLAIN_BY_DEFAULT,
                           description="Include top SHAP contributing features for flagged orders"),
):
    try:
        raw_bytes = file.file.read()
        raw_batch = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    if len(raw_batch) == 0:
        raise HTTPException(status_code=400, detail="Uploaded CSV has no rows")

    missing = [c for c in REQUIRED_COLUMNS if c not in raw_batch.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    try:
        results, summary = score_batch(state["model"], raw_batch, state["reference"], threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

    reasons_by_position = {}
    if explain:
        try:
            features = prepare_batch_for_scoring(raw_batch, state["reference"])
            flagged_positions = results.index[results["flagged"]].tolist()
            reasons_by_position = top_reasons_for_rows(
                state["explainer"], features, flagged_positions, top_n=TOP_N_REASONS
            )
        except Exception as e:
            # Explanation is a bonus on top of the core score -- if it fails
            # for any reason, still return the scores rather than a 500.
            print(f"Warning: SHAP explanation failed, returning scores without reasons: {e}")

    orders = []
    for i, row in enumerate(results.to_dict(orient="records")):
        contributions = reasons_by_position.get(i)
        top_reasons = None
        if contributions:
            top_reasons = [
                FeatureContribution(feature=f, contribution=round(float(c), 4), order_value=str(v))
                for f, c, v in contributions
            ]
        orders.append(OrderResult(**row, top_reasons=top_reasons))

    return ScoreBatchResponse(
        summary=BatchSummary(**summary),
        orders=orders,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7861)