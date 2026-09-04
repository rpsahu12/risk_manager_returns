"""
Batch-scoring API for the Return-Risk Scorer.

POST /score_batch with a CSV of new orders (raw schema -- same columns
as the training data, MINUS anything only known after the fact:
Return_Status, Return_Date, Return_Reason, Days_to_Return, Return_Cost,
Profit_Loss) returns a risk score and flag decision for each order, plus
a batch summary.

At startup, trains the final model once (using the hyperparameters found
via GridSearchCV in model_training.py) and builds the reference artifacts
needed to score future orders consistently -- see deployment_utils.py for
why this matters (repeat-customer history must be looked up, not
recomputed from an isolated new batch).
"""

import io
import pandas as pd
from flask import Flask, request, jsonify

from data_preparation import datapreparation, chronological_split
from deployment_utils import build_reference_artifacts, score_batch, OPERATING_THRESHOLD
from model_utils import fit_final_model

RAW_DATA_PATH = "returns_sustainability_dataset.csv"

REQUIRED_COLUMNS = [
    "User_ID", "Product_Category", "Product_Price", "Order_Quantity",
    "Discount_Applied", "User_Age", "User_Gender", "User_Location",
    "Payment_Method", "Shipping_Method", "CO2_Emissions", "Packaging_Waste",
]

app = Flask(__name__)

print("Loading data and training final model...")
_df_prepared, _cost_reference = datapreparation(RAW_DATA_PATH)
_x_train, _x_test, _y_train, _y_test, _cost_test, _cutoff = chronological_split(
    _df_prepared, _cost_reference
)
MODEL = fit_final_model(_x_train, _y_train)
MODEL_FEATURE_COLUMNS = list(_x_train.columns)

_raw_all = pd.read_csv(RAW_DATA_PATH, parse_dates=["Order_Date"])
_raw_train = _raw_all[_raw_all["Order_Date"] < _cutoff]
REFERENCE = build_reference_artifacts(_raw_train, MODEL_FEATURE_COLUMNS)
print(f"Model ready. Trained on {len(_x_train)} orders. Reference built from "
      f"{len(_raw_train)} historical orders (cutoff={_cutoff}).")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "trained_on_orders": len(_x_train),
        "cutoff_date": str(_cutoff),
        "operating_threshold": OPERATING_THRESHOLD,
    })


@app.route("/score_batch", methods=["POST"])
def score_batch_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "Upload a CSV file under the 'file' field"}), 400

    file = request.files["file"]
    try:
        raw_batch = pd.read_csv(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    if len(raw_batch) == 0:
        return jsonify({"error": "Uploaded CSV has no rows"}), 400

    missing = [c for c in REQUIRED_COLUMNS if c not in raw_batch.columns]
    if missing:
        return jsonify({"error": f"Missing required columns: {missing}"}), 400

    try:
        threshold = float(request.args.get("threshold", OPERATING_THRESHOLD))
    except ValueError:
        return jsonify({"error": "threshold must be a number"}), 400

    try:
        results, summary = score_batch(MODEL, raw_batch, REFERENCE, threshold=threshold)
    except Exception as e:
        return jsonify({"error": f"Scoring failed: {e}"}), 500

    return jsonify({
        "summary": summary,
        "orders": results.to_dict(orient="records"),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7861, debug=False)