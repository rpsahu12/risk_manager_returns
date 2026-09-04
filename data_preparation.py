"""
Data preparation for the Return-Risk Scorer (AI Risk Manager track).

Dataset: Synthetic E-Commerce Returns Management Dataset (Kaggle, sowmihari)

KEY PRINCIPLE:
A feature is only usable if it would actually be known at the moment the
order is placed -- BEFORE we know whether it gets returned. Columns like
Return_Date, Return_Reason, Days_to_Return, Return_Cost, Profit_Loss,
CO2_Saved and Waste_Avoided only exist/take meaningful values *because* a
return happened (or didn't) -- they are the label in disguise, and must
never be fed to the model as features. Return_Cost / Profit_Loss are kept
aside and used later, only for the post-hoc false-positive / false-negative
cost analysis -- not for prediction.
"""

import numpy as np
import pandas as pd

# Columns that only exist because the outcome is already known -- leakage.
# NOTE: verified empirically (not just by name) against the real file --
# CO2_Emissions and Packaging_Waste showed near-identical means across
# Return_Status groups, so they are NOT leaky and are treated as regular
# features below. Only drop columns confirmed (by name or by an actual
# groupby("Return_Status").mean() check) to depend on the return outcome.
LEAKY_COLUMNS = [
    "Return_Date",
    "Return_Reason",
    "Days_to_Return",
    "Return_Cost",
    "Profit_Loss",
    "CO2_Saved",     # only present in some file versions -- confirm before trusting
    "Waste_Avoided", # only present in some file versions -- confirm before trusting
]

# Kept alongside the target for the post-hoc false-positive/false-negative
# cost analysis. Never fed to the model.
COST_REFERENCE_COLUMNS = ["Return_Cost", "Profit_Loss"]

CATEGORICAL_COLUMNS = [
    "Product_Category",
    "Payment_Method",
    "Shipping_Method",
]

TOP_N_LOCATIONS = 15  # bucket everything outside the top-N cities as "Other"


def add_prior_return_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leakage-safe per-user return-rate feature.

    For order #N of a given user, this only uses that user's orders
    #1..N-1 -- never the current order or anything after it. First-time
    customers (no prior orders) fall back to the dataset-wide return rate.
    """
    df = df.sort_values(["User_ID", "Order_Date"]).reset_index(drop=True)

    is_returned = (df["Return_Status"] == "Returned").astype(int)
    df["_is_returned"] = is_returned

    # Orders seen so far for this user, NOT counting the current row
    df["user_prior_orders"] = df.groupby("User_ID").cumcount()

    # Returns among those prior orders (cumulative sum up to and including
    # current row, minus the current row's own outcome)
    cum_returns_incl_current = df.groupby("User_ID")["_is_returned"].cumsum()
    df["user_prior_returns"] = cum_returns_incl_current - df["_is_returned"]

    global_return_rate = is_returned.mean()

    with np.errstate(invalid="ignore", divide="ignore"):
        prior_rate = df["user_prior_returns"] / df["user_prior_orders"]

    df["user_prior_return_rate"] = prior_rate.fillna(global_return_rate)
    df["is_first_order"] = (df["user_prior_orders"] == 0).astype(int)

    # Drop user_prior_returns -- redundant with
    # user_prior_orders * user_prior_return_rate. Keeping user_prior_orders
    # (history depth) and user_prior_return_rate (the actual risk signal)
    # is enough; the raw count adds nothing distinct.
    df = df.drop(columns=["user_prior_returns"])

    df = df.drop(columns=["_is_returned"])
    return df


def bucket_rare_locations(df: pd.DataFrame, top_n: int = TOP_N_LOCATIONS) -> pd.DataFrame:
    """Keep the top-N most frequent locations, bucket the rest as 'Other'."""
    top_locations = df["User_Location"].value_counts().nlargest(top_n).index
    df["User_Location"] = df["User_Location"].where(
        df["User_Location"].isin(top_locations), other="Other"
    )
    return df


def datapreparation(filepath: str):
    """
    Loads the raw CSV and returns (model_df, cost_reference_df).

    model_df: model-ready features -- never contains a leaky column.
    cost_reference_df: Order_Date + Return_Status + the leaky columns
        (Return_Cost, Profit_Loss, CO2_Emissions, Packaging_Waste), same
        row order as model_df, for the post-hoc cost/impact analysis only.

    Mirrors the structure of the churn-project's datapreparation(): clean,
    encode, derive features -- but with the return-specific leakage guard.
    """
    df = pd.read_csv(filepath, parse_dates=["Order_Date"])

    # Leakage-safe per-user history feature -- must run on the RAW
    # Return_Status ("Returned"/"Not Returned") strings, and must happen
    # before we drop Order_Date or User_ID.
    df = add_prior_return_rate(df)

    # Target -- map to 0/1 only AFTER the history feature is computed
    df["Return_Status"] = df["Return_Status"].map({"Returned": 1, "Not Returned": 0})

    # Peel off the leaky columns into a side table BEFORE dropping them,
    # so Return_Cost/Profit_Loss survive for the cost analysis later.
    present_cost_cols = [c for c in COST_REFERENCE_COLUMNS if c in df.columns]
    cost_reference = df[["Order_Date", "Return_Status"] + present_cost_cols].copy()

    # Order value: price * quantity, adjusted for discount.
    # NOTE: confirm whether Discount_Applied is a percent (0-100) or a
    # fraction (0-1) in the real file and adjust the divisor accordingly.
    discount_frac = df["Discount_Applied"] / 100.0
    df["Order_Value"] = df["Product_Price"] * df["Order_Quantity"] * (1 - discount_frac)

    # Drop Product_Price -- it's fully recoverable from
    # Order_Value / (Order_Quantity * (1 - discount_frac)), so keeping both
    # is redundant. Order_Quantity and Discount_Applied stay: they carry
    # signal beyond total spend (bulk-buying behavior, promo sensitivity)
    # that Order_Value alone doesn't capture. Redundant features don't
    # hurt a Random Forest's accuracy much, but they DO split SHAP
    # importance across near-duplicate signals, which muddies the
    # "why was this order flagged" explanation -- and that explanation
    # is the differentiator here.
    df = df.drop(columns=["Product_Price"])

    # Simple demographic encodings, same pattern as the churn project
    df["User_Gender"] = df["User_Gender"].map({"Male": 0, "Female": 1})

    # Bucket long-tail locations before one-hot encoding
    df = bucket_rare_locations(df)

    # One-hot encode categoricals (including bucketed location)
    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS + ["User_Location"],
        drop_first=True,
    )

    # Drop identifiers and leaky columns. Keep Order_Date for the
    # chronological split -- drop it right after splitting, not here.
    id_columns = ["Order_ID", "Product_ID", "User_ID"]
    df = df.drop(columns=[c for c in id_columns if c in df.columns])
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])

    return df, cost_reference


def chronological_split(df: pd.DataFrame, cost_reference: pd.DataFrame = None, test_fraction: float = 0.2):
    """
    Splits by Order_Date instead of randomly: train on the earlier
    portion, test on the most recent slice. This matches how the model
    will actually be used in production (always predicting forward) and
    is a stronger held-out test than a random shuffle.

    If cost_reference is provided (from datapreparation), it is split
    using the exact same cutoff date so cost_test lines up row-for-row
    with x_test/y_test -- needed for the false-positive/negative cost
    table later.
    """
    df = df.sort_values("Order_Date")
    cutoff_index = int(len(df) * (1 - test_fraction))
    cutoff_date = df.iloc[cutoff_index]["Order_Date"]

    train = df[df["Order_Date"] < cutoff_date].drop(columns=["Order_Date"])
    test = df[df["Order_Date"] >= cutoff_date].drop(columns=["Order_Date"])

    x_train = train.drop(columns=["Return_Status"])
    y_train = train["Return_Status"]
    x_test = test.drop(columns=["Return_Status"])
    y_test = test["Return_Status"]

    cost_test = None
    if cost_reference is not None:
        # Align by the shared original row index rather than independently
        # re-filtering cost_reference by date. cost_reference was built
        # earlier in datapreparation() while the dataframe was still sorted
        # by [User_ID, Order_Date] (from add_prior_return_rate) -- a
        # DIFFERENT order than the Order_Date-only sort applied above. Using
        # .loc[test.index] guarantees row i of cost_test corresponds to
        # row i of x_test/y_test regardless of which sort order either
        # dataframe happens to be in.
        cost_test = cost_reference.loc[test.index].drop(columns=["Order_Date", "Return_Status"])

    return x_train, x_test, y_train, y_test, cost_test, cutoff_date


if __name__ == "__main__":
    df, cost_reference = datapreparation("returns_sustainability_dataset.csv")
    print("Prepared shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Cost reference columns:", list(cost_reference.columns))

    x_train, x_test, y_train, y_test, cost_test, cutoff = chronological_split(df, cost_reference)
    print(f"\nChronological cutoff date: {cutoff}")
    print("Train size:", x_train.shape, " Test size:", x_test.shape)
    print("Train return rate:", y_train.mean().round(3))
    print("Test return rate:", y_test.mean().round(3))
    if cost_test is not None:
        print("Cost test size:", cost_test.shape)