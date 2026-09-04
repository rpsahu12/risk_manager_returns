"""
Checks whether the chronological train/test split has a seasonal
Discount_Applied imbalance that would explain low held-out AUC despite
Discount_Applied showing a strong marginal signal overall.

Hypothesis: the cutoff date (Dec 20, 2024) sits inside a holiday
shopping window. If discounting becomes near-universal in that window,
the feature that normally separates classes cleanly (0% discount -> 0%
returns) loses its power in the test set specifically, because almost
no test-period order has 0% discount left to exploit.
"""

import pandas as pd


def compare_discount_distribution(raw_csv_path: str, test_fraction: float = 0.2):
    df = pd.read_csv(raw_csv_path, parse_dates=["Order_Date"])
    df["is_returned"] = (df["Return_Status"] == "Returned").astype(int)
    df = df.sort_values("Order_Date")

    cutoff_index = int(len(df) * (1 - test_fraction))
    cutoff = df.iloc[cutoff_index]["Order_Date"]
    print(f"Cutoff date: {cutoff}")

    train = df[df["Order_Date"] < cutoff]
    test = df[df["Order_Date"] >= cutoff]

    for label, part in [("TRAIN", train), ("TEST", test)]:
        print(f"\n--- {label}: n={len(part)} ---")
        print("Discount_Applied distribution (share of orders at each level):")
        print(part["Discount_Applied"].value_counts(normalize=True).sort_index())

        print("\nReturn rate by discount bucket:")
        bucketed = pd.cut(part["Discount_Applied"], bins=[-1, 0, 10, 20, 100])
        print(part.groupby(bucketed, observed=True)["is_returned"].mean())

    zero_discount_share_train = (train["Discount_Applied"] == 0).mean()
    zero_discount_share_test = (test["Discount_Applied"] == 0).mean()
    print(f"\nShare of orders with ZERO discount -- train: {zero_discount_share_train:.3f}, "
          f"test: {zero_discount_share_test:.3f}")
    if zero_discount_share_test < zero_discount_share_train * 0.5:
        print(">>> Test period has much less zero-discount volume than train -- "
              "this likely explains the AUC drop on held-out data.")


if __name__ == "__main__":
    compare_discount_distribution("returns_sustainability_dataset.csv")
