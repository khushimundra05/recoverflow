"""
ingest_olist.py

Computes REAL per-customer features from the Olist dataset, used by scorer.py
as the customer_value_inputs referenced in config/scoring_config.json:
  - repeat_order_count
  - average_order_value
  - customer_tenure_days

IMPORTANT (per project data-honesty policy):
- customer_id in Olist is PER-ORDER, not a persistent customer identity.
  customer_unique_id is the real repeat-customer key -- use that, not customer_id,
  or repeat_order_count will always read as 1 and silently be wrong.
- order_status values here are Olist's real fulfillment states (delivered,
  shipped, canceled, etc.) -- NOT payment failure or checkout abandonment.
  This script does NOT invent abandonment events. That happens in a separate,
  clearly-labeled simulation step (see simulate_abandonment.py, next file).
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "olist"


def load_raw():
    orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv",
                          parse_dates=["order_purchase_timestamp"])
    customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
    payments = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
    return orders, customers, payments


def build_customer_features(orders: pd.DataFrame,
                             customers: pd.DataFrame,
                             payments: pd.DataFrame,
                             as_of: pd.Timestamp = None) -> pd.DataFrame:
    """
    Returns one row per customer_unique_id with real, defensible features:
      - repeat_order_count: count of DISTINCT orders by this real customer
      - average_order_value: mean total payment_value across their orders
      - customer_tenure_days: days between their first order and `as_of`
                               (defaults to the max timestamp in the dataset,
                               i.e. treats the dataset's own end date as "today")
    """
    if as_of is None:
        as_of = orders["order_purchase_timestamp"].max()

    # order_value per order = sum of payment_value across payment rows for that order
    # (an order can have multiple payment rows, e.g. voucher + credit card)
    order_value = payments.groupby("order_id")["payment_value"].sum().reset_index()
    order_value = order_value.rename(columns={"payment_value": "order_total_value"})

    merged = orders.merge(customers, on="customer_id", how="left")
    merged = merged.merge(order_value, on="order_id", how="left")

    features = merged.groupby("customer_unique_id").agg(
        repeat_order_count=("order_id", "nunique"),
        average_order_value_brl=("order_total_value", "mean"),  # NOTE: BRL, not INR --
        # kept in native currency deliberately. Only the *rank* of this value is used
        # downstream (see normalize_customer_value), so cross-currency comparison is
        # never actually performed. Do NOT relabel this column as INR or display the
        # raw number next to a rupee sign anywhere in the demo.
        first_order_at=("order_purchase_timestamp", "min"),
    ).reset_index()

    features["customer_tenure_days"] = (as_of - features["first_order_at"]).dt.days
    features = features.drop(columns=["first_order_at"])

    return features


def normalize_customer_value(features: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the three raw features to 0-100 each using percentile rank,
    matching scoring_config.json's stated 0-100 normalization convention.
    Percentile rank (not min-max) is used because these distributions are
    heavily right-skewed (a few customers with many repeat orders) -- min-max
    would compress almost everyone near 0.
    """
    out = features.copy()
    for col in ["repeat_order_count", "average_order_value_brl", "customer_tenure_days"]:
        score_name = col.replace("_brl", "") + "_score"
        out[score_name] = (out[col].rank(pct=True) * 100).round(1)

    out["customer_value_score"] = out[
        ["repeat_order_count_score", "average_order_value_score", "customer_tenure_days_score"]
    ].mean(axis=1).round(1)

    return out


if __name__ == "__main__":
    orders, customers, payments = load_raw()
    features = build_customer_features(orders, customers, payments)
    scored = normalize_customer_value(features)

    print(f"Loaded {len(orders):,} real Olist orders, "
          f"{scored['customer_unique_id'].nunique():,} unique real customers")
    print(scored.describe())

    out_path = Path(__file__).parent.parent / "data" / "customer_features.csv"
    scored.to_csv(out_path, index=False)
    print(f"\nSaved real customer features to: {out_path}")
