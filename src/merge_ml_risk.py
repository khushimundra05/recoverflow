"""
merge_ml_risk.py

Merges the real ML-derived ml_customer_propensity_score (from ml_risk_model.py)
into customer_features.csv. Run AFTER ml_risk_model.py has produced
data/customer_propensity_scores.csv.

~1.3% of customers have no score (their orders were dropped during training
due to missing payment/item data or ambiguous order_status) -- filled with
the population median, logged explicitly rather than silently zero-filled.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    cf_path = DATA_DIR / "customer_features.csv"
    risk_path = DATA_DIR / "customer_propensity_scores.csv"

    if not cf_path.exists():
        raise FileNotFoundError("Run ingest_olist.py first.")
    if not risk_path.exists():
        raise FileNotFoundError("Run ml_risk_model.py first.")

    cf = pd.read_csv(cf_path)
    if "ml_customer_propensity_score" in cf.columns:
        cf = cf.drop(columns=["ml_customer_propensity_score"])  # idempotent re-run

    risk = pd.read_csv(risk_path)
    merged = cf.merge(risk[["customer_unique_id", "ml_customer_propensity_score"]],
                       on="customer_unique_id", how="left")

    missing = merged["ml_customer_propensity_score"].isna().sum()
    median_val = merged["ml_customer_propensity_score"].median()
    merged["ml_customer_propensity_score"] = merged["ml_customer_propensity_score"].fillna(median_val)

    merged.to_csv(cf_path, index=False)
    print(f"Merged ML customer propensity score into {len(merged):,} customers "
          f"({missing:,} had no score, filled with population median {median_val:.2f})")


if __name__ == "__main__":
    main()
