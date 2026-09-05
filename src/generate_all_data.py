"""
generate_all_data.py

Single entry point to regenerate all three synthetic event batches with
deliberate customer overlap (for the cross-leak dedup demo). Run this AFTER
ingest_olist.py has produced data/customer_features.csv.

This replaces the ad-hoc sequence of commands run during development, so the
whole batch is reproducible from a clean checkout with one command.
"""

from pathlib import Path
import pandas as pd

from simulate_abandonment import simulate_abandonment_events
from simulate_payment_failures import simulate_payment_failure_events
from simulate_invoices import simulate_invoice_events

DATA_DIR = Path(__file__).parent.parent / "data"


def main(n_abandonment: int = 1000, n_payment_failures: int = 1000, n_invoices: int = 400):
    customer_features_path = DATA_DIR / "customer_features.csv"
    if not customer_features_path.exists():
        raise FileNotFoundError(
            "data/customer_features.csv not found. Run `python src/ingest_olist.py` first "
            "(after placing the 4 Olist CSVs in data/olist/)."
        )

    # 1. Abandonment events first -- their customer IDs become the overlap pool
    abandonment = simulate_abandonment_events(customer_features_path, n_events=n_abandonment)
    abandonment.to_csv(DATA_DIR / "simulated_abandonment_events.csv", index=False)
    overlap_ids = abandonment["customer_unique_id"].tolist()
    print(f"Abandonment events: {len(abandonment)} (customer overlap pool: {len(overlap_ids)})")

    # 2. Payment failures and invoices deliberately reuse some of those customer IDs
    payment_failures = simulate_payment_failure_events(
        customer_features_path, n_events=n_payment_failures, overlap_customer_ids=overlap_ids
    )
    payment_failures.to_csv(DATA_DIR / "simulated_payment_failure_events.csv", index=False)
    print(f"Payment failure events: {len(payment_failures)}")

    invoices = simulate_invoice_events(
        customer_features_path, n_events=n_invoices, overlap_customer_ids=overlap_ids
    )
    invoices.to_csv(DATA_DIR / "simulated_invoice_events.csv", index=False)
    print(f"Invoice events: {len(invoices)}")

    all_events = pd.concat([abandonment, payment_failures, invoices], ignore_index=True)
    dupe_counts = all_events["customer_unique_id"].value_counts()
    multi_leak = (dupe_counts > 1).sum()
    print(f"\nTotal events: {len(all_events)}")
    print(f"Unique customers: {all_events['customer_unique_id'].nunique()}")
    print(f"Customers with 2+ open leaks (real dedup cases): {multi_leak}")


if __name__ == "__main__":
    main()