"""
simulate_invoices.py

Fully simulated overdue-invoice events -- no real dataset exists for this
leak type (established earlier in the project). Attached to real Olist
customers only so cross-leak deduplication has genuine overlapping cases
to work with; the invoice amounts, due dates, and existence of these
invoices are entirely synthetic.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def simulate_invoice_events(customer_features_path: Path, n_events: int = 60,
                             overlap_customer_ids: list = None) -> pd.DataFrame:
    rng = np.random.default_rng(13)  # local, deterministic regardless of import order
    real_customers = pd.read_csv(customer_features_path)

    if overlap_customer_ids:
        overlap_pool = real_customers[real_customers["customer_unique_id"].isin(overlap_customer_ids)]
        overlap_n = min(len(overlap_pool), n_events // 3)
        overlap_sample = overlap_pool.sample(n=overlap_n, random_state=13)
        remaining_n = n_events - len(overlap_sample)
        rest_pool = real_customers[~real_customers["customer_unique_id"].isin(overlap_sample["customer_unique_id"])]
        rest_sample = rest_pool.sample(n=remaining_n, random_state=13)
        sampled = pd.concat([overlap_sample, rest_sample]).reset_index(drop=True)
    else:
        sampled = real_customers.sample(n=n_events, random_state=13).reset_index(drop=True)

    events = pd.DataFrame({
        "event_id": [f"invoice_{i:04d}" for i in range(len(sampled))],
        "leak_type": "invoice_overdue",
        "reason_key": "invoice_overdue",
        "customer_unique_id": sampled["customer_unique_id"].values,
        "repeat_order_count": sampled["repeat_order_count"].values,
        "average_order_value_brl": sampled["average_order_value_brl"].values,
        "customer_tenure_days": sampled["customer_tenure_days"].values,
        "customer_value_score": sampled["customer_value_score"].values,
        "ml_customer_propensity_score": sampled["ml_customer_propensity_score"].values,
        # SIMULATED -- B2B invoice amounts tend to be larger than consumer transactions
        "amount_inr": np.round(rng.lognormal(mean=9.5, sigma=1.3, size=len(sampled)), 2),
        "days_overdue": rng.integers(1, 90, size=len(sampled)),
        "prior_attempt_count": rng.choice([0, 0, 1, 1, 2, 3], size=len(sampled)),
        "is_simulated_event": True,
    })

    return events


if __name__ == "__main__":
    events = simulate_invoice_events(DATA_DIR / "customer_features.csv", n_events=60)
    out_path = DATA_DIR / "simulated_invoice_events.csv"
    events.to_csv(out_path, index=False)
    print(f"Generated {len(events)} SIMULATED invoice events on {events['customer_unique_id'].nunique()} REAL Olist customers")
    print(f"Saved to: {out_path}")
