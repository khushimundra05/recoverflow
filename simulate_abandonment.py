"""
simulate_abandonment.py

Olist contains no pre-purchase checkout/abandonment data -- it only has
COMPLETED orders. This script generates SYNTHETIC abandonment events, but
attaches them to REAL customers (via customer_unique_id) so the customer-value
features (repeat_order_count, average_order_value, customer_tenure_days) used
for scoring are genuine, even though the abandonment event itself is not.

Every row this script produces has is_simulated_event=True. Never merge this
silently with real Razorpay payment-failure events without that flag intact --
the audit ledger and dashboard must be able to distinguish them.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
np.random.seed(42)  # reproducible for the hackathon demo


def simulate_abandonment_events(customer_features_path: Path, n_events: int = 150) -> pd.DataFrame:
    real_customers = pd.read_csv(customer_features_path)

    sampled = real_customers.sample(n=min(n_events, len(real_customers)), random_state=42).reset_index(drop=True)

    events = pd.DataFrame({
        "event_id": [f"abandon_{i:04d}" for i in range(len(sampled))],
        "leak_type": "checkout_abandonment",
        "reason_key": "checkout_abandoned",
        "customer_unique_id": sampled["customer_unique_id"],
        "repeat_order_count": sampled["repeat_order_count"],
        "average_order_value_brl": sampled["average_order_value_brl"],  # native currency, rank-only use
        "customer_tenure_days": sampled["customer_tenure_days"],
        "customer_value_score": sampled["customer_value_score"],
        # Synthetic fields below -- explicitly not derived from any real dataset
        "cart_value_inr": np.round(np.random.lognormal(mean=7.5, sigma=1.0, size=len(sampled)), 2),
        "time_since_abandonment_minutes": np.random.randint(15, 4320, size=len(sampled)),  # 15 min to 72 hrs
        "is_simulated_event": True,
    })

    return events


if __name__ == "__main__":
    events = simulate_abandonment_events(DATA_DIR / "customer_features.csv", n_events=150)
    out_path = DATA_DIR / "simulated_abandonment_events.csv"
    events.to_csv(out_path, index=False)
    print(f"Generated {len(events)} SIMULATED abandonment events on {events['customer_unique_id'].nunique()} REAL Olist customers")
    print(events.head(3).to_string())
    print(f"\nSaved to: {out_path}")
