"""
simulate_payment_failures.py

Generates synthetic payment-failure events. What's real here: the reason_key
values are Razorpay's actual documented reason codes (see reason_mapping.json).
What's simulated: that these specific failures happened at all, and the ₹
amounts attached to them -- we have no live merchant, so there is no real
transaction amount anywhere in this project. This corrects an earlier error
where test amounts were mislabeled revenue_value_is_real=True.

Events are attached to REAL Olist customers (via customer_unique_id) so the
customer-value features are genuine, same pattern as simulate_abandonment.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Roughly realistic relative frequencies -- authored, not measured (we have no
# real failure-rate data by reason). insufficient_funds and payment_cancelled
# dominate in most public payment-failure discussions; fraud is rare.
REASON_WEIGHTS = {
    "insufficient_funds": 0.28,
    "payment_cancelled": 0.20,
    "payment_timedout": 0.12,
    "bank_declined": 0.15,
    "authentication_failed": 0.13,
    "gateway_error": 0.07,
    "upi_provider_downtime": 0.04,
    "fraud_suspected_by_bank": 0.01,
}


def simulate_payment_failure_events(customer_features_path: Path, n_events: int = 150,
                                     overlap_customer_ids: list = None) -> pd.DataFrame:
    rng = np.random.default_rng(7)  # local, deterministic regardless of import order
    real_customers = pd.read_csv(customer_features_path)

    if overlap_customer_ids:
        # Deliberately include some customers who ALSO appear in other leak
        # types, so customer_aggregator.py has real cross-leak cases to dedup.
        overlap_n = min(len(overlap_customer_ids), n_events // 4)
        overlap_sample = real_customers[real_customers["customer_unique_id"].isin(overlap_customer_ids)].sample(
            n=min(overlap_n, (real_customers["customer_unique_id"].isin(overlap_customer_ids)).sum()), random_state=7
        )
        remaining_n = n_events - len(overlap_sample)
        rest_pool = real_customers[~real_customers["customer_unique_id"].isin(overlap_sample["customer_unique_id"])]
        rest_sample = rest_pool.sample(n=remaining_n, random_state=7)
        sampled = pd.concat([overlap_sample, rest_sample]).reset_index(drop=True)
    else:
        sampled = real_customers.sample(n=n_events, random_state=7).reset_index(drop=True)

    reason_keys = rng.choice(
        list(REASON_WEIGHTS.keys()), size=len(sampled), p=list(REASON_WEIGHTS.values())
    )

    events = pd.DataFrame({
        "event_id": [f"payfail_{i:04d}" for i in range(len(sampled))],
        "leak_type": "payment_failure",
        "reason_key": reason_keys,
        "customer_unique_id": sampled["customer_unique_id"].values,
        "repeat_order_count": sampled["repeat_order_count"].values,
        "average_order_value_brl": sampled["average_order_value_brl"].values,
        "customer_tenure_days": sampled["customer_tenure_days"].values,
        "customer_value_score": sampled["customer_value_score"].values,
        "ml_customer_propensity_score": sampled["ml_customer_propensity_score"].values,
        # SIMULATED -- no live merchant means no real transaction amount exists anywhere.
        "amount_inr": np.round(rng.lognormal(mean=8.2, sigma=1.1, size=len(sampled)), 2),
        "prior_attempt_count": rng.choice([0, 0, 0, 1, 1, 2], size=len(sampled)),
        "is_simulated_event": True,
        "reason_code_is_real_razorpay_vocab": True,
    })

    return events


if __name__ == "__main__":
    events = simulate_payment_failure_events(DATA_DIR / "customer_features.csv", n_events=150)
    out_path = DATA_DIR / "simulated_payment_failure_events.csv"
    events.to_csv(out_path, index=False)
    print(f"Generated {len(events)} SIMULATED payment-failure events on {events['customer_unique_id'].nunique()} REAL Olist customers")
    print(events["reason_key"].value_counts())
    print(f"\nSaved to: {out_path}")
