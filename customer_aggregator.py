"""
customer_aggregator.py

A customer can have multiple open leak events simultaneously (e.g. an
abandoned cart AND an overdue invoice). Without this step, the pipeline
would independently contact them twice, same week -- exactly the "pile on"
problem flagged in the earlier review.

This groups all events by customer_unique_id, scores every event, and
picks ONE representative event per customer to act on: the highest
Recovery Opportunity Score among their open leaks. The other events for
that customer are marked 'deferred_same_customer' -- logged, not silently
dropped, so the audit trail can show they were seen and consciously
deprioritized, not missed.
"""

import pandas as pd
from dataclasses import asdict
from diagnoser import diagnose, load_reason_mapping
from scorer import score_event, load_scoring_config


def score_all_events(events_df: pd.DataFrame, mapping: dict, scoring_config: dict) -> list:
    """Runs diagnoser + scorer over every row in a combined events dataframe.
    Returns a list of dicts, one per event, with diagnosis + score attached."""
    results = []
    for _, row in events_df.iterrows():
        event = {"event_id": row["event_id"], "leak_type": row["leak_type"], "reason_key": row["reason_key"]}
        diagnosis = diagnose(event, mapping)
        scored = score_event(
            diagnosis=diagnosis,
            revenue_value_inr=row["amount_inr"],
            revenue_value_is_real=False,  # no live merchant -- see simulate_payment_failures.py note
            customer_value_score=row["customer_value_score"],
            prior_attempt_count=row.get("prior_attempt_count", 0),
            scoring_config=scoring_config,
        )
        results.append({
            "event_id": row["event_id"],
            "customer_unique_id": row["customer_unique_id"],
            "leak_type": row["leak_type"],
            "amount_inr": row["amount_inr"],
            "diagnosis": diagnosis,
            "scored": scored,
        })
    return results


def aggregate_by_customer(scored_events: list) -> tuple:
    """
    Returns (representative_events, deferred_events):
      representative_events: one per customer, the highest-scoring open leak
      deferred_events: all other events for customers who had 2+ open leaks,
                        each tagged with why it was deferred
    """
    df = pd.DataFrame(scored_events)
    df["recovery_opportunity_score"] = df["scored"].apply(lambda s: s.recovery_opportunity_score)

    representative_events = []
    deferred_events = []

    for customer_id, group in df.groupby("customer_unique_id"):
        if len(group) == 1:
            representative_events.append(group.iloc[0].to_dict())
            continue

        # Multiple open leaks for this customer -- pick the highest-scoring one
        sorted_group = group.sort_values("recovery_opportunity_score", ascending=False)
        representative_events.append(sorted_group.iloc[0].to_dict())

        for _, row in sorted_group.iloc[1:].iterrows():
            deferred = row.to_dict()
            deferred["defer_reason"] = (
                f"Customer has {len(group)} open leak events; "
                f"prioritizing event {sorted_group.iloc[0]['event_id']} "
                f"(score {sorted_group.iloc[0]['recovery_opportunity_score']:.1f}) "
                f"over this one (score {row['recovery_opportunity_score']:.1f}) "
                f"to avoid contacting the same customer multiple times."
            )
            deferred_events.append(deferred)

    return representative_events, deferred_events


if __name__ == "__main__":
    from pathlib import Path
    DATA_DIR = Path(__file__).parent.parent / "data"

    abandonment = pd.read_csv(DATA_DIR / "simulated_abandonment_events.csv")
    payment_failures = pd.read_csv(DATA_DIR / "simulated_payment_failure_events.csv")
    invoices = pd.read_csv(DATA_DIR / "simulated_invoice_events.csv")

    all_events = pd.concat([abandonment, payment_failures, invoices], ignore_index=True)
    print(f"Total events across all 3 leak types: {len(all_events)}")
    print(f"Unique customers involved: {all_events['customer_unique_id'].nunique()}")

    mapping = load_reason_mapping()
    scoring_config = load_scoring_config()

    scored_events = score_all_events(all_events, mapping, scoring_config)
    representative, deferred = aggregate_by_customer(scored_events)

    print(f"\nRepresentative events (one per customer): {len(representative)}")
    print(f"Deferred-same-customer events: {len(deferred)}")

    print("\nExample of a dedup decision:")
    if deferred:
        example = deferred[0]
        print(f"  Deferred event: {example['event_id']} ({example['leak_type']}, score {example['recovery_opportunity_score']:.1f})")
        print(f"  Reason: {example['defer_reason']}")
