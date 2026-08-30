"""
ranker.py

Takes the customer-deduplicated representative events (one per customer,
from customer_aggregator.py) and decides which ones actually get acted on
today, given a limited daily_recovery_budget_inr and daily_max_actions_executed
(both from config/merchant_policy.json).

Sorts by recovery_opportunity_score descending, greedily allocates budget
until either the ₹ cap or the action-count cap is hit. Events past the
cutoff are marked 'deferred_budget_exhausted' -- logged with their score so
it's visible they were seen and would be acted on with more capacity, not
silently ignored.
"""

import json
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_policy() -> dict:
    with open(CONFIG_DIR / "merchant_policy.json") as f:
        return json.load(f)


@dataclass
class RankedBatch:
    to_action_today: list
    deferred_budget: list
    total_events_seen: int
    budget_used_inr: float
    budget_available_inr: float
    actions_used: int
    actions_available: int


def rank_and_allocate_budget(representative_events: list, policy: dict = None) -> RankedBatch:
    if policy is None:
        policy = load_policy()

    capacity = policy["capacity"]
    daily_budget = capacity["daily_recovery_budget_inr"]
    max_actions = capacity["daily_max_actions_executed"]

    df = pd.DataFrame(representative_events)
    df["recovery_opportunity_score"] = df["scored"].apply(lambda s: s.recovery_opportunity_score)
    df = df.sort_values("recovery_opportunity_score", ascending=False).reset_index(drop=True)

    to_action = []
    deferred = []
    budget_used = 0.0
    actions_used = 0

    for _, row in df.iterrows():
        amount = row["amount_inr"]
        # Budget is spent against revenue AT RISK being pursued today, not
        # the intervention cost -- this models "how much recovery effort
        # capacity are we allocating," matching the portfolio framing.
        would_use_budget = budget_used + amount
        would_use_actions = actions_used + 1

        if would_use_budget <= daily_budget and would_use_actions <= max_actions:
            to_action.append(row.to_dict())
            budget_used = would_use_budget
            actions_used = would_use_actions
        else:
            deferred_row = row.to_dict()
            reason = []
            if would_use_budget > daily_budget:
                reason.append(f"would exceed daily budget (₹{daily_budget:,.0f})")
            if would_use_actions > max_actions:
                reason.append(f"would exceed daily action cap ({max_actions})")
            deferred_row["defer_reason"] = "Budget exhausted: " + " and ".join(reason)
            deferred.append(deferred_row)

    return RankedBatch(
        to_action_today=to_action,
        deferred_budget=deferred,
        total_events_seen=len(df),
        budget_used_inr=budget_used,
        budget_available_inr=daily_budget,
        actions_used=actions_used,
        actions_available=max_actions,
    )


if __name__ == "__main__":
    from customer_aggregator import score_all_events, aggregate_by_customer
    from diagnoser import load_reason_mapping
    from scorer import load_scoring_config

    DATA_DIR = Path(__file__).parent.parent / "data"
    abandonment = pd.read_csv(DATA_DIR / "simulated_abandonment_events.csv")
    payment_failures = pd.read_csv(DATA_DIR / "simulated_payment_failure_events.csv")
    invoices = pd.read_csv(DATA_DIR / "simulated_invoice_events.csv")
    all_events = pd.concat([abandonment, payment_failures, invoices], ignore_index=True)

    mapping = load_reason_mapping()
    scoring_config = load_scoring_config()
    policy = load_policy()

    scored_events = score_all_events(all_events, mapping, scoring_config)
    representative, deferred_dedup = aggregate_by_customer(scored_events)

    ranked = rank_and_allocate_budget(representative, policy)

    print(f"Total events (post-dedup, one per customer): {ranked.total_events_seen}")
    print(f"Actioned today: {len(ranked.to_action_today)}")
    print(f"Deferred (budget exhausted): {len(ranked.deferred_budget)}")
    print(f"Deferred (same-customer dedup, from earlier step): {len(deferred_dedup)}")
    print(f"Budget used: ₹{ranked.budget_used_inr:,.2f} / ₹{ranked.budget_available_inr:,.0f}")
    print(f"Actions used: {ranked.actions_used} / {ranked.actions_available}")

    print("\nTop 5 prioritized events:")
    for e in ranked.to_action_today[:5]:
        print(f"  {e['event_id']} ({e['leak_type']}) score={e['recovery_opportunity_score']:.1f} amount=₹{e['amount_inr']:,.0f}")

    if ranked.deferred_budget:
        print(f"\nHighest-scoring event that STILL got deferred (budget ran out):")
        top_deferred = max(ranked.deferred_budget, key=lambda x: x["recovery_opportunity_score"])
        print(f"  {top_deferred['event_id']} score={top_deferred['recovery_opportunity_score']:.1f} -- {top_deferred['defer_reason']}")
