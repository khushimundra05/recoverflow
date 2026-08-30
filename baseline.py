"""
baseline.py

The naive strategy every merchant runs today, without an operator:
  - Retry every failed payment immediately (no diagnosis, no cooldown check)
  - Send every abandonment/invoice the same reminder, regardless of value
  - No cross-leak dedup -- a customer with 2 open leaks gets contacted twice
  - No budget cap -- act on everything, no prioritization
  - No escalation threshold -- large invoices get the same reminder as small carts

Run over the IDENTICAL event batch as pipeline.py, so the comparison is
apples-to-apples. This produces the numbers for the agent-vs-baseline table.
"""

import pandas as pd
from pathlib import Path
from executor import execute_action, MockRazorpayClient, ActionStatus
from constraint_engine import ConstraintResult  # reuse the dataclass shape so execute_action works unmodified

DATA_DIR = Path(__file__).parent.parent / "data"

# Naive fixed action per leak type -- no diagnosis, no scoring, no per-event judgment
NAIVE_ACTION_MAP = {
    "payment_failure": "retry_after_delay",
    "checkout_abandonment": "reminder",
    "invoice_overdue": "reminder",
}


def run_naive_baseline(all_events: pd.DataFrame, force_timeout_on_nth_razorpay_call: int = None) -> pd.DataFrame:
    client = MockRazorpayClient(force_timeout_on=force_timeout_on_nth_razorpay_call)
    rows = []

    for _, row in all_events.iterrows():
        action = NAIVE_ACTION_MAP[row["leak_type"]]

        # Fake a ConstraintResult shape so execute_action() can be reused as-is --
        # naive strategy skips constraint checking entirely (that's the point).
        fake_constrained = ConstraintResult(
            event_id=row["event_id"], original_best_action=action, original_best_score=0,
            allowed_actions={action: 0}, blocked_actions={}, final_action=action,
            final_action_score=0, was_overridden=False, requires_human_review=False,
        )

        exec_result = execute_action(
            fake_constrained, customer_id=row["customer_unique_id"], amount_inr=row["amount_inr"],
            client=client, client_label="mock",
        )

        rows.append({
            "event_id": row["event_id"],
            "customer_unique_id": row["customer_unique_id"],
            "leak_type": row["leak_type"],
            "amount_inr": row["amount_inr"],
            "action_taken": action,
            "execution_status": exec_result.status.value,
            "required_human_review_but_didnt_get_one": row["amount_inr"] > 25000,  # policy the naive strategy IGNORES
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    abandonment = pd.read_csv(DATA_DIR / "simulated_abandonment_events.csv")
    payment_failures = pd.read_csv(DATA_DIR / "simulated_payment_failure_events.csv")
    invoices = pd.read_csv(DATA_DIR / "simulated_invoice_events.csv")
    all_events = pd.concat([abandonment, payment_failures, invoices], ignore_index=True)

    baseline_df = run_naive_baseline(all_events, force_timeout_on_nth_razorpay_call=15)
    baseline_df.to_csv(DATA_DIR / "baseline_ledger.csv", index=False)

    print(f"Naive baseline processed ALL {len(baseline_df)} events (no dedup, no budget cap)")
    print(f"Total ₹ 'attempted' (no prioritization): ₹{baseline_df['amount_inr'].sum():,.2f}")
    print(f"Messages/retries sent: {len(baseline_df[baseline_df['action_taken'] != 'no_action'])}")

    contact_counts = baseline_df["customer_unique_id"].value_counts()
    multi_contacted = (contact_counts > 1).sum()
    print(f"Customers contacted MULTIPLE times same day (no dedup): {multi_contacted}")

    ignored_escalations = baseline_df["required_human_review_but_didnt_get_one"].sum()
    print(f"High-value events (>₹25,000) auto-actioned WITHOUT human review: {ignored_escalations}")

    print(f"\nSaved to: {DATA_DIR / 'baseline_ledger.csv'}")
