"""
pipeline.py

The full end-to-end run: score -> dedup -> rank/budget -> constrain -> execute
-> audit. This is what Day 7's "run the full batch" and the deliberate-failure
demo actually look like.

Contact history assumption (authored, not real): since we have no real
historical contact log for these customers, every customer starts this run
with zero prior contacts this week. This is stated explicitly because it's
a real limitation -- a production version would load actual contact history.
"""

import json
import pandas as pd
from pathlib import Path
from dataclasses import asdict

from diagnoser import load_reason_mapping
from scorer import load_scoring_config
from customer_aggregator import score_all_events, aggregate_by_customer
from ranker import rank_and_allocate_budget, load_policy
from constraint_engine import check_constraints
from executor import execute_action, MockRazorpayClient, LEDGER_PATH, ActionStatus

DATA_DIR = Path(__file__).parent.parent / "data"


def run_full_pipeline(force_timeout_on_nth_razorpay_call: int = None):
    # --- Load real+simulated event batch ---
    abandonment = pd.read_csv(DATA_DIR / "simulated_abandonment_events.csv")
    payment_failures = pd.read_csv(DATA_DIR / "simulated_payment_failure_events.csv")
    invoices = pd.read_csv(DATA_DIR / "simulated_invoice_events.csv")
    all_events = pd.concat([abandonment, payment_failures, invoices], ignore_index=True)

    mapping = load_reason_mapping()
    scoring_config = load_scoring_config()
    policy = load_policy()

    # --- Score every event, dedup by customer, allocate budget ---
    scored_events = score_all_events(all_events, mapping, scoring_config)
    representative, deferred_dedup = aggregate_by_customer(scored_events)
    ranked = rank_and_allocate_budget(representative, policy)

    # --- Constrain + execute the events that made the budget cut ---
    if LEDGER_PATH.exists():
        LEDGER_PATH.unlink()  # clean ledger for a reproducible full-batch demo run

    client = MockRazorpayClient(force_timeout_on=force_timeout_on_nth_razorpay_call)
    contact_history = {}  # customer_unique_id -> {"messages_sent_this_week": int, "attempts_this_event": {}}

    audit_rows = []

    for entry in ranked.to_action_today:
        customer_id = entry["customer_unique_id"]
        diagnosis = entry["diagnosis"]
        scored = entry["scored"]
        amount = entry["amount_inr"]

        hist = contact_history.setdefault(customer_id, {"messages_sent_this_week": 0, "hours_since_last_contact": None})
        constrained = check_constraints(
            scored, revenue_value_inr=amount,
            customer_contact_history={
                "messages_sent_this_week": hist["messages_sent_this_week"],
                "hours_since_last_contact": hist["hours_since_last_contact"],
                "attempts_this_event": entry.get("prior_attempt_count", 0),
            },
            policy=policy,
        )

        exec_result = execute_action(
            constrained, customer_id=customer_id, amount_inr=amount,
            client=client, client_label="mock",
        )

        is_contact_action = constrained.final_action in ("reminder", "payment_link", "alternate_method_prompt", "retry_prompt")
        if is_contact_action and exec_result.status == ActionStatus.COMPLETED:
            hist["messages_sent_this_week"] += 1
            hist["hours_since_last_contact"] = 0

        audit_rows.append({
            "event_id": entry["event_id"],
            "customer_unique_id": customer_id,
            "leak_type": entry["leak_type"],
            "amount_inr": amount,
            "root_cause": diagnosis.root_cause,
            "diagnosis_source": diagnosis.source,
            "confidence": diagnosis.confidence,
            "recovery_opportunity_score": scored.recovery_opportunity_score,
            "scorer_preferred_action": scored.best_action,
            "final_action": constrained.final_action,
            "was_overridden_by_policy": constrained.was_overridden,
            "blocked_actions": json.dumps(constrained.blocked_actions),
            "requires_human_review": constrained.requires_human_review,
            "execution_status": exec_result.status.value,
            "external_reference": exec_result.external_reference,
            "execution_error": exec_result.error_message,
            "was_deduplicated": exec_result.was_deduplicated,
        })

    audit_df = pd.DataFrame(audit_rows)
    audit_path = DATA_DIR / "audit_ledger.csv"
    audit_df.to_csv(audit_path, index=False)

    return {
        "audit_df": audit_df,
        "deferred_dedup": deferred_dedup,
        "deferred_budget": ranked.deferred_budget,
        "ranked_summary": {
            "total_events_seen": ranked.total_events_seen,
            "actioned_today": len(ranked.to_action_today),
            "budget_used_inr": ranked.budget_used_inr,
            "budget_available_inr": ranked.budget_available_inr,
        },
    }


if __name__ == "__main__":
    # Force a timeout on the 15th Razorpay-calling action, to prove the
    # graceful-failure path works on a REAL position in a REAL-sized batch,
    # not just a hand-picked isolated test.
    result = run_full_pipeline(force_timeout_on_nth_razorpay_call=15)
    audit_df = result["audit_df"]

    print("=" * 70)
    print("FULL BATCH PIPELINE RESULTS")
    print("=" * 70)
    print(f"Total events seen (post-dedup): {result['ranked_summary']['total_events_seen']}")
    print(f"Actioned today: {result['ranked_summary']['actioned_today']}")
    print(f"Budget used: ₹{result['ranked_summary']['budget_used_inr']:,.2f} / ₹{result['ranked_summary']['budget_available_inr']:,.0f}")
    print(f"Deferred (same-customer dedup): {len(result['deferred_dedup'])}")
    print(f"Deferred (budget exhausted): {len(result['deferred_budget'])}")

    print("\n--- Execution status breakdown ---")
    print(audit_df["execution_status"].value_counts())

    print("\n--- Policy override breakdown ---")
    print(audit_df["was_overridden_by_policy"].value_counts())

    print("\n--- Human escalations forced ---")
    print(audit_df["requires_human_review"].value_counts())

    print("\n--- The deliberate failure case ---")
    timeout_rows = audit_df[audit_df["execution_status"] == "pending_verification"]
    if len(timeout_rows) > 0:
        r = timeout_rows.iloc[0]
        print(f"Event {r['event_id']}: status={r['execution_status']}, error='{r['execution_error']}'")
        print("This event will NOT be auto-retried by the pipeline -- it needs manual/verifier check.")
    else:
        print("(No timeout triggered in this run -- adjust force_timeout_on_nth_razorpay_call)")

    print(f"\nFull audit ledger saved to: {DATA_DIR / 'audit_ledger.csv'}")
