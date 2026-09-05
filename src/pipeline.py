"""
pipeline.py

The full end-to-end run: score -> dedup -> rank/budget -> constrain -> execute
-> audit. This is what Day 7's "run the full batch" and the deliberate-failure
demo actually look like.

Contact history assumption (authored, not real): since we have no real
historical contact log for these customers, every customer starts this run
with zero prior contacts this week. This is stated explicitly because it's
a real limitation -- a production version would load actual contact history.

EXECUTION MODE: the same decision pipeline (score -> dedup -> rank ->
constrain) always runs identically regardless of mode. Only the executor
at the end switches:
  - mock (default): MockRazorpayClient, fully reproducible, used for the
    evaluation/comparison runs (compare_with_outcomes.py). Ledger is wiped
    each run so results are deterministic from a clean slate.
  - real: RealRazorpayClient, hits Razorpay's actual test-mode API. Ledger
    is deliberately NOT wiped in this mode -- re-running a live demo must
    not risk firing duplicate orders/payment-links at Razorpay just because
    the idempotency history got cleared.
Set via razorpay_mode="mock"/"real", or the RECOVERFLOW_RAZORPAY_MODE env var.
"""

import json
import os
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from dotenv import load_dotenv

from diagnoser import load_reason_mapping
from scorer import load_scoring_config
from customer_aggregator import score_all_events, aggregate_by_customer
from ranker import rank_and_allocate_budget, load_policy
from constraint_engine import check_constraints
from executor import (
    execute_action, MockRazorpayClient, RealRazorpayClient, LEDGER_PATH, ActionStatus,
)

DATA_DIR = Path(__file__).parent.parent / "data"
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")  # populates os.environ if .env exists

RAZORPAY_CALLABLE_ACTIONS = {"retry_after_delay", "retry_prompt", "payment_link"}


def run_full_pipeline(
    force_timeout_on_nth_razorpay_call: int = None,
    razorpay_mode: str = None,
    max_live_actions: int = None,
):
    """
    razorpay_mode: "mock" or "real". If None, reads RECOVERFLOW_RAZORPAY_MODE
        env var, defaulting to "mock" if unset.
    max_live_actions: only meaningful when razorpay_mode == "real". Caps how
        many events actually get processed this run, so a live demo doesn't
        accidentally fire dozens of real Razorpay API calls (and risk rate
        limiting). Entries are prioritized so events whose scorer-preferred
        action is Razorpay-callable (retry/payment_link) come first -- so a
        small cap still demonstrates the real integration rather than
        spending it on escalations/reminders by luck of the ranking order.
    """
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

    if razorpay_mode is None:
        razorpay_mode = os.environ.get("RECOVERFLOW_RAZORPAY_MODE", "mock").lower()

    entries_to_process = ranked.to_action_today
    if razorpay_mode == "real" and max_live_actions is not None:
        # Prioritize entries that will ACTUALLY result in a Razorpay call:
        # scorer's preferred action must be callable, AND the amount must be
        # under the human-escalation threshold (otherwise constraint_engine
        # forces escalate_human_review regardless of the preferred action --
        # high-value events tend to rank highest AND tend to trigger
        # escalation, so filtering on preferred-action alone isn't enough).
        escalation_threshold = policy["financial_limits"]["human_escalation_threshold_amount_inr"]
        likely_callable = [
            e for e in entries_to_process
            if e["scored"].best_action in RAZORPAY_CALLABLE_ACTIONS
            and e["amount_inr"] <= escalation_threshold
        ]
        entries_to_process = (likely_callable or entries_to_process)[:max_live_actions]

    # --- Constrain + execute ---
    if razorpay_mode == "mock":
        # Clear only MOCK-labeled ledger entries, for a fully reproducible,
        # deterministic evaluation run. Deliberately preserve any REAL-labeled
        # entries -- a mock evaluation run must never erase the record of
        # what was actually done against Razorpay's real API by a prior
        # live-demo run.
        if LEDGER_PATH.exists():
            with open(LEDGER_PATH) as f:
                ledger = json.load(f)
            real_only = {k: v for k, v in ledger.items() if v.get("client_used") == "real"}
            with open(LEDGER_PATH, "w") as f:
                json.dump(real_only, f, indent=2)
        client = MockRazorpayClient(force_timeout_on=force_timeout_on_nth_razorpay_call)
        client_label = "mock"
    elif razorpay_mode == "real":
        # Deliberately do NOT touch the ledger here -- idempotency must persist
        # across repeated live-demo runs, or a re-run could fire duplicate
        # real orders/payment-links at Razorpay.
        client = RealRazorpayClient()
        client_label = "real"
    else:
        raise ValueError(f"Invalid razorpay_mode: '{razorpay_mode}'. Use 'mock' or 'real'.")

    contact_history = {}  # customer_unique_id -> {"messages_sent_this_week": int, "attempts_this_event": {}}

    audit_rows = []

    for entry in entries_to_process:
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
            client=client, client_label=client_label,
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
            "razorpay_mode": client_label,
        })

    audit_df = pd.DataFrame(audit_rows)

    # Only overwrite the main audit_ledger.csv in mock mode -- a capped real
    # demo run must not clobber the full evaluation's audit trail.
    if razorpay_mode == "mock":
        audit_path = DATA_DIR / "audit_ledger.csv"
        audit_df.to_csv(audit_path, index=False)
    else:
        audit_path = DATA_DIR / "audit_ledger_live_demo.csv"
        audit_df.to_csv(audit_path, index=False)

    return {
        "audit_df": audit_df,
        "deferred_dedup": deferred_dedup,
        "deferred_budget": ranked.deferred_budget,
        "razorpay_mode": razorpay_mode,
        "ranked_summary": {
            "total_events_seen": ranked.total_events_seen,
            "actioned_today": len(ranked.to_action_today),
            "processed_this_run": len(entries_to_process),
            "budget_used_inr": ranked.budget_used_inr,
            "budget_available_inr": ranked.budget_available_inr,
        },
    }


if __name__ == "__main__":
    # Force a timeout on the 8th Razorpay-calling action, to prove the
    # graceful-failure path works on a REAL position in a REAL-sized batch,
    # not just a hand-picked isolated test. (Adjust if the batch size changes
    # the count of Razorpay-callable actions -- check with:
    #   audit_df[audit_df['final_action'].isin(['retry_after_delay','retry_prompt','payment_link'])]
    result = run_full_pipeline(force_timeout_on_nth_razorpay_call=8, razorpay_mode="mock")
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