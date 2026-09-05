"""
outcome_simulator.py

Closes the loop the track bar explicitly asks for: "measured money
recovered across a batch." Without this, the pipeline detects, diagnoses,
scores, and acts -- but never observes whether any action actually worked.

CRITICAL DESIGN RULE: this ground-truth function must be INDEPENDENT of
scorer.py's logic. If we reused recovery_opportunity_score or confidence
values directly as the "true" probability, the evaluation would be
circular -- we'd just be proving our own assumptions back at ourselves.
So this file has its own, separately-authored probability table, with its
own noise, and does not import anything from scorer.py or reason_mapping.json.

HONESTY STATEMENT (state this explicitly in the README/demo):
"Recovered amounts are measured in a deterministic simulated evaluation
environment we built independently of the decision engine, not from real
merchant transactions. No public dataset exposes real intervention→outcome
labels, so this is the closest defensible substitute for evaluating whether
better decisions produce better outcomes."

CALIBRATION CHECK AGAINST REAL RAZORPAY-SPECIFIC PUBLISHED BENCHMARKS:
This table was authored independently, then checked (not tuned to fit)
against real published statistics:
  - Razorpay's own blog states automated retry recovers 15-20% of failed
    transactions overall. Our naive (retry-everything) strategy, run over
    the full simulated batch, produced 19.1% -- within this real range.
  - Razorpay's own blog states customer-side issues cause 67.5% of payment
    failures vs 27.7% ecosystem-side. Our REASON_WEIGHTS in
    simulate_payment_failures.py splits ~73%/27% -- close independent match.
  - A third-party Razorpay-specific report (Recurflux, 2026 -- vendor
    source, treat with same skepticism as any commercial benchmark) puts
    median recovery at 30-45% for processor-native-retry-only strategies,
    and 65-75% for a full multi-layer recovery stack (card updaters,
    multi-channel messaging, etc., which this project does not implement).
    Our operator produced 54.5% -- a defensible middle position: clearly
    better than basic retry, not overclaiming a fully-loaded commercial stack.
Sources: razorpay.com/blog/payment-success-rate-optimization-india,
razorpay.com/blog/transaction-success-rate-what-it-is-and-why-it-matters,
recurflux.com/resources/saas-payment-failure-report (2026).
"""

import hashlib
import numpy as np
import pandas as pd

# Independently authored base recovery probabilities per (reason_key, action)
# combo. Deliberately NOT copied from reason_mapping.json's confidence values --
# written fresh, reflecting plausible real-world dynamics (temporary/technical
# failures recover well with a delayed retry; insufficient funds recovers
# poorly on an immediate retry but better after a delay; fraud-flagged
# events recover poorly regardless of action).
BASE_RECOVERY_PROB = {
    ("insufficient_funds", "retry_after_delay"): 0.55,
    ("insufficient_funds", "payment_link"): 0.45,
    ("insufficient_funds", "alternate_method_prompt"): 0.40,
    ("insufficient_funds", "no_action"): 0.05,
    ("insufficient_funds", "escalate_human_review"): 0.35,

    ("payment_cancelled", "payment_link"): 0.30,
    ("payment_cancelled", "reminder"): 0.22,
    ("payment_cancelled", "no_action"): 0.05,
    ("payment_cancelled", "escalate_human_review"): 0.28,

    ("payment_timedout", "retry_prompt"): 0.60,
    ("payment_timedout", "payment_link"): 0.50,
    ("payment_timedout", "no_action"): 0.05,
    ("payment_timedout", "escalate_human_review"): 0.35,

    ("bank_declined", "alternate_method_prompt"): 0.35,
    ("bank_declined", "payment_link"): 0.28,
    ("bank_declined", "no_action"): 0.05,
    ("bank_declined", "escalate_human_review"): 0.30,

    ("fraud_suspected_by_bank", "alternate_method_prompt"): 0.15,
    ("fraud_suspected_by_bank", "no_action"): 0.05,
    ("fraud_suspected_by_bank", "escalate_human_review"): 0.20,

    ("gateway_error", "retry_after_delay"): 0.70,   # infra issue, not customer fault -- recovers well
    ("gateway_error", "payment_link"): 0.50,
    ("gateway_error", "no_action"): 0.05,
    ("gateway_error", "escalate_human_review"): 0.40,

    ("upi_provider_downtime", "retry_after_delay"): 0.68,
    ("upi_provider_downtime", "alternate_method_prompt"): 0.45,
    ("upi_provider_downtime", "no_action"): 0.05,
    ("upi_provider_downtime", "escalate_human_review"): 0.38,

    ("authentication_failed", "retry_prompt"): 0.58,
    ("authentication_failed", "payment_link"): 0.48,
    ("authentication_failed", "no_action"): 0.05,
    ("authentication_failed", "escalate_human_review"): 0.32,

    ("checkout_abandoned", "reminder"): 0.18,
    ("checkout_abandoned", "payment_link"): 0.25,
    ("checkout_abandoned", "no_action"): 0.03,
    ("checkout_abandoned", "escalate_human_review"): 0.20,

    ("invoice_overdue", "reminder"): 0.30,
    ("invoice_overdue", "payment_plan_offer"): 0.45,
    ("invoice_overdue", "escalate_human_review"): 0.40,
    ("invoice_overdue", "no_action"): 0.05,
}

DEFAULT_PROB = 0.15  # unmapped (reason, action) combos -- conservative fallback


def _deterministic_rng(event_id: str, action: str) -> np.random.Generator:
    """Stable per-(event, action) seed so repeated runs reproduce identically,
    while still giving each event+action its own independent random draw."""
    seed_material = f"{event_id}|{action}".encode()
    seed = int(hashlib.sha256(seed_material).hexdigest(), 16) % (2**32)
    return np.random.default_rng(seed)


def true_recovery_probability(reason_key: str, action: str, prior_attempt_count: int,
                                customer_value_score: float, time_factor_minutes_or_days: float = None) -> float:
    """
    Independent ground-truth probability. NOT derived from scorer.py.
    """
    base = BASE_RECOVERY_PROB.get((reason_key, action), DEFAULT_PROB)

    # Diminishing returns on repeated attempts -- each prior failed attempt
    # on the SAME event reduces the true probability (authored, modest effect)
    attempt_decay = 0.90 ** prior_attempt_count
    prob = base * attempt_decay

    # Small positive effect for higher-value/more-established customers
    # (more likely to have a working payment method on file, etc.)
    customer_adjustment = (customer_value_score - 50) / 50 * 0.05  # max +/-0.05
    prob += customer_adjustment

    # Time decay for abandonment (longer since abandonment = lower recovery)
    # and for overdue invoices (more overdue = lower recovery), both authored.
    if time_factor_minutes_or_days is not None:
        if reason_key == "checkout_abandoned":
            hours = time_factor_minutes_or_days / 60
            prob *= max(0.4, 1 - hours / 100)  # decays toward 0.4x over ~4 days
        elif reason_key == "invoice_overdue":
            prob *= max(0.35, 1 - time_factor_minutes_or_days / 150)  # decays over ~150 days

    return float(np.clip(prob, 0.02, 0.90))


def simulate_outcome(event_id: str, reason_key: str, action: str, amount_inr: float,
                     prior_attempt_count: int, customer_value_score: float,
                     time_factor: float = None) -> dict:
    prob = true_recovery_probability(reason_key, action, prior_attempt_count, customer_value_score, time_factor)
    rng = _deterministic_rng(event_id, action)
    recovered = bool(rng.random() < prob)
    return {
        "event_id": event_id,
        "action": action,
        "true_recovery_probability": round(prob, 3),
        "recovered": recovered,
        "amount_recovered_inr": round(amount_inr, 2) if recovered else 0.0,
    }


def apply_outcomes(events_with_actions: pd.DataFrame, all_events_lookup: pd.DataFrame,
                    action_col: str) -> pd.DataFrame:
    """
    events_with_actions: dataframe with at least [event_id, leak_type/reason_key,
        amount_inr, prior_attempt_count, customer_value_score, action_col]
    all_events_lookup: the original combined event batch, used to pull
        time_since_abandonment_minutes / days_overdue if not already present
    """
    lookup = all_events_lookup.set_index("event_id")
    outcomes = []
    for _, row in events_with_actions.iterrows():
        reason_key = row.get("reason_key")
        if reason_key is None:
            # audit_df from pipeline.py doesn't carry reason_key directly -- pull from lookup
            reason_key = lookup.loc[row["event_id"], "reason_key"]

        time_factor = None
        if reason_key == "checkout_abandoned" and row["event_id"] in lookup.index:
            time_factor = lookup.loc[row["event_id"]].get("time_since_abandonment_minutes")
        elif reason_key == "invoice_overdue" and row["event_id"] in lookup.index:
            time_factor = lookup.loc[row["event_id"]].get("days_overdue")

        outcome = simulate_outcome(
            event_id=row["event_id"], reason_key=reason_key, action=row[action_col],
            amount_inr=row["amount_inr"], prior_attempt_count=row.get("prior_attempt_count", 0),
            customer_value_score=row.get("customer_value_score", 50.0), time_factor=time_factor,
        )
        outcomes.append(outcome)

    outcome_df = pd.DataFrame(outcomes)
    return events_with_actions.merge(outcome_df.drop(columns=["action"]), on="event_id")
