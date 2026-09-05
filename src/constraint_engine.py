"""
constraint_engine.py

Takes a ScoredEvent (from scorer.py) plus the customer's recent action
history, and filters its action_scores against config/merchant_policy.json.

This is deliberately a SEPARATE step from scoring, per the reviewed design:
  1. What's allowed?  <- this file
  2. Among allowed actions, which is best?  <- scorer.py already ranked all
     candidates; this file just removes the disallowed ones and re-picks
     the best of what's left.

Nothing in here is "real data" -- merchant_policy.json values are authored
defaults (see README). What IS real is the logic itself: this is the actual
mechanism that would run in production against a merchant's real configured
policy and a real customer contact-history log.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta

CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_policy() -> dict:
    with open(CONFIG_DIR / "merchant_policy.json") as f:
        return json.load(f)


@dataclass
class ConstraintResult:
    event_id: str
    original_best_action: str
    original_best_score: float
    allowed_actions: dict          # action -> score, filtered
    blocked_actions: dict          # action -> reason blocked
    final_action: str
    final_action_score: float
    was_overridden: bool           # True if final_action != original_best_action
    requires_human_review: bool


def _action_discount_percent(action: str) -> float:
    """Extract discount % from action names like 'discount_10_percent'."""
    if "discount" in action and "percent" in action:
        try:
            return float(action.split("_")[1])
        except (IndexError, ValueError):
            return 0.0
    return 0.0


def check_constraints(
    scored_event,                     # ScoredEvent from scorer.py
    revenue_value_inr: float,
    customer_contact_history: dict,   # e.g. {"messages_sent_this_week": 2, "hours_since_last_contact": 5, "attempts_this_event": 1}
    policy: dict = None,
) -> ConstraintResult:
    if policy is None:
        policy = load_policy()

    blocked = {}
    allowed = dict(scored_event.action_scores)  # start with everything scorer proposed

    contact_limits = policy["contact_limits"]
    financial_limits = policy["financial_limits"]

    for action in list(allowed.keys()):
        # -- Contact frequency cap
        is_contact_action = action in ("reminder", "payment_link", "alternate_method_prompt", "retry_prompt")
        if is_contact_action and customer_contact_history.get("messages_sent_this_week", 0) >= contact_limits["max_messages_per_customer_per_week"]:
            blocked[action] = (
                f"Blocked: customer already contacted "
                f"{customer_contact_history['messages_sent_this_week']} times this week "
                f"(policy max: {contact_limits['max_messages_per_customer_per_week']})"
            )
            del allowed[action]
            continue

        # -- Cooldown window
        hours_since_last = customer_contact_history.get("hours_since_last_contact")
        if is_contact_action and hours_since_last is not None and hours_since_last < contact_limits["customer_cooldown_hours"]:
            blocked[action] = (
                f"Blocked: within cooldown window "
                f"({hours_since_last}h since last contact, policy requires "
                f"{contact_limits['customer_cooldown_hours']}h)"
            )
            del allowed[action]
            continue

        # -- Max attempts on this specific event
        attempts = customer_contact_history.get("attempts_this_event", 0)
        if attempts >= contact_limits["max_recovery_attempts_per_event"]:
            blocked[action] = (
                f"Blocked: max recovery attempts reached for this event "
                f"({attempts}/{contact_limits['max_recovery_attempts_per_event']})"
            )
            del allowed[action]
            continue

        # -- Discount cap
        discount_pct = _action_discount_percent(action)
        if discount_pct > financial_limits["max_auto_discount_percent"]:
            blocked[action] = (
                f"Blocked: {discount_pct}% discount exceeds merchant cap "
                f"of {financial_limits['max_auto_discount_percent']}%"
            )
            del allowed[action]
            continue

    # -- Escalation threshold: high-value events can't be auto-executed at all,
    # regardless of which action scored best -- force human review.
    requires_human_review = revenue_value_inr > financial_limits["human_escalation_threshold_amount_inr"]
    if requires_human_review and "escalate_human_review" not in allowed:
        allowed["escalate_human_review"] = 0.0  # always available as a fallback

    if not allowed:
        # Every candidate action got blocked -- fail safe, don't do nothing silently.
        final_action = "escalate_human_review"
        final_score = 0.0
        allowed[final_action] = final_score
    elif requires_human_review:
        final_action = "escalate_human_review"
        final_score = allowed["escalate_human_review"]
    else:
        final_action = max(allowed, key=allowed.get)
        final_score = allowed[final_action]

    return ConstraintResult(
        event_id=scored_event.event_id,
        original_best_action=scored_event.best_action,
        original_best_score=scored_event.best_action_score,
        allowed_actions=allowed,
        blocked_actions=blocked,
        final_action=final_action,
        final_action_score=final_score,
        was_overridden=(final_action != scored_event.best_action),
        requires_human_review=requires_human_review,
    )


if __name__ == "__main__":
    from diagnoser import diagnose, load_reason_mapping
    from scorer import score_event, load_scoring_config

    mapping = load_reason_mapping()
    scoring_config = load_scoring_config()
    policy = load_policy()

    print("=" * 70)
    print("SCENARIO 1: Normal case -- no constraints hit")
    print("=" * 70)
    event = {"event_id": "evt_A", "leak_type": "payment_failure", "reason_key": "insufficient_funds"}
    diagnosis = diagnose(event, mapping)
    scored = score_event(diagnosis, revenue_value_inr=10000, revenue_value_is_real=True,
                          customer_value_score=71.2, prior_attempt_count=0, scoring_config=scoring_config)
    result = check_constraints(scored, revenue_value_inr=10000,
                                customer_contact_history={"messages_sent_this_week": 0, "hours_since_last_contact": None, "attempts_this_event": 0},
                                policy=policy)
    print(f"Scorer picked: {scored.best_action} (score {scored.best_action_score})")
    print(f"Final action:  {result.final_action} (score {result.final_action_score})")
    print(f"Overridden by policy? {result.was_overridden}")

    print()
    print("=" * 70)
    print("SCENARIO 2: Customer already contacted twice this week -- BLOCKED, forces override")
    print("=" * 70)
    cancel_event = {"event_id": "evt_C", "leak_type": "payment_failure", "reason_key": "payment_cancelled"}
    cancel_diagnosis = diagnose(cancel_event, mapping)
    cancel_scored = score_event(cancel_diagnosis, revenue_value_inr=8000, revenue_value_is_real=True,
                                 customer_value_score=60.0, prior_attempt_count=0, scoring_config=scoring_config)
    result2 = check_constraints(cancel_scored, revenue_value_inr=8000,
                                 customer_contact_history={"messages_sent_this_week": 2, "hours_since_last_contact": 30, "attempts_this_event": 0},
                                 policy=policy)
    print(f"Scorer picked: {cancel_scored.best_action} (score {cancel_scored.best_action_score})")
    print(f"All candidate actions scored: {cancel_scored.action_scores}")
    print(f"Blocked actions: {result2.blocked_actions}")
    print(f"Final action (fallback): {result2.final_action} (score {result2.final_action_score})")
    print(f"Overridden by policy? {result2.was_overridden}")

    print()
    print("=" * 70)
    print("SCENARIO 3: High-value event -- forced human escalation")
    print("=" * 70)
    big_event = {"event_id": "evt_B", "leak_type": "payment_failure", "reason_key": "gateway_error"}
    big_diagnosis = diagnose(big_event, mapping)
    big_scored = score_event(big_diagnosis, revenue_value_inr=45000, revenue_value_is_real=True,
                              customer_value_score=80.0, prior_attempt_count=0, scoring_config=scoring_config)
    result3 = check_constraints(big_scored, revenue_value_inr=45000,
                                 customer_contact_history={"messages_sent_this_week": 0, "hours_since_last_contact": None, "attempts_this_event": 0},
                                 policy=policy)
    print(f"Scorer picked: {big_scored.best_action} (score {big_scored.best_action_score})")
    print(f"Requires human review (>₹{policy['financial_limits']['human_escalation_threshold_amount_inr']})? {result3.requires_human_review}")
    print(f"Final action: {result3.final_action}")
