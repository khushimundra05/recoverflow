"""
scorer.py

Combines a Diagnosis (from diagnoser.py) with event value/customer features
into:
  1. Recovery Opportunity Score  -- should we even spend effort on this event?
  2. Action Score per candidate action -- among allowed actions, which is best?

Per config/scoring_config.json, these are transparent configurable heuristics,
NOT calibrated probabilities or a trained model. See README "Data Honesty".

REAL vs ASSUMED in this file:
  - revenue_value: real for payment_failure (Razorpay order amount) and
    checkout_abandonment (synthetic cart_value_inr -- flagged); simulated for
    invoice_overdue.
  - customer_value_score: REAL, computed from actual Olist order history
    (see ingest_olist.py), used as a rank-based signal only.
  - recovery_potential, confidence: come from diagnoser.py -- real Razorpay
    source/reason for payment failures, capped/authored for simulated leaks.
  - action_cost / risk: authored reference values from scoring_config.json,
    not measured.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field

CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_scoring_config() -> dict:
    with open(CONFIG_DIR / "scoring_config.json") as f:
        return json.load(f)


@dataclass
class ScoredEvent:
    event_id: str
    leak_type: str
    recovery_opportunity_score: float
    score_components: dict
    action_scores: dict           # {action_name: net_score}
    best_action: str
    best_action_score: float
    data_provenance: dict         # per-field real/assumed breakdown, for the audit ledger


def _confidence_to_100(confidence: float) -> float:
    """confidence is 0-1 in reason_mapping.json; scores elsewhere are 0-100."""
    return round(confidence * 100, 1)


def score_event(
    diagnosis,                      # Diagnosis from diagnoser.py
    revenue_value_inr: float,
    revenue_value_is_real: bool,
    customer_value_score: float,    # 0-100, from ingest_olist.py (real, rank-based)
    prior_attempt_count: int = 0,
    scoring_config: dict = None,
) -> ScoredEvent:
    if scoring_config is None:
        scoring_config = load_scoring_config()

    weights = scoring_config["recovery_opportunity_score"]["weights"]

    # --- Recovery potential: derived from diagnosis confidence and prior attempts.
    # More prior failed attempts on the SAME event lowers potential (diminishing
    # returns on retrying something that keeps failing).
    attempt_penalty = min(prior_attempt_count * 15, 60)
    recovery_potential = max(_confidence_to_100(diagnosis.confidence) - attempt_penalty, 0)

    # --- Revenue value: normalize to 0-100 via a simple capped log-ish scale.
    # (For the hackathon batch, better to normalize against the batch's own
    # distribution -- this simple cap is a placeholder until ranker.py has the
    # full batch loaded. Flagged here so it's not mistaken for a calibrated scale.)
    revenue_value_score = min(revenue_value_inr / 1000, 100)  # ₹1000 -> 1pt, caps at ₹1L+

    confidence_score = _confidence_to_100(diagnosis.confidence)

    recovery_opportunity_score = round(
        weights["recovery_potential"] * recovery_potential
        + weights["revenue_value"] * revenue_value_score
        + weights["customer_value"] * customer_value_score
        + weights["confidence"] * confidence_score,
        2,
    )

    # --- Action scoring: among diagnosis.candidate_actions, score each.
    action_cost_ref = scoring_config["action_score"]["action_cost_reference"]
    risk_ref = scoring_config["action_score"]["risk_reference"]
    effectiveness_ref = scoring_config["action_score"]["action_effectiveness_reference"]

    action_scores = {}
    for action in diagnosis.candidate_actions:
        cost = action_cost_ref.get(action, 20)  # unknown action -> assume moderate cost
        risk = risk_ref.get(action, 20)
        effectiveness = effectiveness_ref.get(action, 0.5)  # unknown action -> assume middling effectiveness
        recovery_benefit_estimate = recovery_potential * effectiveness  # proxy, not a rupee prediction
        net_score = round(recovery_benefit_estimate - cost - risk, 2)
        action_scores[action] = net_score

    best_action = max(action_scores, key=action_scores.get)
    best_action_score = action_scores[best_action]

    provenance = {
        "recovery_potential": f"diagnosis.confidence ({'real Razorpay signal' if diagnosis.source != 'simulated' else 'simulated/authored'}) minus prior_attempt_count penalty (authored)",
        "revenue_value": "REAL amount" if revenue_value_is_real else "SIMULATED amount (see event source)",
        "customer_value_score": "REAL -- computed from actual Olist order history",
        "confidence": f"{'REAL Razorpay-derived' if diagnosis.source != 'simulated' else 'authored/capped'} diagnostic confidence",
        "action_cost/risk": "authored reference values (config/scoring_config.json), not measured",
    }

    return ScoredEvent(
        event_id=diagnosis.event_id,
        leak_type=diagnosis.leak_type,
        recovery_opportunity_score=recovery_opportunity_score,
        score_components={
            "recovery_potential": recovery_potential,
            "revenue_value_score": revenue_value_score,
            "customer_value_score": customer_value_score,
            "confidence_score": confidence_score,
        },
        action_scores=action_scores,
        best_action=best_action,
        best_action_score=best_action_score,
        data_provenance=provenance,
    )


if __name__ == "__main__":
    from diagnoser import diagnose, load_reason_mapping

    mapping = load_reason_mapping()
    config = load_scoring_config()

    sample_event = {"event_id": "evt_001", "leak_type": "payment_failure", "reason_key": "insufficient_funds"}
    diagnosis = diagnose(sample_event, mapping)

    scored = score_event(
        diagnosis=diagnosis,
        revenue_value_inr=10000,
        revenue_value_is_real=True,
        customer_value_score=71.2,   # e.g. from a real Olist customer
        prior_attempt_count=1,
        scoring_config=config,
    )

    import pprint
    pprint.pprint(scored)
