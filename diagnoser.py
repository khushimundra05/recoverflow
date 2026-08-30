"""
diagnoser.py

Takes a normalized event (payment failure, checkout abandonment, or overdue
invoice) and returns a diagnosis: root_cause, confidence, candidate_actions.

REAL vs SIMULATED, explicit at the point of use:
- payment_failure events: reason code comes from Razorpay's real API response
  (test-mode). The mapping table itself (config/reason_mapping.json) is real
  Razorpay vocabulary; the confidence/action values attached to it are
  authored priors (see README).
- checkout_abandonment events: the customer context (order history, value)
  is real Olist data; the abandonment flag/timing is synthetic (see
  simulate_abandonment.py). The diagnosis confidence is deliberately capped
  low (0.35) to reflect that we cannot actually know customer intent.
- invoice_overdue events: fully simulated, confidence capped at 0.40.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict

CONFIG_DIR = Path(__file__).parent.parent / "config"


@dataclass
class Diagnosis:
    event_id: str
    leak_type: str          # "payment_failure" | "checkout_abandonment" | "invoice_overdue"
    reason_key: str          # e.g. "insufficient_funds" or "checkout_abandoned"
    root_cause: str
    source: str              # "customer" | "bank" | "gateway" | "simulated"
    confidence: float
    rationale: str
    candidate_actions: list
    data_provenance: str      # human-readable note on what's real vs simulated for THIS event


def load_reason_mapping() -> dict:
    with open(CONFIG_DIR / "reason_mapping.json") as f:
        return json.load(f)


def diagnose(event: dict, reason_mapping: dict = None) -> Diagnosis:
    """
    event expects at minimum:
        {
            "event_id": str,
            "leak_type": "payment_failure" | "checkout_abandonment" | "invoice_overdue",
            "reason_key": str,   # razorpay reason code, or "checkout_abandoned" /
                                  # "invoice_overdue" for the other two leak types
        }
    """
    if reason_mapping is None:
        reason_mapping = load_reason_mapping()

    reason_key = event["reason_key"]
    if reason_key not in reason_mapping:
        # Unknown reason code -- fail safe, don't guess.
        return Diagnosis(
            event_id=event["event_id"],
            leak_type=event["leak_type"],
            reason_key=reason_key,
            root_cause="unknown",
            source="unknown",
            confidence=0.0,
            rationale=f"No mapping found for reason_key='{reason_key}'. Escalate for manual review.",
            candidate_actions=["escalate_human_review"],
            data_provenance="UNMAPPED -- not a known Razorpay reason or a configured simulated leak type.",
        )

    entry = reason_mapping[reason_key]

    if entry["source"] == "simulated":
        provenance = (
            "SIMULATED leak type -- event context (customer history) may be real "
            "(Olist) but the leak event itself and this diagnosis are simulated."
        )
    else:
        provenance = (
            f"REAL Razorpay reason code, source='{entry['source']}' as reported "
            f"by Razorpay's test-mode API."
        )

    return Diagnosis(
        event_id=event["event_id"],
        leak_type=event["leak_type"],
        reason_key=reason_key,
        root_cause=entry["root_cause"],
        source=entry["source"],
        confidence=entry["confidence"],
        rationale=entry["rationale"],
        candidate_actions=entry["candidate_actions"],
        data_provenance=provenance,
    )


if __name__ == "__main__":
    # Quick manual smoke test with one event per leak type
    sample_events = [
        {"event_id": "evt_001", "leak_type": "payment_failure", "reason_key": "insufficient_funds"},
        {"event_id": "evt_002", "leak_type": "payment_failure", "reason_key": "bank_declined"},
        {"event_id": "evt_003", "leak_type": "checkout_abandonment", "reason_key": "checkout_abandoned"},
        {"event_id": "evt_004", "leak_type": "invoice_overdue", "reason_key": "invoice_overdue"},
        {"event_id": "evt_005", "leak_type": "payment_failure", "reason_key": "totally_made_up_code"},
    ]

    mapping = load_reason_mapping()
    for evt in sample_events:
        diagnosis = diagnose(evt, mapping)
        print(json.dumps(asdict(diagnosis), indent=2))
        print("-" * 60)
