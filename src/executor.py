"""
executor.py

Takes a ConstraintResult (from constraint_engine.py) and actually executes
the final_action -- the one part of this pipeline that touches Razorpay's
real test-mode API.

REAL vs ASSUMED:
- RazorpayClient makes REAL calls to api.razorpay.com (Orders / Payment
  Links APIs) when given real test-mode keys. Requires network access this
  sandbox does not have -- run this locally.
- MockRazorpayClient simulates responses (including a deliberate timeout)
  so the idempotency and failure-handling LOGIC can be verified without
  live credentials. Never used to fabricate "real" demo numbers -- outputs
  from mock runs must be labeled as such in the audit ledger.

IDEMPOTENCY DESIGN (per the reviewed correction):
Razorpay's native idempotency header is documented for Payouts/Refunds, not
generically for Orders/Payment Links. So we do NOT rely on a Razorpay-side
idempotency key here. Instead, the Recovery Operator owns deduplication:
every action gets a deterministic action_id = hash(event_id, customer_id,
action_type, payload). Before executing, we check our own ledger:
  - already COMPLETED -> return cached result, don't call the API again
  - already PENDING_VERIFICATION (ambiguous outcome, e.g. timeout) ->
    don't blindly retry; a human/verifier must check real status first
  - not found -> execute, record result
"""

import os
import json
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum

DATA_DIR = Path(__file__).parent.parent / "data"
LEDGER_PATH = DATA_DIR / "action_ledger.json"


class ActionStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"                        # clean failure, safe to reattempt later
    PENDING_VERIFICATION = "pending_verification"  # ambiguous (timeout) -- do NOT blind-retry


@dataclass
class ExecutionResult:
    action_id: str
    event_id: str
    action_type: str
    status: ActionStatus
    external_reference: str = None   # razorpay order_id / payment_link id, if any
    error_message: str = None
    was_deduplicated: bool = False   # True if we skipped calling the API because already done
    client_used: str = "unknown"     # "real" or "mock" -- always logged, never hidden
    timestamp: str = None


# ---------- Idempotency ledger (application-level, our own) ----------

def _load_ledger() -> dict:
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH) as f:
            return json.load(f)
    return {}


def _save_ledger(ledger: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


def make_action_id(event_id: str, customer_id: str, action_type: str, payload: dict,
                    client_label: str) -> str:
    payload_str = json.dumps(payload, sort_keys=True)
    # client_label is included so a mock execution and a real execution of the
    # "same" logical event NEVER collide in the idempotency ledger -- without
    # this, a real-mode run could silently short-circuit to a cached MOCK
    # result and skip the actual Razorpay call entirely (a real bug caught
    # by testing: an "order_MOCK..." reference appeared during a real-mode
    # test run because the ledger didn't distinguish the two).
    raw = f"{event_id}|{customer_id}|{action_type}|{payload_str}|{client_label}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------- Razorpay clients ----------

class RealRazorpayClient:
    """Real HTTP calls to Razorpay's test-mode API. Requires network access
    to api.razorpay.com -- run this locally, not in this sandbox."""

    BASE_URL = "https://api.razorpay.com/v1"

    def __init__(self, key_id: str = None, key_secret: str = None):
        self.key_id = key_id or os.environ.get("RAZORPAY_KEY_ID")
        self.key_secret = key_secret or os.environ.get("RAZORPAY_KEY_SECRET")
        if not self.key_id or not self.key_secret:
            raise ValueError(
                "Missing RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET. "
                "Set them as environment variables (see .env.example)."
            )

    def create_order(self, amount_paise: int, receipt: str) -> dict:
        import requests
        resp = requests.post(
            f"{self.BASE_URL}/orders",
            auth=(self.key_id, self.key_secret),
            json={"amount": amount_paise, "currency": "INR", "receipt": receipt},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def create_payment_link(self, amount_paise: int, description: str, reference_id: str) -> dict:
        import requests
        resp = requests.post(
            f"{self.BASE_URL}/payment_links",
            auth=(self.key_id, self.key_secret),
            json={
                "amount": amount_paise,
                "currency": "INR",
                "description": description,
                "reference_id": reference_id,   # Razorpay's own de-dup hint for payment links
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()


class MockRazorpayClient:
    """Simulates Razorpay responses for local logic-testing without real
    credentials. `force_timeout_on` lets a demo deliberately trigger the
    graceful-failure path on a specific call count."""

    def __init__(self, force_timeout_on: int = None):
        self.call_count = 0
        self.force_timeout_on = force_timeout_on

    def create_order(self, amount_paise: int, receipt: str) -> dict:
        self.call_count += 1
        if self.force_timeout_on == self.call_count:
            raise TimeoutError("Simulated network timeout calling Razorpay Orders API")
        return {"id": f"order_MOCK{self.call_count:06d}", "amount": amount_paise, "status": "created"}

    def create_payment_link(self, amount_paise: int, description: str, reference_id: str) -> dict:
        self.call_count += 1
        if self.force_timeout_on == self.call_count:
            raise TimeoutError("Simulated network timeout calling Razorpay Payment Links API")
        return {"id": f"plink_MOCK{self.call_count:06d}", "short_url": "https://rzp.io/mock-link", "status": "created"}


# ---------- Executor ----------

ACTIONS_REQUIRING_RAZORPAY_CALL = {"retry_after_delay", "retry_prompt", "payment_link"}
ACTIONS_SIMULATED_ONLY = {"reminder", "alternate_method_prompt", "discount_10_percent", "payment_plan_offer"}
ACTIONS_NO_OP = {"no_action", "escalate_human_review"}


def execute_action(
    constraint_result,              # ConstraintResult from constraint_engine.py
    customer_id: str,
    amount_inr: float,
    client,                          # RealRazorpayClient or MockRazorpayClient
    client_label: str,               # "real" or "mock" -- must be explicit
) -> ExecutionResult:
    action_type = constraint_result.final_action
    payload = {"amount_inr": amount_inr, "action": action_type}
    action_id = make_action_id(constraint_result.event_id, customer_id, action_type, payload, client_label)

    ledger = _load_ledger()

    # --- Idempotency check ---
    if action_id in ledger:
        prior = ledger[action_id]
        if prior["status"] == ActionStatus.COMPLETED.value:
            return ExecutionResult(
                action_id=action_id, event_id=constraint_result.event_id, action_type=action_type,
                status=ActionStatus.COMPLETED, external_reference=prior.get("external_reference"),
                was_deduplicated=True, client_used=prior.get("client_used", "unknown"),
                timestamp=prior.get("timestamp"),
            )
        if prior["status"] == ActionStatus.PENDING_VERIFICATION.value:
            # Do NOT blind-retry an ambiguous prior attempt -- this is the
            # graceful-failure behavior: refuse to act again until verified.
            return ExecutionResult(
                action_id=action_id, event_id=constraint_result.event_id, action_type=action_type,
                status=ActionStatus.PENDING_VERIFICATION,
                error_message="Prior attempt outcome unknown (e.g. timeout). Refusing to auto-retry; needs manual/verifier check.",
                was_deduplicated=True, client_used=prior.get("client_used", "unknown"),
                timestamp=prior.get("timestamp"),
            )
        # else: prior FAILED cleanly -- safe to attempt again below

    timestamp = datetime.now(timezone.utc).isoformat()
    amount_paise = int(round(amount_inr * 100))

    result = ExecutionResult(
        action_id=action_id, event_id=constraint_result.event_id, action_type=action_type,
        status=ActionStatus.PENDING, client_used=client_label, timestamp=timestamp,
    )

    try:
        if action_type in ACTIONS_REQUIRING_RAZORPAY_CALL:
            if action_type == "payment_link":
                resp = client.create_payment_link(
                    amount_paise=amount_paise,
                    description=f"Recovery action for event {constraint_result.event_id}",
                    reference_id=action_id,
                )
                result.external_reference = resp["id"]
            else:  # retry_after_delay / retry_prompt -> create a fresh order to attempt payment against
                resp = client.create_order(amount_paise=amount_paise, receipt=action_id)
                result.external_reference = resp["id"]
            result.status = ActionStatus.COMPLETED

        elif action_type in ACTIONS_SIMULATED_ONLY:
            # No real messaging integration -- log what WOULD be sent, don't fabricate an API call.
            result.external_reference = f"simulated:{action_type}"
            result.status = ActionStatus.COMPLETED

        elif action_type in ACTIONS_NO_OP:
            result.external_reference = None
            result.status = ActionStatus.COMPLETED

        else:
            result.status = ActionStatus.FAILED
            result.error_message = f"Unrecognized action_type '{action_type}' -- not in any known execution category."

    except TimeoutError as e:
        # Ambiguous: we don't know if Razorpay actually processed this or not.
        # Mark PENDING_VERIFICATION, NOT failed -- a blind retry here could
        # create a duplicate order/payment link.
        result.status = ActionStatus.PENDING_VERIFICATION
        result.error_message = str(e)

    except Exception as e:
        # Clean/known failure (e.g. 4xx from Razorpay) -- safe to reattempt later.
        result.status = ActionStatus.FAILED
        result.error_message = str(e)

    # Persist to ledger regardless of outcome -- audit trail must show attempts, not just successes.
    ledger[action_id] = asdict(result)
    ledger[action_id]["status"] = result.status.value
    _save_ledger(ledger)

    return result


if __name__ == "__main__":
    from diagnoser import diagnose, load_reason_mapping
    from scorer import score_event, load_scoring_config
    from constraint_engine import check_constraints, load_policy

    mapping = load_reason_mapping()
    scoring_config = load_scoring_config()
    policy = load_policy()

    # Clear ledger for a clean demo run
    if LEDGER_PATH.exists():
        LEDGER_PATH.unlink()

    print("=" * 70)
    print("DEMO 1: Normal execution against MOCK client")
    print("=" * 70)
    event = {"event_id": "evt_exec_001", "leak_type": "payment_failure", "reason_key": "insufficient_funds"}
    diagnosis = diagnose(event, mapping)
    scored = score_event(diagnosis, revenue_value_inr=5000, revenue_value_is_real=True,
                          customer_value_score=65.0, prior_attempt_count=0, scoring_config=scoring_config)
    constrained = check_constraints(scored, revenue_value_inr=5000,
                                     customer_contact_history={"messages_sent_this_week": 0, "hours_since_last_contact": None, "attempts_this_event": 0},
                                     policy=policy)
    mock_client = MockRazorpayClient()
    r1 = execute_action(constrained, customer_id="cust_A", amount_inr=5000, client=mock_client, client_label="mock")
    print(r1)

    print()
    print("=" * 70)
    print("DEMO 2: Same event executed AGAIN -- idempotency should kick in, no duplicate action")
    print("=" * 70)
    r2 = execute_action(constrained, customer_id="cust_A", amount_inr=5000, client=mock_client, client_label="mock")
    print(r2)
    print(f"Was deduplicated? {r2.was_deduplicated}  |  Mock client call_count: {mock_client.call_count} (should still be 1)")

    print()
    print("=" * 70)
    print("DEMO 3: A DIFFERENT event hits a simulated network timeout -- graceful failure")
    print("=" * 70)
    event2 = {"event_id": "evt_exec_002", "leak_type": "payment_failure", "reason_key": "gateway_error"}
    diagnosis2 = diagnose(event2, mapping)
    scored2 = score_event(diagnosis2, revenue_value_inr=3000, revenue_value_is_real=True,
                           customer_value_score=50.0, prior_attempt_count=0, scoring_config=scoring_config)
    constrained2 = check_constraints(scored2, revenue_value_inr=3000,
                                      customer_contact_history={"messages_sent_this_week": 0, "hours_since_last_contact": None, "attempts_this_event": 0},
                                      policy=policy)
    timeout_client = MockRazorpayClient(force_timeout_on=1)  # times out on the very first call
    r3 = execute_action(constrained2, customer_id="cust_B", amount_inr=3000, client=timeout_client, client_label="mock")
    print(r3)

    print()
    print("=" * 70)
    print("DEMO 4: Retry the SAME timed-out event -- must refuse to blind-retry")
    print("=" * 70)
    r4 = execute_action(constrained2, customer_id="cust_B", amount_inr=3000, client=timeout_client, client_label="mock")
    print(r4)
    print(f"Correctly refused to auto-retry an ambiguous prior outcome? {r4.status == ActionStatus.PENDING_VERIFICATION}")