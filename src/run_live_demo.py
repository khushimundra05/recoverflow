"""
run_live_demo.py

Demonstrates the SAME operator pipeline (score -> dedup -> rank -> constrain
-> execute) with the executor switched to Razorpay's REAL test-mode API,
capped at a small number of actions.

This is deliberately NOT a second executor implementation -- it calls the
exact same run_full_pipeline() used everywhere else, just with
razorpay_mode="real" and max_live_actions set. The decision logic, scoring,
and constraint checking are identical to the mock evaluation run; only the
final HTTP calls differ.

Real Razorpay results are written to data/audit_ledger_live_demo.csv,
SEPARATE from the main audit_ledger.csv used by the mock evaluation --
they must never be blended, since the "₹ recovered" headline number comes
exclusively from the independent outcome simulation (outcome_simulator.py),
not from this live run. This script only proves the executor genuinely
speaks Razorpay's API; it does not change or contribute to the measured
recovery numbers.

Requires real credentials in .env (see .env.example) and REAL network
access to api.razorpay.com -- this sandbox cannot reach that domain, so
this script has been written carefully but NOT executed here. Run it
yourself once your Razorpay test-mode keys are set up.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from pipeline import run_full_pipeline


def main(n_events: int = 5):
    key_id = os.environ.get("RAZORPAY_KEY_ID")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET")

    if not key_id or not key_secret or key_id.startswith("rzp_test_your_key"):
        print("ERROR: RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set (or still the")
        print(".env.example placeholder). Set up your test-mode keys first --")
        print("see the README 'Razorpay live demo' section for the walkthrough.")
        sys.exit(1)

    print(f"Running the operator pipeline with razorpay_mode='real', capped at "
          f"{n_events} action(s)...\n")

    result = run_full_pipeline(
        force_timeout_on_nth_razorpay_call=None,  # not meaningful in real mode
        razorpay_mode="real",
        max_live_actions=n_events,
    )
    audit_df = result["audit_df"]

    if len(audit_df) == 0:
        print("No events processed -- check that simulated_*.csv data files exist "
              "(run generate_all_data.py first).")
        return

    print(f"Processed {len(audit_df)} action(s) against Razorpay's REAL test-mode API:\n")
    for _, row in audit_df.iterrows():
        print(f"  [{row['event_id']}] action={row['final_action']} "
              f"status={row['execution_status']} ref={row['external_reference']}")

    print(f"\nDone. Check your Razorpay Dashboard (Test Mode -> Orders / Payment Links)")
    print(f"to see these listed there. Full detail saved to: "
          f"data/audit_ledger_live_demo.csv")
    print(f"\nReminder: these real API calls demonstrate EXECUTION, not the ₹ recovered")
    print(f"headline number -- that comes from outcome_simulator.py's independent")
    print(f"evaluation, run separately via compare_with_outcomes.py (mock mode).")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(n_events=n)