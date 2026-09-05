"""
compare_with_outcomes.py

The final comparison the track bar asks for: measured ₹ recovered, not just
effort allocated. Three strategies evaluated against the SAME independent
outcome simulator (outcome_simulator.py), so the comparison isn't circular:

  1. Naive, unconstrained  -- baseline.py's strategy, all 360 events
  2. Naive, capped to operator's exact budget, arbitrary order -- isolates
     "does prioritization help" from "does spending more help"
  3. Operator (this project) -- scored, deduped, constrained, budget-ranked
"""

import pandas as pd
from pathlib import Path

from pipeline import run_full_pipeline
from baseline import run_naive_baseline, NAIVE_ACTION_MAP
from outcome_simulator import apply_outcomes

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    abandonment = pd.read_csv(DATA_DIR / "simulated_abandonment_events.csv")
    payment_failures = pd.read_csv(DATA_DIR / "simulated_payment_failure_events.csv")
    invoices = pd.read_csv(DATA_DIR / "simulated_invoice_events.csv")
    all_events = pd.concat([abandonment, payment_failures, invoices], ignore_index=True)

    # --- Run the operator ---
    agent_result = run_full_pipeline(force_timeout_on_nth_razorpay_call=15)
    audit_df = agent_result["audit_df"].merge(
        all_events[["event_id", "prior_attempt_count", "customer_value_score"]],
        on="event_id", how="left", suffixes=("", "_dup")
    )
    operator_budget = audit_df["amount_inr"].sum()
    operator_with_outcomes = apply_outcomes(audit_df, all_events, action_col="final_action")

    # --- Run naive, unconstrained (all 360) ---
    baseline_df = run_naive_baseline(all_events, force_timeout_on_nth_razorpay_call=15)
    baseline_df = baseline_df.merge(
        all_events[["event_id", "reason_key", "prior_attempt_count", "customer_value_score"]],
        on="event_id", how="left"
    )
    naive_full_with_outcomes = apply_outcomes(baseline_df, all_events, action_col="action_taken")

    # --- Naive, capped to the SAME budget as the operator, arbitrary (first-come) order ---
    running_total = 0.0
    capped_event_ids = []
    for _, row in all_events.iterrows():
        if running_total + row["amount_inr"] > operator_budget:
            continue
        running_total += row["amount_inr"]
        capped_event_ids.append(row["event_id"])

    naive_capped_df = baseline_df[baseline_df["event_id"].isin(capped_event_ids)].copy()
    naive_capped_with_outcomes = apply_outcomes(naive_capped_df, all_events, action_col="action_taken")

    # --- Summarize ---
    def summarize(df, label):
        return {
            "strategy": label,
            "events_actioned": len(df),
            "budget_spent_inr": df["amount_inr"].sum(),
            "events_recovered": df["recovered"].sum(),
            "recovery_rate_pct": round(100 * df["recovered"].mean(), 1),
            "amount_recovered_inr": df["amount_recovered_inr"].sum(),
            "recovery_per_rupee_spent": round(df["amount_recovered_inr"].sum() / df["amount_inr"].sum(), 3),
        }

    summary = pd.DataFrame([
        summarize(naive_full_with_outcomes, "Naive (unconstrained, all 360)"),
        summarize(naive_capped_with_outcomes, "Naive (same budget as operator, arbitrary order)"),
        summarize(operator_with_outcomes, "Recovery Operator (this project)"),
    ])

    print("=" * 100)
    print("MEASURED ₹ RECOVERED -- independent outcome simulation, same event batch")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\n--- Headline comparison (same ₹ budget spent) ---")
    naive_capped_row = summary.iloc[1]
    operator_row = summary.iloc[2]
    print(f"Naive (arbitrary order):   ₹{naive_capped_row['budget_spent_inr']:,.0f} spent -> ₹{naive_capped_row['amount_recovered_inr']:,.0f} recovered ({naive_capped_row['recovery_rate_pct']}% of events)")
    print(f"Recovery Operator:         ₹{operator_row['budget_spent_inr']:,.0f} spent -> ₹{operator_row['amount_recovered_inr']:,.0f} recovered ({operator_row['recovery_rate_pct']}% of events)")
    lift = operator_row['amount_recovered_inr'] - naive_capped_row['amount_recovered_inr']
    print(f"-> ₹{lift:,.0f} more recovered on the IDENTICAL budget, from smarter targeting alone.")

    out_path = DATA_DIR / "outcome_comparison.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    operator_with_outcomes.to_csv(DATA_DIR / "operator_outcomes_detail.csv", index=False)
    naive_capped_with_outcomes.to_csv(DATA_DIR / "naive_capped_outcomes_detail.csv", index=False)


if __name__ == "__main__":
    main()
