"""
generate_dashboard.py

Reads the actual output files from a completed pipeline run and produces a
single self-contained HTML file (data embedded, no server needed) summarizing
the whole project for a demo: KPIs, naive-vs-operator comparison, ML
classifier metrics, and the full audit trail -- with real vs simulated data
labeled throughout, never blended together silently.

Run this AFTER pipeline.py and compare_with_outcomes.py and ml_risk_model.py.
"""

import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_PATH = Path(__file__).parent.parent / "dashboard.html"


def load_data():
    audit = pd.read_csv(DATA_DIR / "audit_ledger.csv")
    outcome_comparison = pd.read_csv(DATA_DIR / "outcome_comparison.csv")

    # Optional: created by run_live_demo.py. This is kept separate from the
    # evaluation ledger because real Razorpay Test Mode execution is evidence
    # of API execution, not ground-truth revenue recovery.
    live_path = DATA_DIR / "audit_ledger_live_demo.csv"
    live_demo = pd.read_csv(live_path) if live_path.exists() else pd.DataFrame()

    with open(DATA_DIR / "ml_classifier_metrics.json") as f:
        ml_metrics = json.load(f)

    return audit, outcome_comparison, ml_metrics, live_demo


def build_html(
    audit: pd.DataFrame,
    outcome_comparison: pd.DataFrame,
    ml_metrics: dict,
    live_demo: pd.DataFrame,
) -> str:
    naive_row = outcome_comparison[outcome_comparison["strategy"] == "Naive (same budget as operator, arbitrary order)"].iloc[0]
    operator_row = outcome_comparison[outcome_comparison["strategy"] == "Recovery Operator (this project)"].iloc[0]
    lift_inr = operator_row["amount_recovered_inr"] - naive_row["amount_recovered_inr"]
    lift_multiple = operator_row["amount_recovered_inr"] / naive_row["amount_recovered_inr"]

    kpis = {
        "events_seen": int(audit.shape[0]) + 0,  # actioned count; total seen tracked separately below
        "actioned": int(len(audit)),
        "budget_used": float(audit["amount_inr"].sum()),
        "overrides": int(audit["was_overridden_by_policy"].sum()),
        "escalations": int(audit["requires_human_review"].sum()),
        "pending_verification": int((audit["execution_status"] == "pending_verification").sum()),
    }

    model_a = ml_metrics["model_a_class_weight_balanced"]
    model_b = ml_metrics["model_b_smote"]
    selected = ml_metrics["selected_model"]

    audit_display = audit[[
        "event_id", "leak_type", "amount_inr", "root_cause", "confidence",
        "recovery_opportunity_score", "final_action", "was_overridden_by_policy",
        "requires_human_review", "execution_status",
    ]].copy()
    audit_display["amount_inr"] = audit_display["amount_inr"].round(0)
    audit_display["confidence"] = audit_display["confidence"].round(2)
    audit_display["recovery_opportunity_score"] = audit_display["recovery_opportunity_score"].round(1)

    data_json = json.dumps({
        "kpis": kpis,
        "comparison": {
            "naive": {
                "budget": float(naive_row["budget_spent_inr"]),
                "recovered": float(naive_row["amount_recovered_inr"]),
                "rate": float(naive_row["recovery_rate_pct"]),
            },
            "operator": {
                "budget": float(operator_row["budget_spent_inr"]),
                "recovered": float(operator_row["amount_recovered_inr"]),
                "rate": float(operator_row["recovery_rate_pct"]),
            },
            "lift_inr": float(lift_inr),
            "lift_multiple": float(lift_multiple),
        },
        "ml": {"model_a": model_a, "model_b": model_b, "selected": selected},
        "audit": audit_display.to_dict(orient="records"),
        "live_demo": live_demo.to_dict(orient="records"),
    })

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Recovery Operator — Results Ledger</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #EEE8D9;
    --panel: #FBF8EF;
    --ink: #1E2A33;
    --ink-muted: #5B6B72;
    --rule: #C9BFA0;
    --brass: #8A6A24;
    --real: #1F6F5C;
    --simulated: #9C5430;
    --blocked: #8C3A2B;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    line-height: 1.5;
  }
  .page { max-width: 980px; margin: 0 auto; padding: 56px 32px 96px; }
  .masthead {
    border-bottom: 2px solid var(--ink); padding-bottom: 20px; margin-bottom: 40px;
  }
  .masthead h1 {
    font-family: Georgia, "Iowan Old Style", serif;
    font-size: 30px; font-weight: 500; margin: 0 0 6px; letter-spacing: -0.01em;
  }
  .masthead p { margin: 0; color: var(--ink-muted); font-size: 14px; max-width: 640px; }

  .hero {
    display: flex; align-items: baseline; gap: 28px; margin: 40px 0 48px;
    flex-wrap: wrap;
  }
  .hero .big {
    font-family: Georgia, serif; font-size: 64px; color: var(--brass);
    font-variant-numeric: tabular-nums; line-height: 1;
  }
  .hero .label { font-size: 15px; color: var(--ink-muted); max-width: 320px; }

  .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--rule); border: 1px solid var(--rule); margin-bottom: 48px; }
  .kpi { background: var(--panel); padding: 18px 16px; }
  .kpi .n { font-size: 22px; font-variant-numeric: tabular-nums; font-weight: 600; }
  .kpi .l { font-size: 12px; color: var(--ink-muted); margin-top: 2px; }

  section { margin-bottom: 48px; }
  h2 {
    font-family: Georgia, serif; font-size: 19px; font-weight: 500;
    border-bottom: 1px solid var(--rule); padding-bottom: 8px; margin: 0 0 18px;
  }
  .note { font-size: 13px; color: var(--ink-muted); margin-top: 8px; }

  .chart-wrap { background: var(--panel); border: 1px solid var(--rule); padding: 20px; max-width: 520px; }
  .rate-row { display: flex; gap: 28px; margin: 14px 0 4px; max-width: 520px; flex-wrap: wrap; }
  .rate-item { font-size: 14px; }
  .rate-label { color: var(--ink-muted); }
  .rate-value { font-weight: 600; font-variant-numeric: tabular-nums; color: var(--brass); }

  table { width: 100%; border-collapse: collapse; font-size: 13px; background: var(--panel); border: 1px solid var(--rule); }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--rule); }
  th { font-weight: 600; color: var(--ink-muted); font-size: 12px; }
  tr:last-child td { border-bottom: none; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }

  .tag { display: inline-block; padding: 1px 8px; border-radius: 2px; font-size: 11px; font-weight: 600; }
  .tag-real { background: rgba(31,111,92,0.12); color: var(--real); }
  .tag-sim { background: rgba(156,84,48,0.12); color: var(--simulated); }
  .tag-blocked { background: rgba(140,58,43,0.12); color: var(--blocked); }

  #auditSearch {
    width: 100%; padding: 8px 10px; border: 1px solid var(--rule); background: var(--panel);
    font-size: 13px; margin-bottom: 10px; font-family: inherit; color: var(--ink);
  }
  .audit-scroll { max-height: 420px; overflow-y: auto; border: 1px solid var(--rule); }
  .audit-scroll table { border: none; }

  footer { border-top: 1px solid var(--rule); padding-top: 20px; font-size: 12px; color: var(--ink-muted); }
</style>
</head>
<body>
<div class="page">

  <div class="masthead">
    <h1>Revenue Recovery Operator — Results Ledger</h1>
    <p>Detects payment failures, checkout abandonment, and overdue invoices; scores, prioritizes under
    a daily budget, checks merchant policy, executes safely, and measures recovery against an independent
    outcome simulation. Real Razorpay error taxonomy and real Olist customer data throughout; simulated
    data is labeled, never disguised as real.</p>
  </div>

  <div class="hero">
    <div>
      <div class="big" id="heroNumber">—</div>
      <div class="label">more recovered than a naive strategy, on the identical ₹ budget spent</div>
    </div>
  </div>

  <div class="kpi-row" id="kpiRow"></div>

  <section>
    <h2>Naive strategy vs. Recovery Operator — same budget, independent outcome simulation</h2>
    <div class="chart-wrap"><canvas id="comparisonChart" height="220"></canvas></div>
    <div class="rate-row">
      <div class="rate-item"><span class="rate-label">Recovery rate — Naive:</span> <span id="naiveRate" class="rate-value"></span></div>
      <div class="rate-item"><span class="rate-label">Recovery rate — Operator:</span> <span id="operatorRate" class="rate-value"></span></div>
    </div>
    <p class="note">Recovered amounts come from an outcome-simulation model built independently of the
    scoring engine (no live merchant exists to provide real ground truth). Its output was checked — not
    tuned — against real Razorpay-published recovery-rate benchmarks; see README for sources.</p>
  </section>

  <section>
    <h2>ML customer propensity model — real Olist labels, held-out test set</h2>
    <table id="mlTable"></table>
    <p class="note">Predicts real historical order completion (canceled/unavailable vs.
    delivered/shipped) — a genuine label Olist provides. This does <em>not</em> predict whether a
    simulated recovery action succeeds; no dataset has that label. Its output is blended into the
    customer-value component of the scorer, not treated as a recovery-outcome prediction.
    ~0.47% positive class — PR-AUC (not accuracy) is the honest metric here. Two imbalance-handling
    strategies were compared on the identical untouched test set; the better one (by PR-AUC) was
    selected automatically.</p>
  </section>

  <section>
    <h2>Live Razorpay Test Mode — execution proof</h2>
    <div class="live-summary">
      <span class="tag tag-real">REAL</span>
      <strong id="liveCount">0</strong> operator-selected action(s) executed against Razorpay Test Mode
    </div>
    <div class="live-scroll"><table id="liveTable"></table></div>
    <p class="note">
      These are real API execution results from <code>run_live_demo.py</code>.
      The Razorpay references below prove that the executor reached Razorpay Test Mode.
      They are deliberately kept separate from the simulated recovery amounts above:
      a Test Mode order being created is not evidence that a customer actually paid.
    </p>
  </section>

  <section>
    <h2>Audit trail — every action taken today</h2>
    <input id="auditSearch" type="text" placeholder="Search by event id, leak type, or action…">
    <div class="audit-scroll"><table id="auditTable"></table></div>
  </section>

  <footer>
    Data provenance: <span class="tag tag-real">Real</span> Razorpay error taxonomy, Olist customer/order
    history &nbsp;·&nbsp; <span class="tag tag-sim">Simulated</span> event occurrences, amounts, outcome
    evaluation (independently built, checked against real published benchmarks) &nbsp;·&nbsp;
    <span class="tag tag-blocked">Blocked</span> = action stopped by merchant policy, fallback applied.
  </footer>

</div>

<script>
const DATA = __DATA_JSON__;

document.getElementById('heroNumber').textContent =
  '₹' + Math.round(DATA.comparison.lift_inr).toLocaleString('en-IN') +
  ' (' + DATA.comparison.lift_multiple.toFixed(1) + '×)';

const kpiRow = document.getElementById('kpiRow');
const kpis = [
  [DATA.kpis.actioned, 'Events actioned today'],
  ['₹' + Math.round(DATA.kpis.budget_used).toLocaleString('en-IN'), 'Budget spent'],
  [DATA.kpis.overrides, 'Policy overrides (blocked → fallback)'],
  [DATA.kpis.escalations, 'Forced human escalations'],
];
kpiRow.innerHTML = kpis.map(([n,l]) => `<div class="kpi"><div class="n">${n}</div><div class="l">${l}</div></div>`).join('');

new Chart(document.getElementById('comparisonChart'), {
  type: 'bar',
  data: {
    labels: ['₹ recovered'],
    datasets: [
      { label: 'Naive (arbitrary order)', backgroundColor: '#9C5430',
        data: [DATA.comparison.naive.recovered] },
      { label: 'Recovery Operator', backgroundColor: '#1F6F5C',
        data: [DATA.comparison.operator.recovered] },
    ]
  },
  options: {
    responsive: true,
    plugins: { tooltip: { callbacks: { label: (ctx) => ctx.dataset.label + ': ₹' + Math.round(ctx.parsed.y).toLocaleString('en-IN') } } },
    scales: { y: { beginAtZero: true, ticks: { callback: (v) => '₹' + (v/1000).toFixed(0) + 'K' } } }
  }
});
document.getElementById('naiveRate').textContent = DATA.comparison.naive.rate + '%';
document.getElementById('operatorRate').textContent = DATA.comparison.operator.rate + '%';

const mlTable = document.getElementById('mlTable');
const mlRows = [
  ['Model', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC', 'Selected'],
  [DATA.ml.model_a.label, DATA.ml.model_a.precision, DATA.ml.model_a.recall, DATA.ml.model_a.roc_auc, DATA.ml.model_a.pr_auc, DATA.ml.model_a.label === DATA.ml.selected ? '✓' : ''],
  [DATA.ml.model_b.label, DATA.ml.model_b.precision, DATA.ml.model_b.recall, DATA.ml.model_b.roc_auc, DATA.ml.model_b.pr_auc, DATA.ml.model_b.label === DATA.ml.selected ? '✓' : ''],
];
mlTable.innerHTML = '<tr>' + mlRows[0].map(h => `<th>${h}</th>`).join('') + '</tr>' +
  mlRows.slice(1).map(r => '<tr>' + r.map((c,i) => `<td class="${i>0 && i<5 ? 'num' : ''}">${c}</td>`).join('') + '</tr>').join('');

const liveRows = DATA.live_demo || [];
document.getElementById('liveCount').textContent = liveRows.length;

const liveTable = document.getElementById('liveTable');
const liveCols = ['event_id', 'final_action', 'amount_inr', 'execution_status', 'external_reference', 'execution_error'];
const liveLabels = ['Event', 'Action', '₹', 'Status', 'Razorpay reference', 'Error'];

if (liveRows.length === 0) {
  liveTable.innerHTML =
    '<tr><td>No live-demo ledger found. Run <code>python src/run_live_demo.py 5</code> first, then regenerate the dashboard.</td></tr>';
} else {
  liveTable.innerHTML =
    '<tr>' + liveLabels.map(l => `<th>${l}</th>`).join('') + '</tr>' +
    liveRows.map(r => '<tr>' + liveCols.map(c => {
      let v = r[c] ?? '';
      if (c === 'amount_inr' && v !== '') v = Math.round(v).toLocaleString('en-IN');
      if (c === 'execution_status' && v === 'completed') {
        v = `<span class="status-ok">✓ ${v}</span>`;
      }
      if (c === 'external_reference' && v !== '') {
        v = `<span class="live-ref">${v}</span>`;
      }
      return `<td>${v}</td>`;
    }).join('') + '</tr>').join('');
}

function renderAudit(rows) {
  const table = document.getElementById('auditTable');
  const cols = ['event_id','leak_type','amount_inr','root_cause','final_action','was_overridden_by_policy','requires_human_review','execution_status'];
  const labels = ['Event','Leak type','₹','Root cause','Action taken','Overridden','Escalated','Status'];
  table.innerHTML = '<tr>' + labels.map(l => `<th>${l}</th>`).join('') + '</tr>' +
    rows.map(r => '<tr>' + cols.map(c => {
      let v = r[c];
      if (c === 'amount_inr') v = Math.round(v).toLocaleString('en-IN');
      if (c === 'was_overridden_by_policy' || c === 'requires_human_review') v = v ? 'Yes' : '';
      return `<td class="${c==='amount_inr' ? 'num' : ''}">${v}</td>`;
    }).join('') + '</tr>').join('');
}
renderAudit(DATA.audit);

document.getElementById('auditSearch').addEventListener('input', (e) => {
  const q = e.target.value.toLowerCase();
  renderAudit(DATA.audit.filter(r =>
    String(r.event_id).toLowerCase().includes(q) ||
    String(r.leak_type).toLowerCase().includes(q) ||
    String(r.final_action).toLowerCase().includes(q)
  ));
});
</script>
</body>
</html>
"""
    return html.replace("__DATA_JSON__", data_json)


if __name__ == "__main__":
    audit, outcome_comparison, ml_metrics, live_demo = load_data()
    html = build_html(audit, outcome_comparison, ml_metrics, live_demo)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {OUT_PATH}")