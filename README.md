# Recovery Operator

**A bounded revenue recovery agent for Razorpay merchants — built for the AI Revenue Recovery track.**

### 🚀 Live Demo

**[Open RecoverFlow Dashboard](https://recoverflow-five.vercel.app/)**

Most recovery systems retry every failed payment the same way and message every abandoned cart identically. This project asks a different question:

> **When recovery capacity is limited, which revenue leaks are actually worth acting on, what is the safest permitted intervention, and when should the system stop?**

On an identical **₹500,000 daily recovery budget**, across **2,400 simulated revenue-leak events mapped onto 2,013 customers from the real Olist dataset**, the independent outcome simulation estimates **₹165,850 vs. ₹49,669 for an arbitrary-order naive strategy — a 3.34× difference in simulated recovery**.

The recovery figures are **not real money recovered from Razorpay**. They are outputs of a separate offline outcome evaluation.

---

## What it does

```text
Failed payment / Abandoned checkout / Overdue invoice
                         │
                         ▼
        ML customer propensity signal
          (real Olist training data)
                         │
                         ▼
             Diagnose root cause
          (Razorpay error taxonomy)
                         │
                         ▼
              Recovery Opportunity
                    Score
                         │
                         ▼
       Deduplicate across customer's
                 open leaks
                         │
                         ▼
       Check merchant policy constraints
                         │
                         ▼
          Score permitted actions
                         │
                         ▼
       Rank & allocate fixed daily budget
                         │
                         ▼
         Execute safely / idempotently
                         │
                         ▼
       Verify outcome / handle failures
                         │
                         ▼
                 Audit trail
```

---

## Why this is different from "retry everything"

- **Not every event deserves the same effort.** The Recovery Opportunity Score combines recovery potential, revenue value, customer value, and diagnostic confidence. A fixed daily budget forces prioritization rather than indiscriminate recovery.

- **Real data wherever it genuinely exists; simulation labeled everywhere else.** Razorpay's published payment-error taxonomy and Olist's real e-commerce order/customer data are used where relevant. Specific revenue-leak events and recovery outcomes are explicitly simulated.

- **A trained ML model, used within its evidence.** A Random Forest is trained on real Olist order outcomes to produce a customer-behavior propensity signal. It is **not** presented as a recovery-success predictor because the required intervention → outcome labels do not exist in the dataset.

- **Policy is a hard gate, not a suggestion.** Contact-frequency limits, cooldowns, discount caps, recovery-attempt limits, and human-escalation thresholds are checked before execution. Blocked actions can produce an explicit fallback rather than silently proceeding.

- **Failures are handled like a real system.** An ambiguous timeout is marked `pending_verification` rather than blindly retried. An application-level idempotency ledger prevents duplicate execution.

- **The recovery number is independently evaluated.** The outcome simulator is separate from the decision engine, so the scorer does not directly determine its own evaluation result.

---

## Decision logic

### 1. Recovery Opportunity Score

The current score is:

```text
0.40 × Recovery Potential
+ 0.25 × Revenue Value
+ 0.20 × Customer Value
+ 0.15 × Diagnostic Confidence
```

These are **initial authored policy weights**, not weights learned from historical recovery outcomes.

| Component             | Weight | Current meaning                                                                         |
| --------------------- | -----: | --------------------------------------------------------------------------------------- |
| Recovery Potential    |    40% | How promising the opportunity appears based on event characteristics and recovery rules |
| Revenue Value         |    25% | Relative monetary value of the transaction/opportunity                                  |
| Customer Value        |    20% | Historical importance/reliability of the customer                                       |
| Diagnostic Confidence |    15% | Confidence in the diagnosis, **not** probability of recovery                            |

The component values are normalized prioritization signals. They should **not** be interpreted as calibrated probabilities.

- **Recovery Potential:** currently rule-based, using event characteristics such as failure reason, previous attempts and timing.
- **Revenue Value:** normalized transaction value.
- **Customer Value:** historical customer behavior, including the ML propensity signal.
- **Diagnostic Confidence:** confidence in the diagnosis, not probability that the intervention will succeed.

### 2. Action Score

For actions that pass the hard policy gate:

```text
Expected Recovery − Action Cost − Risk
```

These are normalized decision signals rather than precise financial forecasts.

- **Expected Recovery:** currently simulated/configured estimate based on event and action characteristics.
- **Action Cost:** relative cost of taking the intervention.
- **Risk:** penalty representing downside or merchant-defined risk.

### 3. Budget-constrained ranking

The system ranks allowed opportunities and allocates a simulated **₹500,000 daily recovery budget** from highest priority downward.

The ₹500,000 budget and the specific policy thresholds are **illustrative values chosen for this evaluation, not Razorpay defaults or industry-standard values**.

---

## Results

Same approximately **₹500,000 budget**, same **2,400-event batch**, independently evaluated:

| Strategy                | Events actioned | Recovery rate | Simulated recovery |
| ----------------------- | --------------: | ------------: | -----------------: |
| Naive (arbitrary order) |             196 |         13.3% |            ₹49,669 |
| **Recovery Operator**   |          **21** |     **47.6%** |       **₹165,850** |

**→ ₹116,181 more simulated recovery on essentially the identical budget.**

The operator acts on far fewer events because its constrained ranking selects only the highest-priority opportunities under the simulated policy and budget. This evaluation does **not** establish that those priorities would be optimal in production; that would require real intervention → outcome data.

### ML classifier

Real Olist labels, held-out test set:

- ROC-AUC: **0.642**
- PR-AUC: **0.0085**
- Random PR baseline: **~0.0047**

The positive class is highly imbalanced (~0.47%), so PR-AUC is more informative than accuracy for this evaluation.

The class-weighted Random Forest was selected over SMOTE after comparing both approaches on the same untouched test set.

The ML model's output is used as a **customer-behavior propensity signal**, not as a recovery-success probability.

---

## Data honesty

The distinction between real data, authored assumptions, and simulation is deliberate.

| Component                                | Real                                                                    | Simulated / authored                |
| ---------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------- |
| Payment failure taxonomy                 | Razorpay's published error structure (`reason`, `source`, `step`, etc.) | —                                   |
| Customer purchase history                | Olist — 99,441 real orders / 96,096 real customers                      | —                                   |
| ML training labels                       | Olist `order_status` outcomes                                           | —                                   |
| Specific leak events                     | —                                                                       | Simulated and explicitly flagged    |
| Event amounts                            | —                                                                       | Simulated                           |
| Scoring weights                          | —                                                                       | Authored policy weights             |
| Policy limits / budget                   | —                                                                       | Illustrative merchant-policy values |
| Action effectiveness / recovery outcomes | —                                                                       | Independent outcome simulation      |
| Razorpay execution                       | Real Razorpay Test Mode API calls                                       | —                                   |

Olist is used as the **customer-history layer**. It is not treated as a proxy for Razorpay payment behavior or recovery behavior.

The specific abandonment, payment-failure, and overdue-invoice events are generated using real customer profiles and simulated event attributes.

---

## Assumptions → production reality

The current system makes its assumptions explicit rather than presenting them as learned facts.

| Current component / parameter         | Current source                      | What production would need                                           |
| ------------------------------------- | ----------------------------------- | -------------------------------------------------------------------- |
| 40/25/20/15 Opportunity Score weights | Authored policy weights             | Historical decisions + outcomes to optimize/calibrate prioritization |
| Recovery Potential                    | Rule-based event characteristics    | Historical leak context + eventual recovery outcomes                 |
| Diagnostic Confidence                 | Authored diagnostic rules           | Confirmed historical diagnoses → calibrated diagnostic model         |
| Revenue Value                         | Normalized transaction amount       | Actual merchant transaction economics                                |
| Customer Value                        | Historical behavior + ML propensity | Longitudinal transaction history / CLV data                          |
| Expected Recovery                     | Simulated/configured                | Customer × intervention × outcome history                            |
| Action Cost                           | Simulated/configured                | Actual communication, discount, processing and human-review costs    |
| Risk                                  | Simulated/configured                | Merchant-specific risk outcomes, complaints, chargebacks, etc.       |
| Customer cooldown / message limits    | Illustrative merchant policy        | Merchant communication policy + observed customer behavior           |
| Maximum recovery attempts             | Illustrative merchant policy        | Merchant economics, risk tolerance and historical recovery curves    |
| Discount cap                          | Illustrative merchant policy        | Merchant margin and discount policy                                  |
| Human escalation threshold            | Illustrative merchant policy        | Merchant risk/operations policy and transaction economics            |
| ₹500,000 daily budget                 | Evaluation assumption               | Merchant-defined recovery budget / economic constraints              |

### What data would enable better models?

**Recovery probability**

Historical payment/revenue-leak events containing the features available at decision time and the eventual outcome could support a calibrated probability model.

**Diagnostic confidence**

Historical events with confirmed root causes would allow a diagnostic classifier to be calibrated. Methods such as **isotonic regression or Platt scaling** could turn model scores into better-calibrated probabilities.

**Action effectiveness**

The most valuable dataset would be:

```text
Customer context
      +
Revenue-leak context
      +
Intervention taken
      +
Eventual outcome
```

With enough observations, **uplift modeling** or a **contextual bandit** could be considered because the goal is to estimate the incremental effect of an intervention, rather than simply predict whether a customer will eventually pay.

**Customer value**

Longitudinal transaction data could support a proper CLV model.

**Policy weights and thresholds**

Historical recovery, cost, customer-experience and risk outcomes could be used to evaluate and calibrate the current authored parameters.

> **The production path is not "add AI everywhere." It is to collect the data required for each decision, validate the labels, and replace authored assumptions with calibrated models where sufficient evidence exists.**

---

## Real Razorpay Test Mode execution

The decision pipeline can switch from mock execution to Razorpay's real Test Mode API.

The live demonstration is deliberately capped at five actions and is logged separately in:

```text
data/audit_ledger_live_demo.csv
```

The Test Mode execution proves that the executor genuinely communicates with Razorpay's API.

It **does not** change or validate the ₹165,850 recovery figure.

The recovery figure comes exclusively from the independent offline evaluation because Test Mode does not provide independent real-world ground truth for whether the recovery strategy was effective.

---

## Problems faced and how they were handled

### No real intervention → outcome dataset

The required customer × intervention × outcome data was not available publicly.

**Response:** separate the decision engine from the outcome simulator and label the simulated evaluation explicitly.

### Multiple leaks for one customer

A customer can simultaneously appear in multiple leak streams.

**Response:** customer-level aggregation, cooldowns, and deduplication prevent independent recovery workflows from repeatedly contacting the same customer.

### Mock/live execution separation

During development, cached mock execution could potentially be reused when switching to the real executor.

**Response:** client-aware execution identity and separate mock/live ledgers.

### Ambiguous API timeout

A timeout does not establish whether the external operation succeeded.

**Response:** transition to `pending_verification` rather than blindly retrying.

---

## Honest limitations & production roadmap

| Limitation                                    | Why it exists here                                                                                             | Production direction                                                                                                          |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| No real event → action → outcome history      | No live merchant intervention dataset                                                                          | Log every decision and outcome; learn/calibrate parameters after sufficient volume                                            |
| Diagnostic confidence is authored             | No confirmed historical diagnosis labels                                                                       | Train and calibrate a diagnostic model                                                                                        |
| ML propensity signal is weak (PR-AUC ~0.0085) | Olist `order_status` reflects logistics/seller outcomes, not payment-recovery behavior                         | Train on a merchant's own payment/recovery history                                                                            |
| Recovery outcomes are simulated               | No independent live recovery ground truth                                                                      | Replace with webhook-confirmed real outcomes                                                                                  |
| Policy values are illustrative                | No real merchant configured them                                                                               | Use per-merchant configurable policies                                                                                        |
| No causal uplift measurement                  | Offline simulation cannot establish counterfactual causal effects                                              | Use randomized experimentation / causal evaluation where appropriate                                                          |
| RBI e-mandate rules are out of scope          | Current scope covers payment failures, abandonment and overdue invoices rather than subscription/mandate flows | Add a dedicated subscription-recovery module with the required compliance logic                                               |
| Real Razorpay demo capped at 5 actions        | Avoid rate limits; Test Mode does not prove ROI                                                                | Production rollout would require webhooks, monitoring, and appropriate native Razorpay idempotency mechanisms where supported |

---

## Quick start

```bash
git clone <this-repo>
cd recovery-operator
pip install -r requirements.txt
```

Download the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place these four files in `data/olist/`:

```text
olist_orders_dataset.csv
olist_customers_dataset.csv
olist_order_payments_dataset.csv
olist_order_items_dataset.csv
```

Run:

```bash
python3 src/ingest_olist.py           # real per-customer features from Olist
python3 src/ml_risk_model.py         # trains + evaluates ML propensity model
python3 src/merge_ml_risk.py          # merges ML signal into customer features
python3 src/generate_all_data.py      # generates the 2,400-event batch
python3 src/pipeline.py               # full decision pipeline (mock execution)
python3 src/compare_with_outcomes.py  # naive vs. operator evaluation
python3 src/generate_dashboard.py     # produces dashboard.html
```

Open `dashboard.html` in any browser. It is self-contained and requires no server.

### Optional: real Razorpay Test Mode execution

```bash
cp .env.example .env
# add Razorpay Test Mode Key ID / Secret

python3 src/run_live_demo.py 5
```

This makes up to five real Test Mode API calls and logs them separately. It does **not** alter the simulated recovery evaluation.

---

## Project structure

```text
config/
  reason_mapping.json       Razorpay reason codes → root cause, confidence, candidate actions
  merchant_policy.json      Illustrative hard constraints and budget
  scoring_config.json       Transparent Opportunity / Action Score weights

src/
  ingest_olist.py           Real per-customer features from Olist
  ml_risk_model.py          Real ML model: customer propensity
  merge_ml_risk.py          Merges ML signal into customer features

  simulate_*.py             Generates the 3 simulated leak-type event batches
  generate_all_data.py      Creates the combined batch with deliberate customer overlap

  diagnoser.py               Event → root cause + confidence
  reason_matcher.py          TF-IDF fallback for unmapped reason codes
  scorer.py                  Recovery Opportunity Score + Action Score
  customer_aggregator.py     Cross-leak customer deduplication
  ranker.py                  Daily-budget-constrained prioritization
  constraint_engine.py       Merchant policy enforcement / fallback
  executor.py                Safe mock or real Razorpay execution
  outcome_simulator.py       Independent outcome evaluator
  pipeline.py                Full end-to-end run
  baseline.py                Naive comparison strategy
  compare_with_outcomes.py   Same batch + same budget head-to-head evaluation
  run_live_demo.py           Capped Razorpay Test Mode demonstration
  generate_dashboard.py      Self-contained results dashboard
```

---

## Built for

**Razorpay Hackathon — Track 03: AI Revenue Recovery**
