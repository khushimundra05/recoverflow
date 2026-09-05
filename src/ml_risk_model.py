"""
ml_risk_model.py

The ONE place in this project that uses a real trained ML model on real
labels. Everywhere else (scorer.py, reason_mapping.json, merchant_policy.json)
is authored heuristics -- deliberately, because we have no real
(event, action, outcome) triples to train a recovery-decision model on (see
scoring_config.json's _production_roadmap_note).

IMPORTANT FRAMING CORRECTION: this model predicts real historical order
completion (delivered/shipped vs canceled/unavailable) -- a REAL label from
Olist. It does NOT predict, and must never be described as predicting,
whether a specific simulated payment-failure/abandonment/invoice recovery
action will succeed. Olist has no such label; no public dataset does. Calling
this a "recovery risk classifier" would overclaim and invite a fair
"where are your real recovery labels?" challenge.

What it legitimately IS: a real, trained signal about a customer's historical
reliability/propensity to complete orders, derived from real payment and
order behavior. It's used downstream as one input to the CUSTOMER VALUE
component of the scorer (alongside repeat-purchase rank, order value rank,
tenure rank) -- not as a proxy for whether THIS recovery attempt will work.

Olist is different: order_status is a REAL, ground-truth label. We define:
    order_completed_poorly = 1  if order_status in {canceled, unavailable}
    order_completed_poorly = 0  if order_status in {delivered, shipped}
(orders in other in-limbo statuses -- invoiced, processing, created, approved
-- are dropped; their eventual outcome is ambiguous, and they're a small
fraction of the data)

This is a REAL supervised learning task: real features (payment behavior,
order value, customer history), real labels, real train/test split, real
precision/recall/AUC on held-out data. Extremely imbalanced (~0.47% positive
class) -- handled explicitly by comparing class_weight='balanced' against
SMOTE oversampling on the SAME held-out test set, and by reporting
precision/recall/AUC instead of accuracy, which would be misleading here.

Output: a per-customer ml_customer_propensity_score (0-100, HIGHER = more
reliable/higher-propensity customer historically) -- inverted from the raw
cancellation probability so it reads consistently with the other
customer-value signals it's blended with in scorer.py.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "olist"
OUT_DIR = Path(__file__).parent.parent / "data"


def load_and_build_features() -> pd.DataFrame:
    orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
    payments = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
    customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
    items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")

    # Label: only keep orders with an unambiguous final outcome
    label_map = {"delivered": 0, "shipped": 0, "canceled": 1, "unavailable": 1}
    orders = orders[orders["order_status"].isin(label_map.keys())].copy()
    orders["completed_poorly"] = orders["order_status"].map(label_map)

    # Payment features: dominant payment_type, total installments, total payment value
    pay_agg = payments.groupby("order_id").agg(
        payment_type=("payment_type", lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown"),
        payment_installments=("payment_installments", "max"),
        payment_value=("payment_value", "sum"),
    ).reset_index()

    # Item features: item count, total price, total freight
    item_agg = items.groupby("order_id").agg(
        num_items=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
    ).reset_index()

    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(pay_agg, on="order_id", how="left")
    df = df.merge(item_agg, on="order_id", how="left")

    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek

    # Drop rows with missing engineered features (a handful of orders with no payment/item rows)
    df = df.dropna(subset=["payment_type", "payment_installments", "payment_value",
                            "num_items", "total_price", "total_freight"])

    return df


def _evaluate_model(model, X_test, y_test, label: str, baseline_positive_rate: float) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "label": label,
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        # PR-AUC (average precision) is the more honest metric under heavy
        # imbalance -- compare it to the "random" baseline (= the positive
        # class rate itself), not to 1.0.
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "pr_auc_random_baseline": round(baseline_positive_rate, 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),  # [[TN, FP], [FN, TP]]
    }
    return metrics, y_proba


def train_and_evaluate(df: pd.DataFrame):
    feature_cols_numeric = [
        "payment_installments", "payment_value", "num_items",
        "total_price", "total_freight", "purchase_month", "purchase_dayofweek",
    ]
    feature_cols_categorical = ["payment_type", "customer_state"]

    X = df[feature_cols_numeric + feature_cols_categorical]
    y = df["completed_poorly"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    baseline_positive_rate = y_test.mean()

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_categorical),
    ], remainder="passthrough")

    # --- Model A: class_weight='balanced' only (the original approach) ---
    model_a = SklearnPipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    model_a.fit(X_train, y_train)
    metrics_a, _ = _evaluate_model(model_a, X_test, y_test, "class_weight=balanced only", baseline_positive_rate)

    # --- Model B: SMOTE oversampling on TRAINING data only, then a plain
    # (non-reweighted) classifier. Test set is NEVER touched by SMOTE --
    # resampling the test set would leak synthetic information into
    # evaluation and invalidate the metrics. imblearn's Pipeline correctly
    # applies SMOTE only during .fit(), never during .predict()/.score().
    model_b = ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1,
        )),
    ])
    model_b.fit(X_train, y_train)
    metrics_b, _ = _evaluate_model(model_b, X_test, y_test, "SMOTE oversampling (train only)", baseline_positive_rate)

    print("=" * 70)
    print("ML CUSTOMER PROPENSITY MODEL -- comparing two imbalance-handling strategies")
    print("on the SAME held-out test set (never resampled)")
    print("Predicts REAL historical order completion (delivered/shipped vs")
    print("canceled/unavailable). Does NOT predict recovery-action outcomes --")
    print("no dataset has those labels. Used downstream as a customer-value signal.")
    print("=" * 70)
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"Positive class (completed_poorly) rate: {100 * baseline_positive_rate:.3f}% -- heavily imbalanced")
    print(f"\nNote: this project uses predict_proba() as a CONTINUOUS risk score")
    print(f"downstream (scorer.py), not a binary decision -- so ROC-AUC and PR-AUC")
    print(f"(threshold-independent) matter more here than precision/recall at the")
    print(f"default 0.5 threshold, which are reported for completeness only.\n")

    for m in [metrics_a, metrics_b]:
        print(f"--- {m['label']} ---")
        print(f"  Precision: {m['precision']} | Recall: {m['recall']} | F1: {m['f1']}")
        print(f"  ROC-AUC: {m['roc_auc']} (0.5 = random)")
        print(f"  PR-AUC: {m['pr_auc']} (random baseline = positive class rate = {m['pr_auc_random_baseline']})")
        print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {m['confusion_matrix']}")
        print()

    # Pick whichever model has the better PR-AUC (the honest metric here) for
    # the actual downstream risk scores.
    best_model, best_metrics = (model_b, metrics_b) if metrics_b["pr_auc"] >= metrics_a["pr_auc"] else (model_a, metrics_a)
    print(f"Selected for downstream use: {best_metrics['label']} (higher PR-AUC)")

    all_metrics = {"model_a_class_weight_balanced": metrics_a, "model_b_smote": metrics_b,
                    "selected_model": best_metrics["label"]}

    return best_model, all_metrics, X, feature_cols_numeric, feature_cols_categorical


def score_all_customers(model, df: pd.DataFrame, feature_cols_numeric: list, feature_cols_categorical: list) -> pd.DataFrame:
    X_all = df[feature_cols_numeric + feature_cols_categorical]
    poor_completion_proba = model.predict_proba(X_all)[:, 1]
    df = df.copy()
    df["order_poor_completion_proba"] = poor_completion_proba

    customer_risk = df.groupby("customer_unique_id").agg(
        mean_poor_completion_proba=("order_poor_completion_proba", "mean"),
        orders_scored=("order_poor_completion_proba", "count"),
    ).reset_index()
    # INVERT: higher score = more reliable/higher-propensity customer historically.
    # This reads consistently with the other customer-value signals (repeat
    # orders, order value, tenure) it gets blended with in scorer.py -- all of
    # those are "higher = better", so this should be too.
    customer_risk["ml_customer_propensity_score"] = (
        (1 - customer_risk["mean_poor_completion_proba"]) * 100
    ).round(2)
    customer_risk = customer_risk.drop(columns=["mean_poor_completion_proba"])

    return customer_risk


if __name__ == "__main__":
    df = load_and_build_features()
    print(f"Loaded {len(df):,} orders with unambiguous outcomes for training "
          f"(dropped in-limbo statuses: invoiced/processing/created/approved)\n")

    model, metrics, X, num_cols, cat_cols = train_and_evaluate(df)

    with open(OUT_DIR / "ml_classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    customer_risk = score_all_customers(model, df, num_cols, cat_cols)
    customer_risk.to_csv(OUT_DIR / "customer_propensity_scores.csv", index=False)

    print(f"\nScored {len(customer_risk):,} customers -- saved to data/customer_propensity_scores.csv")
    print(f"Metrics saved to data/ml_classifier_metrics.json")
    print(f"\nPropensity score distribution (higher = more reliable historically):\n{customer_risk['ml_customer_propensity_score'].describe()}")
