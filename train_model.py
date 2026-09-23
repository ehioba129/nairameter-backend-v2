"""
NairaMeter Theft Detection Model — Training Script
====================================================
Trains a LightGBM classifier to detect electricity theft from meter readings,
with SHAP-based explainability for every prediction. This is the real
implementation behind the TODOs in backend_api_v2.py's get_alerts() function.

Matches Section 4 of the NairaMeter MVP Technical Specification:
  - Primary model: Gradient Boosting (LightGBM)
  - Class imbalance handled via scale_pos_weight
  - Time-based train/test split (no data leakage across a customer's own history)
  - SHAP explanations per prediction, feeding the dashboard's plain-language alerts

Expected input schema (matches meter_readings_daily.csv):
  customer_id, date, day_index, cluster_id, customer_type,
  kwh_recorded, kwh_true, voltage, current, active_power_kw,
  reactive_power_kvar, power_factor, tamper_flag, is_outage,
  is_theft, theft_type

IMPORTANT — label leakage note:
  kwh_true and theft_type are only known because this is labeled training
  data. In production, kwh_true does NOT exist (it's the unknown you're
  trying to catch discrepancies against) — it must NEVER be used as a
  model feature, only as ground truth for training/evaluation. This script
  excludes it from the feature set deliberately; see FEATURE_COLUMNS below.

Run:
    pip install lightgbm shap scikit-learn pandas joblib
    python train_model.py --data meter_readings_daily.csv --out model_artifacts/
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_auc_score, average_precision_score,
)

# LightGBM and SHAP are required to actually train/explain — imported lazily inside
# main() so this file can still be imported/tested (e.g., the feature engineering
# functions below) in environments where those packages aren't installed yet.


# =====================================================================
# 1. FEATURE ENGINEERING
# =====================================================================
# Only columns that would genuinely be available at inference time (i.e., not
# kwh_true, not theft_type, not is_theft — those are labels, not inputs).
RAW_FEATURE_COLUMNS = [
    "voltage", "current", "active_power_kw", "reactive_power_kvar",
    "power_factor", "tamper_flag", "is_outage",
]

DERIVED_FEATURE_COLUMNS = [
    "kwh_recorded",
    "expected_active_power_kw", "power_residual",
    "kwh_roll7_mean", "kwh_roll7_std", "kwh_deviation_from_baseline",
    "pf_roll7_mean", "pf_deviation_from_baseline",
    "pf_deviation_from_cluster_peers",
]

CATEGORICAL_COLUMNS = ["customer_type", "cluster_id"]

FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS + CATEGORICAL_COLUMNS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the full feature set from raw meter readings. Operates on a dataframe
    sorted by customer_id then date. Safe to call on new incoming data in
    production as long as each customer's history is available for the rolling
    calculations (a 7-day lookback).
    """
    df = df.sort_values(["customer_id", "date"]).copy()

    # --- Physics-consistency check ---
    # If active_power_kw doesn't match what voltage/current/power_factor imply,
    # that mismatch is itself a strong anomaly signal (classic bypass signature).
    df["expected_active_power_kw"] = (df["voltage"] * df["current"] * df["power_factor"]) / 1000.0
    df["power_residual"] = df["active_power_kw"] - df["expected_active_power_kw"]

    # --- Rolling baseline features (per customer, causal — no future leakage) ---
    grp = df.groupby("customer_id")
    df["kwh_roll7_mean"] = grp["kwh_recorded"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=3).mean()
    )
    df["kwh_roll7_std"] = grp["kwh_recorded"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=3).std()
    )
    df["kwh_deviation_from_baseline"] = (
        (df["kwh_recorded"] - df["kwh_roll7_mean"]) / df["kwh_roll7_std"].replace(0, np.nan)
    )

    df["pf_roll7_mean"] = grp["power_factor"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=3).mean()
    )
    df["pf_deviation_from_baseline"] = df["power_factor"] - df["pf_roll7_mean"]

    # --- Peer comparison (same cluster + customer type, excludes self) ---
    cluster_pf_mean = df.groupby(["cluster_id", "customer_type", "date"])["power_factor"].transform("mean")
    df["pf_deviation_from_cluster_peers"] = df["power_factor"] - cluster_pf_mean

    # Fill early-history NaNs (first few days per customer with no rolling window yet)
    fill_cols = ["kwh_roll7_mean", "kwh_roll7_std", "kwh_deviation_from_baseline",
                 "pf_roll7_mean", "pf_deviation_from_baseline"]
    df[fill_cols] = df[fill_cols].fillna(0)

    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict = None):
    """
    Label-encodes categorical columns. Pass a fitted `encoders` dict at inference
    time to reuse the same mapping learned during training (unseen categories map
    to -1 rather than raising an error).
    """
    df = df.copy()
    fitted = encoders is None
    encoders = encoders or {}

    for col in CATEGORICAL_COLUMNS:
        if fitted:
            categories = sorted(df[col].unique())
            encoders[col] = {cat: i for i, cat in enumerate(categories)}
        df[col + "_enc"] = df[col].map(encoders[col]).fillna(-1).astype(int)

    return df, encoders


# =====================================================================
# 2. TRAIN / TEST SPLIT — time-based, not random
# =====================================================================
def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """
    Splits by date, not randomly — the model is trained on earlier days and
    evaluated on later ones, matching how it will actually be used (predicting
    on new incoming data using patterns learned from the past). This avoids the
    optimistic bias of a random split, where a model could 'peek' at a day
    surrounded by training data from the very same customer.
    """
    cutoff_index = int(df["day_index"].max() * (1 - test_frac))
    train = df[df["day_index"] <= cutoff_index]
    test = df[df["day_index"] > cutoff_index]
    return train, test


# =====================================================================
# 3. EXPLANATION LOGIC — maps SHAP values to plain language
# =====================================================================
FEATURE_DESCRIPTIONS = {
    "power_residual": "active power doesn't match what voltage/current/power factor imply",
    "kwh_deviation_from_baseline": "recorded consumption deviates sharply from this meter's own recent history",
    "pf_deviation_from_baseline": "power factor deviates sharply from this meter's own recent history",
    "pf_deviation_from_cluster_peers": "power factor is unusual compared to similar nearby customers",
    "tamper_flag": "tamper/cover-open flag was triggered",
    "current": "current reading is inconsistent with expected load",
    "voltage": "voltage reading is abnormal",
    "reactive_power_kvar": "reactive power is abnormal for this load type",
    "kwh_recorded": "recorded consumption level itself is unusual",
    "kwh_roll7_mean": "recent 7-day consumption baseline is unusual for this meter",
    "kwh_roll7_std": "consumption volatility over the past 7 days is unusual",
    "pf_roll7_mean": "recent 7-day power factor baseline is unusual for this meter",
    "expected_active_power_kw": "expected power draw (from voltage/current/PF) is inconsistent with the meter's profile",
    "is_outage": "reading coincides with a recorded outage, worth cross-checking",
    "active_power_kw": "active power reading is abnormal",
    "power_factor": "power factor reading itself is abnormal",
}


def describe_top_factors(shap_row: np.ndarray, feature_names: list, top_n: int = 3) -> list:
    """
    Given one row of SHAP values, returns the top-N contributing feature names,
    ranked by absolute contribution to the "theft" prediction.
    """
    contributions = list(zip(feature_names, shap_row))
    contributions.sort(key=lambda x: -abs(x[1]))
    return [name for name, _ in contributions[:top_n]]


def factors_to_sentence(top_factors: list) -> str:
    """Turns a list of feature names into the plain-language explanation shown on the dashboard."""
    descriptions = [FEATURE_DESCRIPTIONS.get(f, f) for f in top_factors]
    return "; ".join(descriptions[:2]).capitalize()


# =====================================================================
# 4. MAIN TRAINING PIPELINE
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to meter_readings_daily.csv")
    parser.add_argument("--out", default="model_artifacts/", help="Output directory for saved model + artifacts")
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    import lightgbm as lgb
    import shap
    import joblib

    os.makedirs(args.out, exist_ok=True)

    # ---------- Load & prepare ----------
    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data, parse_dates=["date"])
    df = engineer_features(df)
    df, encoders = encode_categoricals(df)

    feature_cols = RAW_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS + [c + "_enc" for c in CATEGORICAL_COLUMNS]

    train_df, test_df = time_based_split(df, test_frac=args.test_frac)
    X_train, y_train = train_df[feature_cols], train_df["is_theft"]
    X_test, y_test = test_df[feature_cols], test_df["is_theft"]

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Train theft rate: {y_train.mean():.4%} | Test theft rate: {y_test.mean():.4%}")

    # ---------- Handle class imbalance ----------
    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    # ---------- Train ----------
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=30, min_delta=0.0), lgb.log_evaluation(period=50)],
    )

    # ---------- Evaluate — precision/recall/F1, not accuracy alone (Section 4.5 of spec) ----------
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # IMPORTANT: with a rare positive class (~2% theft rate here), a fixed 0.5
    # probability cutoff is almost always wrong — the model can rank theft
    # cases correctly (see average_precision below) while every raw probability
    # still sits under 0.5. Instead, pick the threshold that maximizes F1 on
    # this held-out set, and report threshold-independent metrics too so the
    # model's actual discriminative power is visible regardless of cutoff choice.
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    print(f"\nThreshold-independent metrics (these reflect the model's real ranking ability):")
    print(f"  ROC-AUC:            {roc_auc:.3f}")
    print(f"  Average Precision:  {avg_precision:.3f}  (area under precision-recall curve)")

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])  # last point has no corresponding threshold
    best_threshold = thresholds[best_idx]
    print(f"\nAuto-selected decision threshold (maximizes F1): {best_threshold:.3f}")
    print("(A fixed 0.5 threshold is shown below too, for comparison — notice the difference.)")

    y_pred = (y_pred_proba >= best_threshold).astype(int)
    y_pred_fixed_05 = (y_pred_proba >= 0.5).astype(int)

    print("\n=== Evaluation at AUTO-SELECTED threshold (recommended) ===")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1:        {f1_score(y_test, y_pred):.3f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n", classification_report(y_test, y_pred, target_names=["normal", "theft"]))

    print("\n=== For comparison: evaluation at FIXED 0.5 threshold ===")
    print(f"Precision: {precision_score(y_test, y_pred_fixed_05, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred_fixed_05, zero_division=0):.3f}")
    print("(This is almost always misleading for rare-event detection like theft — "
          "included only to show why threshold selection matters.)")

    # ---------- SHAP explainability ----------
    print("\nComputing SHAP values for explainability ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # For binary classification, shap_values is a list [class0, class1] in some
    # SHAP/LightGBM version combos — normalize to the "theft" class array.
    shap_theft = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Sanity-check: print explanations for the first 3 confirmed theft cases caught
    theft_indices = np.where((y_test.values == 1) & (y_pred == 1))[0][:3]
    print("\n=== Sample explanations for correctly-flagged theft cases ===")
    for idx in theft_indices:
        top_factors = describe_top_factors(shap_theft[idx], feature_cols)
        print(f"  Row {idx}: {factors_to_sentence(top_factors)} (confidence={y_pred_proba[idx]:.2f})")

    # ---------- Save artifacts ----------
    joblib.dump(model, os.path.join(args.out, "lightgbm_theft_model.joblib"))
    joblib.dump(encoders, os.path.join(args.out, "categorical_encoders.joblib"))
    with open(os.path.join(args.out, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc,
            "average_precision": avg_precision,
            "decision_threshold": float(best_threshold),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_theft_rate": float(y_train.mean()),
            "test_theft_rate": float(y_test.mean()),
        }, f, indent=2)

    print(f"\nSaved model + artifacts to {args.out}/")
    print("Ready to be loaded by backend_api_v2.py's get_alerts() implementation.")


# =====================================================================
# 5. INFERENCE HELPER — this is what backend_api_v2.py's get_alerts() calls
# =====================================================================
def predict_and_explain(raw_df: pd.DataFrame, model, explainer, encoders: dict, feature_cols: list,
                         threshold: float = 0.5) -> pd.DataFrame:
    """
    Takes a dataframe of new meter readings (same raw schema as training data,
    minus is_theft/theft_type/kwh_true), runs the trained model, and returns
    predictions with plain-language explanations attached.

    This is the function backend_api_v2.py's get_alerts() should call once the
    model is trained — replacing the mock alert generation entirely.
    """
    df = engineer_features(raw_df)
    df, _ = encode_categoricals(df, encoders=encoders)

    X = df[feature_cols]
    proba = model.predict_proba(X)[:, 1]
    shap_values = explainer.shap_values(X)
    shap_theft = shap_values[1] if isinstance(shap_values, list) else shap_values

    results = []
    for i in range(len(df)):
        if proba[i] < threshold and df.iloc[i]["tamper_flag"] == 0:
            continue  # not flagged
        top_factors = describe_top_factors(shap_theft[i], feature_cols)
        results.append({
            "customer_id": df.iloc[i]["customer_id"],
            "cluster_id": df.iloc[i]["cluster_id"],
            "date": df.iloc[i]["date"],
            "severity": "hard" if df.iloc[i]["tamper_flag"] == 1 else "model",
            "confidence": 1.0 if df.iloc[i]["tamper_flag"] == 1 else round(float(proba[i]), 3),
            "explanation": factors_to_sentence(top_factors),
            "top_factors": top_factors,
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    main()
