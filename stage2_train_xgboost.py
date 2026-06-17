"""
stage2_train_xgboost.py  (v2 — chronological train/test split)
─────────────────────────────────────────────────────────────
v2 FIX: switched from random stratified split to a CHRONOLOGICAL
split (train on the earlier 80% of the timeline, test on the
later 20%, no shuffling). Random splitting on time-series data
lets near-duplicate adjacent windows from the same failure session
leak between train and test, inflating apparent performance.
Cross-validation switched from StratifiedKFold to TimeSeriesSplit
for the same reason.

Usage:
  python stage2_train_xgboost.py
"""

import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")


def load_data():
    print("Loading dataset...")
    if not os.path.exists(config.DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {config.DATASET_PATH}. "
            "Run stage1_extract_features.py first."
        )

    df = pd.read_csv(config.DATASET_PATH)
    # Ensure strict chronological order before any splitting
    df = df.sort_values("window_start").reset_index(drop=True)
    print(f"  ✓ Loaded {len(df):,} windows × {len(df.columns)} columns (sorted chronologically)")

    feature_cols = [c for c in df.columns
                    if c not in ["window_start", "window_end", "label"]]
    X = df[feature_cols]
    y = df["label"]

    print(f"  Normal (0)  : {(y == 0).sum():,}")
    print(f"  Failure (1) : {(y == 1).sum():,}")

    imbalance_ratio = (y == 0).sum() / max((y == 1).sum(), 1)
    print(f"  Imbalance ratio : {imbalance_ratio:.1f}:1")

    return X, y, feature_cols


def train_xgboost(X_train, y_train, scale_pos_weight: float):
    model = XGBClassifier(
        n_estimators      = 500,
        max_depth         = 4,
        learning_rate     = 0.01,
        subsample         = 0.7,
        colsample_bytree  = 0.7,
        min_child_weight  = 5,
        gamma             = 0.1,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        scale_pos_weight  = scale_pos_weight,
        use_label_encoder = False,
        eval_metric       = "aucpr",
        random_state      = 42,
        n_jobs            = -1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, label: str = "Test set"):
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_pred_prob) if len(y_test.unique()) > 1 else 0.0

    print(f"\n  {label} Results:")
    print(f"  {'─' * 40}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    print(f"  {'─' * 40}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Failure'])}")

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "roc_auc":   round(roc_auc, 4),
    }


def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Failure"],
                yticklabels=["Normal", "Failure"])
    plt.title("XGBoost — Confusion Matrix (chronological holdout)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_PATH, "xgboost_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix saved → {path}")


def plot_roc_curve(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc         = roc_auc_score(y_test, y_pred_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("XGBoost — ROC Curve (chronological holdout)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_PATH, "xgboost_roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ ROC curve saved → {path}")


def plot_feature_importance(model, feature_cols: list):
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    top20      = importance.nlargest(20)

    plt.figure(figsize=(8, 7))
    top20.sort_values().plot(kind="barh", color="steelblue")
    plt.title("XGBoost — Top 20 Most Important Features")
    plt.xlabel("Feature Importance Score")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_PATH, "xgboost_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Feature importance plot saved → {path}")

    print("\n  Top 10 most predictive log features:")
    for feat, score in importance.nlargest(10).items():
        print(f"    {feat:<45s} {score:.4f}")


def main():
    os.makedirs(config.PLOTS_PATH, exist_ok=True)

    print("═" * 60)
    print("  Stage 2: XGBoost Model Training  (v2 — chronological split)")
    print("  MSc Research — Failure Prediction ML Pipeline")
    print("═" * 60)

    X, y, feature_cols = load_data()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"\n  ✓ Scaler saved → {config.SCALER_PATH}")

    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()
    scale_pos_weight = n_negative / max(n_positive, 1)
    print(f"\n  scale_pos_weight = {scale_pos_weight:.2f}")

    # ── Time-series cross-validation (NOT random K-fold) ───
    print("\nRunning time-series cross-validation (5 folds, forward-chaining)...")
    cv_model = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=5, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, use_label_encoder=False,
        eval_metric="aucpr", random_state=42, n_jobs=-1,
    )
    cv      = TimeSeriesSplit(n_splits=5)
    cv_f1   = cross_val_score(cv_model, X_scaled, y, cv=cv, scoring="f1")
    cv_roc  = cross_val_score(cv_model, X_scaled, y, cv=cv, scoring="roc_auc")

    print(f"  CV F1      : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"  CV ROC-AUC : {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")

    # ── Chronological 80/20 split — train on earlier time, test on later ──
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx],         y.iloc[split_idx:]

    print(f"\n  Train: {len(X_train):,} (earlier timeline) | Test: {len(X_test):,} (later, unseen timeline)")
    print(f"  Train failures: {y_train.sum()}  |  Test failures: {y_test.sum()}")

    print("\nTraining XGBoost model...")
    model = train_xgboost(X_train, y_train, scale_pos_weight)
    print("  ✓ Training complete")

    results = evaluate(model, X_test, y_test, "Test set (chronological holdout, last 20% of timeline)")
    results["cv_f1_mean"]   = round(float(cv_f1.mean()),  4)
    results["cv_roc_mean"]  = round(float(cv_roc.mean()), 4)

    print("\nGenerating plots...")
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    plot_feature_importance(model, feature_cols)

    joblib.dump(model, config.XGBOOST_MODEL_PATH)
    print(f"\n  ✓ XGBoost model saved → {config.XGBOOST_MODEL_PATH}")

    existing = {}
    if os.path.exists(config.RESULTS_PATH):
        with open(config.RESULTS_PATH) as f:
            existing = json.load(f)
    existing["xgboost"] = results
    with open(config.RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  ✓ Results saved → {config.RESULTS_PATH}")

    print("\n  Next step: python stage3_train_lstm.py")
    print("═" * 60)


if __name__ == "__main__":
    main()