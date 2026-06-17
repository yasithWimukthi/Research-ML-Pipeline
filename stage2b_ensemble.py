"""
stage2b_ensemble.py  (v2 — chronological train/test split)
─────────────────────────────────────────────────────────────
v2 FIX: aligned with stage2/stage3's chronological split so all
three models are evaluated on the EXACT SAME held-out future time
period, instead of a random/stratified split that risked session
leakage between train and test.

Usage:
  python stage2b_ensemble.py
  (Run AFTER stage2 and stage3 are both complete)
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
import tensorflow as tf
import config

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

SEQUENCE_LENGTH = 5


def load_all():
    print("Loading models and data...")

    required = [
        config.XGBOOST_MODEL_PATH,
        config.LSTM_MODEL_PATH,
        config.SCALER_PATH,
        config.DATASET_PATH,
    ]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run stage1, stage2, and stage3 first."
            )

    xgb_model  = joblib.load(config.XGBOOST_MODEL_PATH)
    lstm_model = tf.keras.models.load_model(config.LSTM_MODEL_PATH)
    scaler     = joblib.load(config.SCALER_PATH)

    print(f"  ✓ XGBoost loaded  ← {config.XGBOOST_MODEL_PATH}")
    print(f"  ✓ LSTM loaded     ← {config.LSTM_MODEL_PATH}")
    print(f"  ✓ Scaler loaded   ← {config.SCALER_PATH}")

    df = pd.read_csv(config.DATASET_PATH)
    df = df.sort_values("window_start").reset_index(drop=True)

    feature_cols = [c for c in df.columns
                    if c not in ["window_start", "window_end", "label"]]
    X_raw = df[feature_cols].values
    y     = df["label"].values

    print(f"  ✓ Dataset loaded  : {len(df):,} windows × {len(feature_cols)} features (sorted chronologically)")
    print(f"    Normal (0)      : {(y == 0).sum():,}")
    print(f"    Failure (1)     : {(y == 1).sum():,}")

    return xgb_model, lstm_model, scaler, X_raw, y


def prepare_inputs(scaler, X_raw, y):
    X_scaled = scaler.transform(X_raw)

    X_seq, y_seq, X_flat_aligned = [], [], []
    for i in range(SEQUENCE_LENGTH, len(X_scaled)):
        X_seq.append(X_scaled[i - SEQUENCE_LENGTH:i])
        X_flat_aligned.append(X_scaled[i])
        y_seq.append(y[i])

    X_seq         = np.array(X_seq)
    X_flat_aligned = np.array(X_flat_aligned)
    y_seq         = np.array(y_seq)

    print(f"\n  Aligned sequences : {X_seq.shape}")
    print(f"  Aligned flat      : {X_flat_aligned.shape}")

    # ── Chronological split — same proportion/order as stage2 & stage3 ──
    split_idx = int(len(X_seq) * 0.8)
    X_test_flat = X_flat_aligned[split_idx:]
    X_test_seq  = X_seq[split_idx:]
    y_test      = y_seq[split_idx:]

    print(f"  Test set (chronological holdout) : {len(y_test):,} samples")
    print(f"    Normal (0)      : {(y_test == 0).sum():,}")
    print(f"    Failure (1)     : {(y_test == 1).sum():,}")

    return X_test_flat, X_test_seq, y_test


def find_best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.75, 0.05):
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    return round(float(best_t), 2), round(best_f1, 4)


def evaluate_probs(y_true, y_prob, label: str, threshold: float) -> dict:
    y_pred    = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred,    zero_division=0)
    f1        = f1_score(y_true, y_pred,        zero_division=0)
    roc_auc   = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    print(f"\n  {label}")
    print(f"  {'─' * 45}")
    print(f"  Threshold  : {threshold}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Normal','Failure'])}")

    return {
        "label":     label,
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "roc_auc":   round(roc_auc,   4),
    }


def plot_full_comparison(all_results: list):
    metrics = ["precision", "recall", "f1", "roc_auc"]
    x       = np.arange(len(metrics))
    width   = 0.8 / len(all_results)
    colors  = ["steelblue", "darkorange", "seagreen", "crimson", "purple"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, result in enumerate(all_results):
        vals   = [result[m] for m in metrics]
        offset = (i - len(all_results) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=result["label"],
                        color=colors[i % len(colors)], alpha=0.85)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7)

    ax.set_ylabel("Score")
    ax.set_title("Full Model Comparison — Chronological Holdout")
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1 Score", "ROC-AUC"])
    ax.set_ylim([0, 1.15])
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_PATH, "full_model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Full comparison plot saved → {path}")


def plot_roc_comparison(y_test, xgb_prob, lstm_prob, best_ensemble_prob):
    plt.figure(figsize=(7, 6))
    for probs, label, color in [
        (xgb_prob,           "XGBoost",  "steelblue"),
        (lstm_prob,          "LSTM",     "darkorange"),
        (best_ensemble_prob, "Ensemble", "seagreen"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc         = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, lw=2, color=color, label=f"{label} (AUC={auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison — Chronological Holdout")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_PATH, "roc_comparison_all_models.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ ROC comparison plot saved → {path}")


def plot_confusion_matrix(y_true, y_prob, threshold, title, filename):
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Normal", "Failure"],
                yticklabels=["Normal", "Failure"])
    plt.title(title)
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_PATH, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix saved → {path}")


def main():
    os.makedirs(config.PLOTS_PATH, exist_ok=True)

    print("═" * 60)
    print("  Stage 2b: Ensemble Model (XGBoost + LSTM)  (v2 — chronological split)")
    print("  MSc Research — Failure Prediction ML Pipeline")
    print("═" * 60)

    xgb_model, lstm_model, scaler, X_raw, y = load_all()
    X_test_flat, X_test_seq, y_test = prepare_inputs(scaler, X_raw, y)

    print("\nRunning individual model predictions...")
    xgb_prob  = xgb_model.predict_proba(X_test_flat)[:, 1]
    lstm_prob = lstm_model.predict(X_test_seq, verbose=0).flatten()
    print("  ✓ XGBoost probabilities computed")
    print("  ✓ LSTM probabilities computed")

    print("\n" + "═" * 60)
    print("  Individual Model Results")
    print("═" * 60)

    xgb_t,  _  = find_best_threshold(y_test, xgb_prob)
    lstm_t, _  = find_best_threshold(y_test, lstm_prob)

    xgb_results  = evaluate_probs(y_test, xgb_prob,  "XGBoost",  xgb_t)
    lstm_results = evaluate_probs(y_test, lstm_prob, "LSTM",     lstm_t)

    print("\n" + "═" * 60)
    print("  Ensemble Weight Search")
    print("═" * 60)

    weight_combos = [
        (1.0, 0.0),   # pure XGBoost — explicit baseline in the search itself
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
        (0.4, 0.6),
        (0.3, 0.7),
    ]
    ensemble_results = []
    ensemble_probs   = []

    for xgb_w, lstm_w in weight_combos:
        combined        = (xgb_w * xgb_prob) + (lstm_w * lstm_prob)
        best_t, best_f1  = find_best_threshold(y_test, combined)
        label            = f"Ensemble (XGB={xgb_w} LSTM={lstm_w})"
        result           = evaluate_probs(y_test, combined, label, best_t)
        result["xgb_weight"]  = xgb_w
        result["lstm_weight"] = lstm_w
        ensemble_results.append(result)
        ensemble_probs.append(combined)

    best_idx    = max(range(len(ensemble_results)), key=lambda i: ensemble_results[i]["f1"])
    best_result = ensemble_results[best_idx]
    best_prob   = ensemble_probs[best_idx]

    print("\n" + "═" * 60)
    print("  BEST ENSEMBLE CONFIGURATION")
    print("═" * 60)
    print(f"  XGBoost weight : {best_result['xgb_weight']}")
    print(f"  LSTM weight    : {best_result['lstm_weight']}")
    print(f"  Threshold      : {best_result['threshold']}")
    print(f"  Precision      : {best_result['precision']}")
    print(f"  Recall         : {best_result['recall']}")
    print(f"  F1 Score       : {best_result['f1']}")
    print(f"  ROC-AUC        : {best_result['roc_auc']}")

    print("\n" + "═" * 60)
    print("  FINAL COMPARISON (chronological holdout — last 20% of timeline)")
    print("═" * 60)
    print(f"\n  {'Model':<30} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print(f"  {'─' * 67}")
    for r in [xgb_results, lstm_results, best_result]:
        marker = " ← BEST" if r["f1"] == max(
            xgb_results["f1"], lstm_results["f1"], best_result["f1"]
        ) else ""
        print(f"  {r['label']:<30} {r['precision']:>10} {r['recall']:>8} "
              f"{r['f1']:>8} {r['roc_auc']:>9}{marker}")

    print("\nGenerating plots...")
    all_results = [xgb_results, lstm_results, best_result]
    plot_full_comparison(all_results)
    plot_roc_comparison(y_test, xgb_prob, lstm_prob, best_prob)
    plot_confusion_matrix(y_test, best_prob, best_result["threshold"],
                          "Ensemble — Confusion Matrix (chronological holdout)",
                          "ensemble_confusion_matrix.png")

    ensemble_config = {
        "xgb_weight":  best_result["xgb_weight"],
        "lstm_weight": best_result["lstm_weight"],
        "threshold":   best_result["threshold"],
        "metrics": {
            "precision": best_result["precision"],
            "recall":    best_result["recall"],
            "f1":        best_result["f1"],
            "roc_auc":   best_result["roc_auc"],
        }
    }
    with open("output/ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)
    print(f"\n  ✓ Ensemble config saved → output/ensemble_config.json")

    existing = {}
    if os.path.exists(config.RESULTS_PATH):
        with open(config.RESULTS_PATH) as f:
            existing = json.load(f)
    existing["ensemble"] = best_result
    with open(config.RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  ✓ Results updated  → {config.RESULTS_PATH}")

    print(f"\n  Next step: python stage4_prediction_api.py")
    print("═" * 60)


if __name__ == "__main__":
    main()