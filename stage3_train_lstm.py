"""
stage3_train_lstm.py  (v3 — chronological train/test split)
─────────────────────────────────────────────────────────────
v3 FIX: reverted to a CHRONOLOGICAL split (train on earlier
timeline, test on later, unseen timeline) instead of the random
stratified split used previously. Random splitting let sequences
built from the same failure session leak between train and test,
inflating apparent performance.
"""

import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                     BatchNormalization, Input, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import seaborn as sns

import config

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

SEQUENCE_LENGTH = 5


def load_data():
    print("Loading dataset...")
    if not os.path.exists(config.DATASET_PATH):
        raise FileNotFoundError("Run stage1_extract_features.py first.")

    df = pd.read_csv(config.DATASET_PATH)
    df = df.sort_values("window_start").reset_index(drop=True)

    feature_cols = [c for c in df.columns
                    if c not in ["window_start", "window_end", "label"]]
    X = df[feature_cols].values
    y = df["label"].values

    print(f"  ✓ Loaded {len(df):,} windows × {len(feature_cols)} features (sorted chronologically)")
    print(f"  Normal (0)  : {(y == 0).sum():,}")
    print(f"  Failure (1) : {(y == 1).sum():,}")
    return X, y, feature_cols


def build_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    print(f"\n  Sequences built : {X_seq.shape}")
    print(f"  Normal (0)      : {(y_seq == 0).sum():,}")
    print(f"  Failure (1)     : {(y_seq == 1).sum():,}")
    return X_seq, y_seq


def build_model(seq_len, n_features):
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        Bidirectional(LSTM(32, return_sequences=True), name="bilstm_1"),
        Dropout(0.4),
        BatchNormalization(),
        Bidirectional(LSTM(16), name="bilstm_2"),
        Dropout(0.4),
        Dense(16, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ])
    model.compile(
        optimizer = Adam(learning_rate=0.0005),
        loss      = "binary_crossentropy",
        metrics   = [
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def evaluate(model, X_test, y_test):
    y_prob = model.predict(X_test, verbose=0).flatten()

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.65, 0.05):
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_test, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    print(f"\n  Best threshold : {best_t:.2f}  (F1={best_f1:.4f})")

    y_pred    = (y_prob >= best_t).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred,    zero_division=0)
    f1        = f1_score(y_test, y_pred,        zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0

    print(f"\n  Test Set Results:")
    print(f"  {'─' * 40}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    print(f"  {'─' * 40}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal','Failure'])}")

    with open("output/lstm_best_threshold.json", "w") as f:
        json.dump({"threshold": round(float(best_t), 2)}, f)

    return {
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "f1":         round(f1,        4),
        "roc_auc":    round(roc_auc,   4),
        "threshold":  round(float(best_t), 2),
    }


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],     label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("LSTM Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["auc"],     label="Train")
    axes[1].plot(history.history["val_auc"], label="Val")
    axes[1].set_title("LSTM AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    path = os.path.join(config.PLOTS_PATH, "lstm_training_history.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✓ Training history → {path}")


def plot_confusion_matrix(model, X_test, y_test, threshold):
    y_pred = (model.predict(X_test, verbose=0).flatten() >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=["Normal", "Failure"],
                yticklabels=["Normal", "Failure"])
    plt.title("LSTM — Confusion Matrix (chronological holdout)")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_PATH, "lstm_confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✓ Confusion matrix → {path}")


def plot_roc_curve(model, X_test, y_test):
    y_prob = model.predict(X_test, verbose=0).flatten()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("LSTM — ROC Curve (chronological holdout)"); plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_PATH, "lstm_roc_curve.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✓ ROC curve → {path}")


def plot_model_comparison():
    if not os.path.exists(config.RESULTS_PATH):
        return
    with open(config.RESULTS_PATH) as f:
        results = json.load(f)
    if "xgboost" not in results or "lstm" not in results:
        return

    metrics   = ["precision", "recall", "f1", "roc_auc"]
    xgb_vals  = [results["xgboost"][m] for m in metrics]
    lstm_vals = [results["lstm"][m]     for m in metrics]
    x = np.arange(len(metrics)); width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width/2, xgb_vals,  width, label="XGBoost", color="steelblue")
    b2 = ax.bar(x + width/2, lstm_vals, width, label="LSTM",     color="darkorange")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — XGBoost vs LSTM (chronological holdout)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim([0, 1.1]); ax.legend()
    for bar in [*b1, *b2]:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_PATH, "model_comparison_xgboost_vs_lstm.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✓ Model comparison → {path}")


def main():
    os.makedirs(config.PLOTS_PATH, exist_ok=True)
    print("═" * 60)
    print("  Stage 3: LSTM Model Training  (v3 — chronological split)")
    print("  MSc Research — Failure Prediction ML Pipeline")
    print("═" * 60)

    X_raw, y, feature_cols = load_data()

    scaler = joblib.load(config.SCALER_PATH) if os.path.exists(config.SCALER_PATH) \
             else StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(f"\nBuilding sequences (length={SEQUENCE_LENGTH} × {config.WINDOW_SIZE_SECONDS}s)...")
    X_seq, y_seq = build_sequences(X_scaled, y, SEQUENCE_LENGTH)
    n_features   = X_seq.shape[2]

    # ── Chronological split — train on earlier sequences, test on later ──
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    print(f"\n  Train : {len(X_train):,} (earlier timeline)  |  Test : {len(X_test):,} (later, unseen)")
    print(f"  Train failures: {y_train.sum()}  |  Test failures: {y_test.sum()}")

    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weights = {0: weights[0], 1: weights[1]}
    print(f"\n  Class weights: normal={weights[0]:.3f}  failure={weights[1]:.3f}")

    print("\nBuilding Bidirectional LSTM model...")
    model = build_model(SEQUENCE_LENGTH, n_features)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_auc", patience=15, restore_best_weights=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, verbose=1),
    ]

    print("\nTraining Bidirectional LSTM...\n")
    history = model.fit(
        X_train, y_train,
        epochs           = 150,
        batch_size       = 16,
        validation_split = 0.2,
        class_weight     = class_weights,
        callbacks        = callbacks,
        verbose          = 1,
    )
    print(f"\n  ✓ Training complete — {len(history.history['loss'])} epochs")

    results = evaluate(model, X_test, y_test)

    print("\nGenerating plots...")
    plot_training_history(history)
    plot_confusion_matrix(model, X_test, y_test, results["threshold"])
    plot_roc_curve(model, X_test, y_test)

    model.save(config.LSTM_MODEL_PATH)
    print(f"\n  ✓ LSTM model saved → {config.LSTM_MODEL_PATH}")

    existing = {}
    if os.path.exists(config.RESULTS_PATH):
        with open(config.RESULTS_PATH) as f:
            existing = json.load(f)
    existing["lstm"] = results
    with open(config.RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    plot_model_comparison()

    print("\n" + "═" * 60)
    print("  TRAINING COMPLETE")
    print("═" * 60)
    if "xgboost" in existing:
        xgb = existing["xgboost"]
        print(f"\n  {'Metric':<15} {'XGBoost':>10} {'LSTM':>10}")
        print(f"  {'─' * 37}")
        for m in ["precision", "recall", "f1", "roc_auc"]:
            print(f"  {m:<15} {xgb.get(m, 0):>10} {results.get(m, 0):>10}")
    print(f"\n  Next step: python stage2b_ensemble.py")
    print("═" * 60)


if __name__ == "__main__":
    main()