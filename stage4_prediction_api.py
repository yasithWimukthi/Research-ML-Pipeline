"""
stage4_prediction_api.py  (v2 — ensemble + rolling features)
─────────────────────────────────────────────────────────────
FastAPI service that runs real-time failure predictions
using the ensemble model (XGBoost + LSTM combined).

How it works:
  Every 30 seconds (one window), it:
    1. Queries Elasticsearch for the latest logs
    2. Extracts base features (same as stage1)
    3. Computes rolling features from a history buffer
       (mirrors what stage1's add_rolling_features does)
    4. Runs XGBoost and LSTM separately
    5. Combines scores using ensemble weights from ensemble_config.json
    6. Fires an alert if ensemble score exceeds threshold
    7. Returns per-service risk breakdown for the dashboard

IMPORTANT: This must match stage1_extract_features.py and
stage3_train_lstm.py exactly:
  - SEQUENCE_LENGTH = 5   (same as stage3)
  - Same rolling feature logic as stage1 (roll3, roll5, delta)
  - Same noisy feature removal (no info_count / log_count)

Usage:
  python stage4_prediction_api.py

Endpoints:
  GET  /health              — API health check
  GET  /prediction/latest   — Latest prediction for all services
  GET  /prediction/history  — Last 50 predictions
  GET  /alerts/active       — Active alerts only
  POST /prediction/trigger  — Manually trigger a prediction now
"""

import json
import os
import threading
import time
import warnings
from collections import deque
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import uvicorn
from elasticsearch import Elasticsearch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

import config
from stage1_extract_features import extract_window_features, add_cross_service_features

warnings.filterwarnings("ignore")

# ── Must match stage3_train_lstm.py exactly ───────────────
SEQUENCE_LENGTH              = 5
PREDICTION_INTERVAL_SECONDS  = 30

# Same key patterns used in stage1's add_rolling_features()
ROLLING_KEY_PATTERNS = [
    "error_rate", "warn_rate",
    "heap_pct", "cpu_pct",
    "response_ms", "duration_ms",
    "services_with_errors", "system_error_rate",
]

# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title       = "Failure Prediction API",
    description = "Real-time microservice failure prediction — MSc Research",
    version     = "2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Shared state ──────────────────────────────────────────
state = {
    "latest_prediction":  None,
    "prediction_history": deque(maxlen=50),
    "active_alerts":      [],
    "models_loaded":      False,
    "xgboost_model":      None,
    "lstm_model":         None,
    "scaler":             None,
    "feature_cols":       None,
    # Stores last 5 RAW base-feature dicts (pre-scaling) — used to
    # compute rolling mean / delta exactly like stage1 does on the dataset
    "rolling_buffer":     deque(maxlen=5),
    # Stores last SEQUENCE_LENGTH SCALED feature vectors — fed to LSTM
    "sequence_buffer":    deque(maxlen=SEQUENCE_LENGTH),
    # Loaded from stage2b_ensemble.py output
    "ensemble_config":    {"xgb_weight": 0.5, "lstm_weight": 0.5, "threshold": 0.65},
}
state_lock = threading.Lock()


# ──────────────────────────────────────────────────────────
#  Load models on startup
# ──────────────────────────────────────────────────────────
def load_models():
    print("Loading ML models...")

    try:
        state["xgboost_model"] = joblib.load(config.XGBOOST_MODEL_PATH)
        print(f"  ✓ XGBoost loaded   ← {config.XGBOOST_MODEL_PATH}")
    except Exception as e:
        print(f"  ✗ XGBoost failed   : {e}")

    try:
        state["lstm_model"] = tf.keras.models.load_model(config.LSTM_MODEL_PATH)
        print(f"  ✓ LSTM loaded      ← {config.LSTM_MODEL_PATH}")
    except Exception as e:
        print(f"  ✗ LSTM failed      : {e}")

    try:
        state["scaler"] = joblib.load(config.SCALER_PATH)
        print(f"  ✓ Scaler loaded    ← {config.SCALER_PATH}")
    except Exception as e:
        print(f"  ✗ Scaler failed    : {e}")

    try:
        with open(config.FEATURE_INFO_PATH) as f:
            info = json.load(f)
        state["feature_cols"] = info["features"]
        print(f"  ✓ Features loaded  : {len(state['feature_cols'])} features")
    except Exception as e:
        print(f"  ✗ Feature info failed : {e}")

    # Load ensemble weights + threshold (from stage2b_ensemble.py)
    ensemble_path = "output/ensemble_config.json"
    try:
        with open(ensemble_path) as f:
            state["ensemble_config"] = json.load(f)
        cfg = state["ensemble_config"]
        print(f"  ✓ Ensemble config  : XGB={cfg['xgb_weight']} "
              f"LSTM={cfg['lstm_weight']} threshold={cfg['threshold']}")
    except Exception:
        print(f"  ⚠ Ensemble config not found — using defaults (0.5/0.5/0.65)")
        print(f"    Run stage2b_ensemble.py to generate it")

    state["models_loaded"] = (
        state["xgboost_model"] is not None and
        state["lstm_model"]    is not None and
        state["scaler"]        is not None and
        state["feature_cols"]  is not None
    )
    print(f"\n  Models ready: {state['models_loaded']}")
    if state["feature_cols"]:
        print(f"  Expecting {len(state['feature_cols'])} features per prediction")


# ──────────────────────────────────────────────────────────
#  Query latest logs from Elasticsearch
# ──────────────────────────────────────────────────────────
def fetch_latest_window(es: Elasticsearch) -> pd.DataFrame:
    now   = datetime.now(timezone.utc)
    since = now - timedelta(seconds=config.WINDOW_SIZE_SECONDS)

    resp = es.search(
        index = config.ES_INDEX,
        body  = {
            "query": {
                "range": {
                    "@timestamp": {"gte": since.isoformat(), "lte": now.isoformat()}
                }
            },
            "size": 2000,
            "sort": [{"@timestamp": "asc"}],
        },
    )
    hits = [h["_source"] for h in resp["hits"]["hits"]]
    if not hits:
        return pd.DataFrame()

    df = pd.DataFrame(hits)
    if "@timestamp" in df.columns:
        df["@timestamp"] = pd.to_datetime(df["@timestamp"], utc=True)
    return df


# ──────────────────────────────────────────────────────────
#  Compute rolling features for the CURRENT window
#  This must mirror stage1_extract_features.add_rolling_features()
#  but computed incrementally using a sliding buffer of past windows.
# ──────────────────────────────────────────────────────────
def apply_rolling_features(current_row: dict) -> dict:
    """
    current_row  : base features for THIS window only (no rolling yet)
    Uses state["rolling_buffer"] which holds the last 5 RAW base-feature
    dicts (most recent last) to compute roll3 / roll5 / delta.
    """
    history = list(state["rolling_buffer"])   # oldest → newest, raw base features
    base_cols = [
        c for c in current_row.keys()
        if any(p in c for p in ROLLING_KEY_PATTERNS)
    ]

    for col in base_cols:
        current_val = current_row.get(col, 0.0)
        hist_vals   = [h.get(col, 0.0) for h in history] + [current_val]

        roll3 = float(np.mean(hist_vals[-3:]))
        roll5 = float(np.mean(hist_vals[-5:]))
        delta = float(current_val - hist_vals[-2]) if len(hist_vals) >= 2 else 0.0

        current_row[f"{col}__roll3"] = round(roll3, 4)
        current_row[f"{col}__roll5"] = round(roll5, 4)
        current_row[f"{col}__delta"] = round(delta, 4)

    return current_row


# ──────────────────────────────────────────────────────────
#  Run prediction on current window
# ──────────────────────────────────────────────────────────
def run_prediction(es: Elasticsearch) -> dict:
    now       = datetime.now(timezone.utc)
    window_df = fetch_latest_window(es)

    if window_df.empty:
        return build_empty_prediction(now, reason="No logs in current window")

    # ── Step 1: base features (same as stage1, no rolling yet) ──
    base_row = {}
    for service in config.SERVICES:
        base_row.update(extract_window_features(window_df, service))
    base_row = add_cross_service_features(base_row, config.SERVICES)

    # ── Step 2: rolling features using history buffer ────────────
    feature_row = apply_rolling_features(dict(base_row))

    # Push this window's RAW base features into the buffer for next time
    # (must store base_row, not feature_row, to avoid compounding rolled values)
    state["rolling_buffer"].append(base_row)

    # ── Step 3: build feature vector in EXACT trained column order ──
    missing = [c for c in state["feature_cols"] if c not in feature_row]
    feature_vector = np.array([
        feature_row.get(col, 0.0) for col in state["feature_cols"]
    ]).reshape(1, -1)

    # ── Step 4: scale ──────────────────────────────────────────────
    feature_scaled = state["scaler"].transform(feature_vector)

    # ── Step 5: XGBoost prediction ───────────────────────────────────
    xgb_score = float(state["xgboost_model"].predict_proba(feature_scaled)[0][1])

    # ── Step 6: LSTM prediction (needs SEQUENCE_LENGTH windows buffered) ──
    state["sequence_buffer"].append(feature_scaled[0])
    lstm_ready = len(state["sequence_buffer"]) == SEQUENCE_LENGTH
    lstm_score = 0.0
    if lstm_ready:
        seq        = np.array(list(state["sequence_buffer"])).reshape(1, SEQUENCE_LENGTH, -1)
        lstm_score = float(state["lstm_model"].predict(seq, verbose=0)[0][0])

    # ── Step 7: ensemble score using saved weights ───────────────────
    cfg = state["ensemble_config"]
    if lstm_ready:
        ensemble_score  = cfg["xgb_weight"] * xgb_score + cfg["lstm_weight"] * lstm_score
        alert_threshold = cfg["threshold"]
    else:
        # LSTM buffer still filling — fall back to XGBoost alone
        ensemble_score  = xgb_score
        alert_threshold = 0.6

    # ── Per-service risk scores (heuristic, for dashboard breakdown) ──
    service_scores = {}
    for service in config.SERVICES:
        err_rate   = feature_row.get(f"{service}__error_rate",        0.0)
        warn_rate  = feature_row.get(f"{service}__warn_rate",         0.0)
        resp_time  = feature_row.get(f"{service}__avg_response_ms",   0.0)
        heap_pct   = feature_row.get(f"{service}__avg_heap_pct",      0.0)
        cpu_pct    = feature_row.get(f"{service}__avg_cpu_pct",       0.0)
        injection  = feature_row.get(f"{service}__injection_active", 0)
        err_delta  = feature_row.get(f"{service}__error_rate__delta", 0.0)
        heap_delta = feature_row.get(f"{service}__avg_heap_pct__delta", 0.0)

        svc_score = min(1.0, (
            err_rate                     * 0.30 +
            warn_rate                    * 0.10 +
            min(resp_time / 10000, 1.0)  * 0.15 +
            min(heap_pct  / 100,   1.0)  * 0.15 +
            min(cpu_pct   / 100,   1.0)  * 0.10 +
            max(err_delta,  0.0)         * 0.10 +
            max(heap_delta, 0.0)         * 0.05 +
            injection                    * 0.05
        ))
        service_scores[service] = round(svc_score, 4)

    # ── Alerts ─────────────────────────────────────────────────────
    alerts = []
    if ensemble_score >= alert_threshold:
        top_service = max(service_scores, key=service_scores.get)
        severity    = "CRITICAL" if ensemble_score >= 0.80 else "WARNING"
        alerts.append({
            "id":           f"alert-{int(now.timestamp())}",
            "timestamp":    now.isoformat(),
            "severity":     severity,
            "message":      f"Failure predicted — {severity} — risk {ensemble_score:.0%}",
            "service":      top_service,
            "risk_score":   round(ensemble_score, 4),
            "predicted_in": "< 5 minutes",
        })

    return {
        "timestamp":        now.isoformat(),
        "xgboost_score":    round(xgb_score, 4),
        "lstm_score":       round(lstm_score, 4),
        "ensemble_score":   round(ensemble_score, 4),
        "risk_level":       score_to_level(ensemble_score, alert_threshold),
        "service_scores":   service_scores,
        "alerts":           alerts,
        "log_count":        len(window_df),
        "lstm_ready":       lstm_ready,
        "missing_features": missing,   # should be [] once buffer warms up
        "ensemble_weights": {
            "xgb":       cfg["xgb_weight"],
            "lstm":      cfg["lstm_weight"],
            "threshold": alert_threshold,
        },
        "features_sample": {
            "global_error_rate":   feature_row.get("global__system_error_rate", 0.0),
            "global_avg_resp_ms":  feature_row.get("global__avg_response_ms",   0.0),
            "global_avg_heap_pct": feature_row.get("global__avg_heap_pct",      0.0),
            "global_avg_cpu_pct":  feature_row.get("global__avg_cpu_pct",       0.0),
        },
    }


def score_to_level(score: float, threshold: float) -> str:
    if score >= 0.80:            return "CRITICAL"
    if score >= threshold:       return "WARNING"
    if score >= threshold * 0.7: return "ELEVATED"
    return "NORMAL"


def build_empty_prediction(ts, reason: str = "") -> dict:
    return {
        "timestamp":       ts.isoformat(),
        "xgboost_score":   0.0,
        "lstm_score":      0.0,
        "ensemble_score":  0.0,
        "risk_level":      "NORMAL",
        "service_scores":  {s: 0.0 for s in config.SERVICES},
        "alerts":          [],
        "log_count":       0,
        "lstm_ready":      False,
        "reason":          reason,
    }


# ──────────────────────────────────────────────────────────
#  Background prediction loop
# ──────────────────────────────────────────────────────────
def prediction_loop():
    es = Elasticsearch(config.ES_HOST)
    print(f"\nPrediction loop started — every {PREDICTION_INTERVAL_SECONDS}s")
    print(f"LSTM needs {SEQUENCE_LENGTH} windows to warm up "
          f"(~{SEQUENCE_LENGTH * PREDICTION_INTERVAL_SECONDS}s) before it activates\n")

    while True:
        try:
            if state["models_loaded"]:
                result = run_prediction(es)
                with state_lock:
                    state["latest_prediction"] = result
                    state["prediction_history"].appendleft(result)
                    state["active_alerts"] = result.get("alerts", [])

                level = result["risk_level"]
                score = result["ensemble_score"]
                xgb   = result["xgboost_score"]
                lstm  = result["lstm_score"]
                logs  = result["log_count"]
                ready = "✓" if result["lstm_ready"] else "⏳ warming up"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"{level:<10} ensemble={score:.3f} "
                      f"xgb={xgb:.3f} lstm={lstm:.3f} {ready} logs={logs}")

                if result["alerts"]:
                    for alert in result["alerts"]:
                        print(f"  🚨 ALERT: {alert['message']} [{alert['service']}]")

        except Exception as e:
            print(f"Prediction error: {e}")

        time.sleep(PREDICTION_INTERVAL_SECONDS)


# ──────────────────────────────────────────────────────────
#  API Endpoints
# ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    cfg = state["ensemble_config"]
    return {
        "status":          "ok",
        "models_loaded":   state["models_loaded"],
        "ensemble_config": cfg,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }


@app.get("/prediction/latest")
def get_latest_prediction():
    with state_lock:
        if state["latest_prediction"] is None:
            return build_empty_prediction(
                datetime.now(timezone.utc), reason="Prediction not run yet — wait 30s"
            )
        return state["latest_prediction"]


@app.get("/prediction/history")
def get_prediction_history():
    with state_lock:
        return {"count": len(state["prediction_history"]),
                "predictions": list(state["prediction_history"])}


@app.get("/alerts/active")
def get_active_alerts():
    with state_lock:
        return {"count": len(state["active_alerts"]), "alerts": state["active_alerts"]}


@app.post("/prediction/trigger")
def trigger_prediction():
    if not state["models_loaded"]:
        return {"error": "Models not loaded yet"}
    es     = Elasticsearch(config.ES_HOST)
    result = run_prediction(es)
    with state_lock:
        state["latest_prediction"] = result
        state["prediction_history"].appendleft(result)
        state["active_alerts"] = result.get("alerts", [])
    return result


# ──────────────────────────────────────────────────────────
#  Startup
# ──────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    print("═" * 60)
    print("  Failure Prediction API v2 — Ensemble Model")
    print("  MSc Research — Microservices Failure Prediction")
    print("═" * 60)
    load_models()
    thread = threading.Thread(target=prediction_loop, daemon=True)
    thread.start()
    print("\n  API ready  → http://localhost:8000")
    print("  API docs   → http://localhost:8000/docs")
    print("═" * 60)


# ──────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "stage4_prediction_api:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = False,
        workers = 1,
    )