"""
stage4_prediction_api.py
─────────────────────────────────────────────────────────────
FastAPI service that runs real-time failure predictions.

How it works:
  Every 30 seconds (one window), it:
    1. Queries Elasticsearch for the latest logs
    2. Extracts the same features as stage1
    3. Runs both XGBoost and LSTM models
    4. Returns a risk score (0.0 - 1.0) per service
    5. Fires an alert if risk score exceeds threshold

The React dashboard polls this API to show live predictions.

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

# ── Config ────────────────────────────────────────────────
PREDICTION_INTERVAL_SECONDS = 30     # how often to run predictions
ALERT_THRESHOLD             = 0.6    # risk score above this = alert
SEQUENCE_LENGTH             = 10     # must match stage3

# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title       = "Failure Prediction API",
    description = "Real-time microservice failure prediction — MSc Research",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # React dashboard can call this
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Shared state ──────────────────────────────────────────
state = {
    "latest_prediction":  None,
    "prediction_history": deque(maxlen=50),   # last 50 predictions
    "active_alerts":      [],
    "models_loaded":      False,
    "xgboost_model":      None,
    "lstm_model":         None,
    "scaler":             None,
    "feature_cols":       None,
    "feature_buffer":     deque(maxlen=SEQUENCE_LENGTH),  # for LSTM sequences
}
state_lock = threading.Lock()


# ──────────────────────────────────────────────────────────
#  Load models on startup
# ──────────────────────────────────────────────────────────
def load_models():
    print("Loading ML models...")
    try:
        state["xgboost_model"] = joblib.load(config.XGBOOST_MODEL_PATH)
        print(f"  ✓ XGBoost loaded  ← {config.XGBOOST_MODEL_PATH}")
    except Exception as e:
        print(f"  ✗ XGBoost failed  : {e}")

    try:
        state["lstm_model"] = tf.keras.models.load_model(config.LSTM_MODEL_PATH)
        print(f"  ✓ LSTM loaded     ← {config.LSTM_MODEL_PATH}")
    except Exception as e:
        print(f"  ✗ LSTM failed     : {e}")

    try:
        state["scaler"] = joblib.load(config.SCALER_PATH)
        print(f"  ✓ Scaler loaded   ← {config.SCALER_PATH}")
    except Exception as e:
        print(f"  ✗ Scaler failed   : {e}")

    # Load feature columns from feature_info.json
    try:
        with open(config.FEATURE_INFO_PATH) as f:
            info = json.load(f)
        state["feature_cols"] = info["features"]
        print(f"  ✓ Feature list loaded ({len(state['feature_cols'])} features)")
    except Exception as e:
        print(f"  ✗ Feature info failed : {e}")

    state["models_loaded"] = (
        state["xgboost_model"] is not None and
        state["scaler"]        is not None and
        state["feature_cols"]  is not None
    )
    print(f"\n  Models ready: {state['models_loaded']}")


# ──────────────────────────────────────────────────────────
#  Query latest logs from Elasticsearch
# ──────────────────────────────────────────────────────────
def fetch_latest_window(es: Elasticsearch) -> pd.DataFrame:
    """Fetch logs from the last WINDOW_SIZE_SECONDS seconds."""
    now   = datetime.now(timezone.utc)
    since = now - timedelta(seconds=config.WINDOW_SIZE_SECONDS)

    resp = es.search(
        index = config.ES_INDEX,
        body  = {
            "query": {
                "range": {
                    "@timestamp": {
                        "gte": since.isoformat(),
                        "lte": now.isoformat(),
                    }
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
#  Run prediction on current window
# ──────────────────────────────────────────────────────────
def run_prediction(es: Elasticsearch) -> dict:
    """
    Fetch latest window, extract features, run both models,
    return structured prediction result.
    """
    now        = datetime.now(timezone.utc)
    window_df  = fetch_latest_window(es)

    if window_df.empty:
        return build_empty_prediction(now, reason="No logs in current window")

    # ── Extract features ───────────────────────────────────
    feature_row = {}
    for service in config.SERVICES:
        feature_row.update(extract_window_features(window_df, service))
    feature_row = add_cross_service_features(feature_row, config.SERVICES)

    # Build feature vector in correct column order
    feature_vector = np.array([
        feature_row.get(col, 0.0)
        for col in state["feature_cols"]
    ]).reshape(1, -1)

    # Scale
    feature_scaled = state["scaler"].transform(feature_vector)

    # ── XGBoost prediction ─────────────────────────────────
    xgb_score = 0.0
    if state["xgboost_model"] is not None:
        xgb_prob  = state["xgboost_model"].predict_proba(feature_scaled)[0]
        xgb_score = float(xgb_prob[1])   # probability of class 1 (pre-failure)

    # ── LSTM prediction ────────────────────────────────────
    lstm_score = 0.0
    if state["lstm_model"] is not None:
        # Add current window to the sequence buffer
        state["feature_buffer"].append(feature_scaled[0])

        if len(state["feature_buffer"]) == SEQUENCE_LENGTH:
            seq         = np.array(list(state["feature_buffer"]))
            seq         = seq.reshape(1, SEQUENCE_LENGTH, -1)
            lstm_score  = float(state["lstm_model"].predict(seq, verbose=0)[0][0])

    # ── Ensemble score (average of both models) ───────────
    if state["lstm_model"] is not None and len(state["feature_buffer"]) == SEQUENCE_LENGTH:
        ensemble_score = (xgb_score + lstm_score) / 2
    else:
        ensemble_score = xgb_score   # fallback to XGBoost only

    # ── Per-service risk scores ────────────────────────────
    service_scores = {}
    for service in config.SERVICES:
        err_rate  = feature_row.get(f"{service}__error_rate",    0.0)
        warn_rate = feature_row.get(f"{service}__warn_rate",     0.0)
        resp_time = feature_row.get(f"{service}__avg_response_ms", 0.0)
        heap_pct  = feature_row.get(f"{service}__avg_heap_pct",  0.0)
        cpu_pct   = feature_row.get(f"{service}__avg_cpu_pct",   0.0)
        injection = feature_row.get(f"{service}__injection_active", 0)

        # Weighted combination of service-level signals
        svc_score = min(1.0, (
            err_rate  * 0.40 +
            warn_rate * 0.15 +
            min(resp_time / 10000, 1.0) * 0.20 +
            min(heap_pct  / 100,   1.0) * 0.15 +
            min(cpu_pct   / 100,   1.0) * 0.10 +
            injection * 0.30
        ))
        service_scores[service] = round(svc_score, 4)

    # ── Build alerts ───────────────────────────────────────
    alerts = []
    if ensemble_score >= ALERT_THRESHOLD:
        # Find which service has the highest individual score
        top_service = max(service_scores, key=service_scores.get)
        alerts.append({
            "id":           f"alert-{int(now.timestamp())}",
            "timestamp":    now.isoformat(),
            "severity":     "CRITICAL" if ensemble_score >= 0.8 else "WARNING",
            "message":      f"Failure predicted — risk score {ensemble_score:.0%}",
            "service":      top_service,
            "risk_score":   round(ensemble_score, 4),
            "predicted_in": "< 5 minutes",
        })

    # ── Build result ───────────────────────────────────────
    result = {
        "timestamp":       now.isoformat(),
        "xgboost_score":   round(xgb_score,      4),
        "lstm_score":      round(lstm_score,      4),
        "ensemble_score":  round(ensemble_score,  4),
        "risk_level":      score_to_level(ensemble_score),
        "service_scores":  service_scores,
        "alerts":          alerts,
        "log_count":       len(window_df),
        "lstm_ready":      len(state["feature_buffer"]) == SEQUENCE_LENGTH,
        "features_sample": {
            "global_error_rate":   feature_row.get("global__system_error_rate", 0.0),
            "global_avg_resp_ms":  feature_row.get("global__avg_response_ms",   0.0),
            "global_avg_heap_pct": feature_row.get("global__avg_heap_pct",      0.0),
            "global_avg_cpu_pct":  feature_row.get("global__avg_cpu_pct",       0.0),
        },
    }

    return result


def score_to_level(score: float) -> str:
    if score >= 0.8:  return "CRITICAL"
    if score >= 0.6:  return "WARNING"
    if score >= 0.4:  return "ELEVATED"
    return "NORMAL"


def build_empty_prediction(ts, reason: str = "") -> dict:
    return {
        "timestamp":      ts.isoformat(),
        "xgboost_score":  0.0,
        "lstm_score":     0.0,
        "ensemble_score": 0.0,
        "risk_level":     "NORMAL",
        "service_scores": {s: 0.0 for s in config.SERVICES},
        "alerts":         [],
        "log_count":      0,
        "lstm_ready":     False,
        "reason":         reason,
    }


# ──────────────────────────────────────────────────────────
#  Background prediction loop
# ──────────────────────────────────────────────────────────
def prediction_loop():
    es = Elasticsearch(config.ES_HOST)
    print(f"\nPrediction loop started — running every {PREDICTION_INTERVAL_SECONDS}s")

    while True:
        try:
            if state["models_loaded"]:
                result = run_prediction(es)

                with state_lock:
                    state["latest_prediction"] = result
                    state["prediction_history"].appendleft(result)

                    # Update active alerts
                    state["active_alerts"] = result.get("alerts", [])

                level = result["risk_level"]
                score = result["ensemble_score"]
                logs  = result["log_count"]
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Prediction: {level:<10} score={score:.3f} logs={logs}")

        except Exception as e:
            print(f"Prediction loop error: {e}")

        time.sleep(PREDICTION_INTERVAL_SECONDS)


# ──────────────────────────────────────────────────────────
#  API Endpoints
# ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": state["models_loaded"],
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }


@app.get("/prediction/latest")
def get_latest_prediction():
    """Latest prediction — called by dashboard every 30s."""
    with state_lock:
        if state["latest_prediction"] is None:
            return build_empty_prediction(
                datetime.now(timezone.utc),
                reason="No prediction run yet"
            )
        return state["latest_prediction"]


@app.get("/prediction/history")
def get_prediction_history():
    """Last 50 predictions — used for dashboard trend charts."""
    with state_lock:
        return {
            "count":       len(state["prediction_history"]),
            "predictions": list(state["prediction_history"]),
        }


@app.get("/alerts/active")
def get_active_alerts():
    """Active alerts only — used for dashboard alert panel."""
    with state_lock:
        return {
            "count":  len(state["active_alerts"]),
            "alerts": state["active_alerts"],
        }


@app.post("/prediction/trigger")
def trigger_prediction():
    """Manually trigger a prediction immediately — useful for viva demo."""
    if not state["models_loaded"]:
        return {"error": "Models not loaded"}
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
    print("  Failure Prediction API — Starting")
    print("  MSc Research — Microservices Failure Prediction")
    print("═" * 60)

    # Load models
    load_models()

    # Start background prediction loop in a daemon thread
    thread = threading.Thread(target=prediction_loop, daemon=True)
    thread.start()

    print("\n  API ready at http://localhost:8000")
    print("  Docs     at http://localhost:8000/docs")
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
