"""
stage1_extract_features.py  (v4 — fixes failure-anchor timing bug)
─────────────────────────────────────────────────────────────
Queries Elasticsearch, groups logs into 30-second time windows,
extracts ML features, adds rolling trend features, labels windows.

v4 FIX (critical):
  For GRADUAL failures (memory leak, DB pool exhaustion), the
  5-minute "pre-failure" label was anchored to the STARTED event —
  but STARTED just marks when the injection script began, not when
  the system actually became critical. For a memory leak, heap is
  still ~0% right at STARTED and only becomes dangerous several
  minutes later. This meant the model was trained on "pre-failure"
  windows where heap was still normal, while the genuinely high-heap
  period (which happens AFTER STARTED) was labelled "normal" (0) —
  exactly backwards from what we want to predict.

  Fix: for MEMORY_LEAK and DB_POOL_EXHAUSTION specifically, anchor
  the 5-minute window to the first `status=CRITICAL` log instead of
  `status=STARTED`. Both injection types already emit this marker
  (heap > 70% for memory leak; pool exhausted for DB). All other
  injection types (CPU overload, slow query, gateway timeout, high
  latency) have immediate effect upon STARTED, so they keep using
  STARTED as before — no change needed for those.

  v3 fix (global_error_rate denominator bug) is also included here.

Usage:
  python stage1_extract_features.py
"""

import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from tqdm import tqdm

import config

warnings.filterwarnings("ignore")

# Failure types that take time to BUILD toward a critical state.
# For these, STARTED ≠ the actual failure onset — we anchor to
# the explicit CRITICAL marker instead. All other types have
# immediate effect, so STARTED remains a valid anchor for them.
GRADUAL_FAILURE_TYPES = {"MEMORY_LEAK", "DB_POOL_EXHAUSTION"}


def connect_es():
    es = Elasticsearch(config.ES_HOST)
    if not es.ping():
        raise ConnectionError(f"Cannot connect to Elasticsearch at {config.ES_HOST}")
    print(f"✓ Connected to Elasticsearch at {config.ES_HOST}")
    return es


def fetch_all_logs(es: Elasticsearch) -> pd.DataFrame:
    print("\nFetching logs from Elasticsearch...")
    all_docs = []
    resp = es.search(
        index=config.ES_INDEX,
        body={"query": {"match_all": {}}, "sort": [{"@timestamp": "asc"}], "size": 1000},
        scroll="5m",
    )
    scroll_id = resp["_scroll_id"]
    hits      = resp["hits"]["hits"]
    total     = resp["hits"]["total"]["value"]
    print(f"  Total log documents: {total:,}")

    with tqdm(total=total, desc="  Fetching", unit="docs") as pbar:
        while hits:
            all_docs.extend([h["_source"] for h in hits])
            pbar.update(len(hits))
            resp  = es.scroll(scroll_id=scroll_id, scroll="5m")
            hits  = resp["hits"]["hits"]

    es.clear_scroll(scroll_id=scroll_id)

    if not all_docs:
        raise ValueError("No logs found in Elasticsearch. Run load_generator.py first.")

    df = pd.DataFrame(all_docs)
    df["@timestamp"] = pd.to_datetime(df["@timestamp"], utc=True)
    df = df.sort_values("@timestamp").reset_index(drop=True)

    print(f"  ✓ Loaded {len(df):,} log entries")
    print(f"  Time range: {df['@timestamp'].min()} → {df['@timestamp'].max()}")
    return df


def find_failure_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the list of failure 'anchor' timestamps used for labelling.

    For SUDDEN failure types: anchor = STARTED event (unchanged).
    For GRADUAL failure types (memory leak, DB pool exhaustion):
      anchor = first CRITICAL event AFTER the matching STARTED event,
      for the same service + injection type. If a session never
      reached CRITICAL (e.g. recovered early), it's skipped — there's
      no legitimate "approaching failure" signal to label in that case.
    """
    print("\nFinding failure injection events...")

    started_mask = (
        df["message"].str.contains("FAILURE_INJECTION", na=False) &
        df["message"].str.contains("status=STARTED",    na=False)
    )
    started_df = df[started_mask].copy()

    if started_df.empty:
        print("  ⚠ No failure injection STARTED events found.")
        return pd.DataFrame(columns=["timestamp", "injection_type", "service", "anchor_type"])

    started_df["injection_type"] = started_df["message"].str.extract(r"type=(\w+)")
    started_df = started_df[["@timestamp", "injection_type", "service"]].rename(
        columns={"@timestamp": "timestamp"}
    )

    # Find all CRITICAL events (only relevant for gradual types)
    critical_mask = (
        df["message"].str.contains("FAILURE_INJECTION", na=False) &
        df["message"].str.contains("status=CRITICAL",   na=False)
    )
    critical_df = df[critical_mask].copy()
    if not critical_df.empty:
        critical_df["injection_type"] = critical_df["message"].str.extract(r"type=(\w+)")
        critical_df = critical_df[["@timestamp", "injection_type", "service"]].rename(
            columns={"@timestamp": "timestamp"}
        )

    anchors = []
    skipped_no_critical = 0

    for _, row in started_df.iterrows():
        inj_type = row["injection_type"]
        service  = row["service"]
        start_ts = row["timestamp"]

        if inj_type in GRADUAL_FAILURE_TYPES and not critical_df.empty:
            candidates = critical_df[
                (critical_df["service"]        == service) &
                (critical_df["injection_type"] == inj_type) &
                (critical_df["timestamp"]      > start_ts)
            ].sort_values("timestamp")

            if not candidates.empty:
                anchor_ts = candidates.iloc[0]["timestamp"]
                anchors.append({
                    "timestamp":      anchor_ts,
                    "injection_type": inj_type,
                    "service":        service,
                    "anchor_type":    "CRITICAL",
                })
            else:
                skipped_no_critical += 1
        else:
            anchors.append({
                "timestamp":      start_ts,
                "injection_type": inj_type,
                "service":        service,
                "anchor_type":    "STARTED",
            })

    result = pd.DataFrame(anchors)

    print(f"  ✓ Found {len(started_df)} STARTED events total")
    if skipped_no_critical > 0:
        print(f"  ⚠ Skipped {skipped_no_critical} gradual-failure sessions that never reached CRITICAL")
    print(f"  ✓ Using {len(result)} failure anchors for labelling:")
    for _, row in result.iterrows():
        print(f"    [{row['timestamp'].strftime('%H:%M:%S')}] "
              f"{row['injection_type']:<20} on {row['service']:<20} (anchor={row['anchor_type']})")

    return result


def build_windows(df: pd.DataFrame) -> list:
    start   = df["@timestamp"].min().floor(f"{config.WINDOW_SIZE_SECONDS}s")
    end     = df["@timestamp"].max().ceil(f"{config.WINDOW_SIZE_SECONDS}s")
    windows = []
    current = start
    freq    = pd.Timedelta(seconds=config.WINDOW_SIZE_SECONDS)
    while current < end:
        windows.append((current, current + freq))
        current += freq
    print(f"\n  Built {len(windows):,} time windows of {config.WINDOW_SIZE_SECONDS}s each")
    return windows


def extract_window_features(window_df: pd.DataFrame, service: str) -> dict:
    svc_df     = window_df[window_df["service"] == service] \
                 if "service" in window_df.columns else pd.DataFrame()
    total_logs = len(svc_df)

    if total_logs == 0:
        return {
            f"{service}__error_count":     0,
            f"{service}__warn_count":      0,
            f"{service}__error_rate":      0.0,
            f"{service}__warn_rate":       0.0,
            f"{service}__avg_response_ms": 0.0,
            f"{service}__max_response_ms": 0.0,
            f"{service}__p95_response_ms": 0.0,
            f"{service}__avg_duration_ms": 0.0,
            f"{service}__max_duration_ms": 0.0,
            f"{service}__avg_heap_pct":    0.0,
            f"{service}__max_heap_pct":    0.0,
            f"{service}__avg_cpu_pct":     0.0,
            f"{service}__max_cpu_pct":     0.0,
            f"{service}__injection_active": 0,
            f"{service}__total_logs":      0,
        }

    levels      = svc_df["level"].str.upper() if "level" in svc_df.columns else pd.Series(dtype=str)
    error_count = int((levels == "ERROR").sum())
    warn_count  = int((levels == "WARN").sum())

    resp_times = pd.to_numeric(svc_df.get("response_time_ms", pd.Series()), errors="coerce").dropna()
    durations  = pd.to_numeric(svc_df.get("duration_ms",      pd.Series()), errors="coerce").dropna()
    heap_vals  = pd.to_numeric(svc_df.get("heap_used_pct",    pd.Series()), errors="coerce").dropna()
    cpu_vals   = pd.to_numeric(svc_df.get("cpu_load_pct",     pd.Series()), errors="coerce").dropna()

    injection_active = int(
        svc_df["message"].str.contains("FAILURE_INJECTION", na=False).any()
    ) if "message" in svc_df.columns else 0

    return {
        f"{service}__error_count":     error_count,
        f"{service}__warn_count":      warn_count,
        f"{service}__error_rate":      round(error_count / total_logs, 4),
        f"{service}__warn_rate":       round(warn_count  / total_logs, 4),
        f"{service}__avg_response_ms": round(float(resp_times.mean()),         2) if len(resp_times) else 0.0,
        f"{service}__max_response_ms": round(float(resp_times.max()),          2) if len(resp_times) else 0.0,
        f"{service}__p95_response_ms": round(float(resp_times.quantile(0.95)), 2) if len(resp_times) else 0.0,
        f"{service}__avg_duration_ms": round(float(durations.mean()), 2) if len(durations) else 0.0,
        f"{service}__max_duration_ms": round(float(durations.max()),  2) if len(durations) else 0.0,
        f"{service}__avg_heap_pct":    round(float(heap_vals.mean()), 2) if len(heap_vals) else 0.0,
        f"{service}__max_heap_pct":    round(float(heap_vals.max()),  2) if len(heap_vals) else 0.0,
        f"{service}__avg_cpu_pct":     round(float(cpu_vals.mean()),  2) if len(cpu_vals)  else 0.0,
        f"{service}__max_cpu_pct":     round(float(cpu_vals.max()),   2) if len(cpu_vals)  else 0.0,
        f"{service}__injection_active": injection_active,
        f"{service}__total_logs":      total_logs,
    }


def add_cross_service_features(row: dict, services: list) -> dict:
    total_errors = sum(row.get(f"{s}__error_count", 0) for s in services)
    total_warns  = sum(row.get(f"{s}__warn_count",  0) for s in services)
    total_logs   = sum(row.get(f"{s}__total_logs",  0) for s in services)

    all_resp = [row.get(f"{s}__avg_response_ms", 0) for s in services if row.get(f"{s}__avg_response_ms", 0) > 0]
    all_heap = [row.get(f"{s}__avg_heap_pct",    0) for s in services if row.get(f"{s}__avg_heap_pct",    0) > 0]
    all_cpu  = [row.get(f"{s}__avg_cpu_pct",     0) for s in services if row.get(f"{s}__avg_cpu_pct",     0) > 0]

    row["global__total_errors"]         = total_errors
    row["global__total_warns"]          = total_warns
    row["global__system_error_rate"]    = round(total_errors / max(total_logs, 1), 4)
    row["global__avg_response_ms"]      = round(float(np.mean(all_resp)), 2) if all_resp else 0.0
    row["global__max_response_ms"]      = round(float(np.max(all_resp)),  2) if all_resp else 0.0
    row["global__avg_heap_pct"]         = round(float(np.mean(all_heap)), 2) if all_heap else 0.0
    row["global__avg_cpu_pct"]          = round(float(np.mean(all_cpu)),  2) if all_cpu  else 0.0
    row["global__services_with_errors"] = sum(
        1 for s in services if row.get(f"{s}__error_count", 0) > 0
    )
    return row


def label_window(window_start, window_end, failure_events: pd.DataFrame) -> int:
    if failure_events.empty:
        return 0
    horizon = pd.Timedelta(seconds=config.PREDICTION_HORIZON_SECONDS)
    for _, failure in failure_events.iterrows():
        failure_time = failure["timestamp"]
        if (failure_time - horizon) <= window_start < failure_time:
            return 1
    return 0


def add_rolling_features(dataset: pd.DataFrame) -> pd.DataFrame:
    print("\n  Adding rolling trend features...")

    key_patterns = [
        "error_rate", "warn_rate",
        "heap_pct", "cpu_pct",
        "response_ms", "duration_ms",
        "services_with_errors", "system_error_rate",
    ]

    numeric_cols = [
        c for c in dataset.columns
        if c not in ["window_start", "window_end", "label"]
        and dataset[c].dtype in [float, int, "float64", "int64"]
        and any(p in c for p in key_patterns)
    ]

    new_cols_added = 0
    for col in numeric_cols:
        dataset[f"{col}__roll3"] = (
            dataset[col].rolling(window=3, min_periods=1).mean().round(4)
        )
        dataset[f"{col}__roll5"] = (
            dataset[col].rolling(window=5, min_periods=1).mean().round(4)
        )
        dataset[f"{col}__delta"] = (
            dataset[col].diff().fillna(0).round(4)
        )
        new_cols_added += 3

    print(f"  ✓ Added {new_cols_added} rolling features across {len(numeric_cols)} base metrics")
    return dataset


def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs(config.PLOTS_PATH, exist_ok=True)

    print("═" * 60)
    print("  Stage 1: Feature Extraction  (v4 — failure anchor fix)")
    print("  MSc Research — Failure Prediction ML Pipeline")
    print("═" * 60)

    es       = connect_es()
    df       = fetch_all_logs(es)
    failures = find_failure_events(df)
    windows  = build_windows(df)

    print("\nExtracting features from each time window...")
    rows = []

    for win_start, win_end in tqdm(windows, desc="  Processing windows"):
        mask      = (df["@timestamp"] >= win_start) & (df["@timestamp"] < win_end)
        window_df = df[mask]

        if len(window_df) < config.MIN_LOGS_PER_WINDOW:
            continue

        row = {
            "window_start": win_start.isoformat(),
            "window_end":   win_end.isoformat(),
        }
        for service in config.SERVICES:
            row.update(extract_window_features(window_df, service))

        row = add_cross_service_features(row, config.SERVICES)
        row["label"] = label_window(win_start, win_end, failures)
        rows.append(row)

    dataset = pd.DataFrame(rows)

    print(f"\n  ✓ Base dataset: {len(dataset):,} windows × {len(dataset.columns)} columns")
    print(f"  Label distribution:")
    print(f"    Normal (0)      : {(dataset['label'] == 0).sum():,} windows")
    print(f"    Pre-failure (1) : {(dataset['label'] == 1).sum():,} windows")

    if (dataset['label'] == 1).sum() == 0:
        print("\n  ⚠ WARNING: No pre-failure windows found!")

    noisy_cols = [c for c in dataset.columns if
                  c.endswith('__info_count') or
                  c.endswith('__log_count')  or
                  c.endswith('__total_logs')]
    dataset = dataset.drop(columns=noisy_cols)
    print(f"\n  Dropped {len(noisy_cols)} noisy/internal features: {noisy_cols}")

    if "global__system_error_rate" in dataset.columns:
        rate_stats = dataset["global__system_error_rate"].describe()
        print(f"\n  global__system_error_rate sanity check:")
        print(f"    mean={rate_stats['mean']:.4f}  max={rate_stats['max']:.4f}  "
              f"% of windows ==1.0: {(dataset['global__system_error_rate'] == 1.0).mean():.1%}")

    if "order-service__avg_heap_pct" in dataset.columns:
        heap_in_failure = dataset.loc[dataset["label"] == 1, "order-service__avg_heap_pct"]
        heap_in_normal  = dataset.loc[dataset["label"] == 0, "order-service__avg_heap_pct"]
        print(f"\n  order-service__avg_heap_pct sanity check:")
        print(f"    Pre-failure (1) windows — mean heap%: {heap_in_failure.mean():.2f}")
        print(f"    Normal (0) windows      — mean heap%: {heap_in_normal.mean():.2f}")
        print(f"    (Pre-failure mean should now be noticeably HIGHER than normal)")

    dataset = add_rolling_features(dataset)

    print(f"\n  ✓ Final dataset: {len(dataset):,} windows × {len(dataset.columns)} columns")

    dataset.to_csv(config.DATASET_PATH, index=False)
    print(f"  ✓ Dataset saved → {config.DATASET_PATH}")

    feature_cols = [c for c in dataset.columns if c not in ["window_start", "window_end", "label"]]
    feature_info = {
        "total_windows":        len(dataset),
        "normal_windows":       int((dataset["label"] == 0).sum()),
        "pre_failure_windows":  int((dataset["label"] == 1).sum()),
        "feature_count":        len(feature_cols),
        "features":             feature_cols,
        "window_size_seconds":  config.WINDOW_SIZE_SECONDS,
        "prediction_horizon_s": config.PREDICTION_HORIZON_SECONDS,
        "failure_events_found": len(failures),
    }
    with open(config.FEATURE_INFO_PATH, "w") as f:
        json.dump(feature_info, f, indent=2)
    print(f"  ✓ Feature info saved → {config.FEATURE_INFO_PATH}")

    print("\n  Next step: python stage2_train_xgboost.py")
    print("═" * 60)


if __name__ == "__main__":
    main()