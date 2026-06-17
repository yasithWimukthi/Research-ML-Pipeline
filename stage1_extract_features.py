"""
stage1_extract_features.py  (v5 — dual labelling scheme: predictive vs detection)
─────────────────────────────────────────────────────────────
Queries Elasticsearch, groups logs into 30-second time windows,
extracts ML features, adds rolling trend features, labels windows.

v5 CHANGE (methodology refinement):
  Failure types are now split into two categories with DIFFERENT
  labelling logic, because they have fundamentally different
  failure dynamics:

  GRADUAL types (currently: MEMORY_LEAK) — these have a genuine,
    multi-minute build-up before becoming critical (e.g. heap
    climbing toward exhaustion). For these, label=1 marks the
    5 minutes BEFORE the first `status=CRITICAL` log — a true
    "predict before it happens" task. Confirmed working via live
    testing (alert fired ~20pp before the system's own CRITICAL
    threshold).

  SUDDEN types (CPU_OVERLOAD, SLOW_QUERY, GATEWAY_TIMEOUT,
    HIGH_LATENCY, DB_POOL_EXHAUSTION) — these become critical
    within seconds of STARTED (e.g. DB pool exhausts in ~5s,
    first timeout fires ~35-40s later). There is no physical
    precursor signal in the preceding 5 minutes — the system is
    genuinely healthy until the injection script fires. Trying to
    label "5 min before STARTED" as pre-failure for these just
    captures ordinary pre-injection traffic, not real symptoms.

    Instead, for SUDDEN types, label=1 marks the ENTIRE active
    period from STARTED to the matching RECOVERED event (per
    service). This reframes the task for these types as RAPID
    DETECTION (recognise "currently failing" quickly) rather than
    advance prediction — an honest, achievable goal given these
    failures have no natural lead time.

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

# GRADUAL: genuine multi-minute build-up before crisis — predictive task.
GRADUAL_FAILURE_TYPES = {"MEMORY_LEAK"}

# SUDDEN: instant/near-instant onset — detection task, not prediction.
SUDDEN_FAILURE_TYPES = {
    "CPU_OVERLOAD", "SLOW_QUERY", "GATEWAY_TIMEOUT",
    "HIGH_LATENCY", "DB_POOL_EXHAUSTION",
}

# Safety cap if a SUDDEN session's RECOVERED event can't be found
# (e.g. logs end mid-session) — avoids mislabelling a huge stretch
# of unrelated future data as "failure".
MAX_SUDDEN_DURATION_MINUTES = 15


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


def find_failure_events(df: pd.DataFrame):
    """
    Returns two DataFrames:
      predictive_anchors — for GRADUAL types: a single timestamp per
        session (first CRITICAL log). label_window() marks the 5
        minutes BEFORE this as label=1.
      active_periods — for SUDDEN types: a (start, end) interval per
        session, spanning STARTED to the matching RECOVERED event
        (per service). label_window() marks ANY window inside this
        interval as label=1.
    """
    print("\nFinding failure injection events...")

    started_mask = (
        df["message"].str.contains("FAILURE_INJECTION", na=False) &
        df["message"].str.contains("status=STARTED",    na=False)
    )
    started_df = df[started_mask].copy()

    if started_df.empty:
        print("  ⚠ No failure injection STARTED events found.")
        return (pd.DataFrame(columns=["timestamp", "injection_type", "service"]),
                pd.DataFrame(columns=["start", "end", "injection_type", "service"]))

    started_df["injection_type"] = started_df["message"].str.extract(r"type=(\w+)")
    started_df = started_df[["@timestamp", "injection_type", "service"]].rename(
        columns={"@timestamp": "timestamp"}
    ).sort_values("timestamp")

    # CRITICAL events — only used for GRADUAL types
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

    # RECOVERED events — used to close out SUDDEN type active periods
    recovered_mask = (
        df["message"].str.contains("FAILURE_INJECTION", na=False) &
        df["message"].str.contains("status=RECOVERED",  na=False)
    )
    recovered_df = df[recovered_mask].copy()
    if not recovered_df.empty:
        recovered_df = recovered_df[["@timestamp", "service"]].rename(
            columns={"@timestamp": "timestamp"}
        ).sort_values("timestamp")

    predictive_anchors = []
    active_periods      = []
    skipped_no_critical  = 0
    capped_no_recovery   = 0

    for _, row in started_df.iterrows():
        inj_type = row["injection_type"]
        service  = row["service"]
        start_ts = row["timestamp"]

        if inj_type in GRADUAL_FAILURE_TYPES:
            if not critical_df.empty:
                candidates = critical_df[
                    (critical_df["service"]        == service) &
                    (critical_df["injection_type"] == inj_type) &
                    (critical_df["timestamp"]      > start_ts)
                ].sort_values("timestamp")
                if not candidates.empty:
                    predictive_anchors.append({
                        "timestamp":      candidates.iloc[0]["timestamp"],
                        "injection_type": inj_type,
                        "service":        service,
                    })
                else:
                    skipped_no_critical += 1

        elif inj_type in SUDDEN_FAILURE_TYPES:
            end_ts = None
            if not recovered_df.empty:
                candidates = recovered_df[
                    (recovered_df["service"]   == service) &
                    (recovered_df["timestamp"] > start_ts)
                ].sort_values("timestamp")
                if not candidates.empty:
                    end_ts = candidates.iloc[0]["timestamp"]

            if end_ts is None:
                end_ts = start_ts + pd.Timedelta(minutes=MAX_SUDDEN_DURATION_MINUTES)
                capped_no_recovery += 1

            active_periods.append({
                "start":          start_ts,
                "end":            end_ts,
                "injection_type": inj_type,
                "service":        service,
            })

    predictive_df = pd.DataFrame(predictive_anchors)
    active_df     = pd.DataFrame(active_periods)

    print(f"  ✓ Found {len(started_df)} STARTED events total")
    if skipped_no_critical > 0:
        print(f"  ⚠ Skipped {skipped_no_critical} gradual sessions that never reached CRITICAL")
    if capped_no_recovery > 0:
        print(f"  ⚠ Capped {capped_no_recovery} sudden sessions with no RECOVERED event found "
              f"(used {MAX_SUDDEN_DURATION_MINUTES}-min safety cap)")

    print(f"\n  ✓ {len(predictive_df)} PREDICTIVE anchors (gradual types — 5 min before CRITICAL):")
    for _, row in predictive_df.iterrows():
        print(f"    [{row['timestamp'].strftime('%H:%M:%S')}] {row['injection_type']:<15} on {row['service']}")

    print(f"\n  ✓ {len(active_df)} ACTIVE periods (sudden types — entire STARTED→RECOVERED span):")
    for _, row in active_df.iterrows():
        duration_s = (row["end"] - row["start"]).total_seconds()
        print(f"    [{row['start'].strftime('%H:%M:%S')} → {row['end'].strftime('%H:%M:%S')}] "
              f"{row['injection_type']:<15} on {row['service']:<20} ({duration_s:.0f}s)")

    return predictive_df, active_df


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


def label_window(window_start, window_end, predictive_anchors: pd.DataFrame, active_periods: pd.DataFrame) -> int:
    horizon = pd.Timedelta(seconds=config.PREDICTION_HORIZON_SECONDS)

    # GRADUAL types: 5-minute predictive window before CRITICAL
    if not predictive_anchors.empty:
        for _, anchor in predictive_anchors.iterrows():
            failure_time = anchor["timestamp"]
            if (failure_time - horizon) <= window_start < failure_time:
                return 1

    # SUDDEN types: entire active STARTED→RECOVERED period
    if not active_periods.empty:
        for _, period in active_periods.iterrows():
            if period["start"] <= window_start < period["end"]:
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
    print("  Stage 1: Feature Extraction  (v5 — dual labelling scheme)")
    print("  MSc Research — Failure Prediction ML Pipeline")
    print("═" * 60)

    es       = connect_es()
    df       = fetch_all_logs(es)
    predictive_anchors, active_periods = find_failure_events(df)
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
        row["label"] = label_window(win_start, win_end, predictive_anchors, active_periods)
        rows.append(row)

    dataset = pd.DataFrame(rows)

    print(f"\n  ✓ Base dataset: {len(dataset):,} windows × {len(dataset.columns)} columns")
    print(f"  Label distribution:")
    print(f"    Normal (0)      : {(dataset['label'] == 0).sum():,} windows")
    print(f"    Failure (1)     : {(dataset['label'] == 1).sum():,} windows")

    if (dataset['label'] == 1).sum() == 0:
        print("\n  ⚠ WARNING: No failure windows found!")

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
        print(f"    Failure (1) windows — mean heap%: {heap_in_failure.mean():.2f}")
        print(f"    Normal (0) windows  — mean heap%: {heap_in_normal.mean():.2f}")

    if "global__avg_response_ms" in dataset.columns:
        resp_in_failure = dataset.loc[dataset["label"] == 1, "global__avg_response_ms"]
        resp_in_normal  = dataset.loc[dataset["label"] == 0, "global__avg_response_ms"]
        print(f"\n  global__avg_response_ms sanity check (validates SUDDEN-type fix):")
        print(f"    Failure (1) windows — mean response ms: {resp_in_failure.mean():.2f}")
        print(f"    Normal (0) windows  — mean response ms: {resp_in_normal.mean():.2f}")
        print(f"    (Failure mean should now be MUCH higher than normal)")

    if "global__system_error_rate" in dataset.columns:
        err_in_failure = dataset.loc[dataset["label"] == 1, "global__system_error_rate"]
        err_in_normal  = dataset.loc[dataset["label"] == 0, "global__system_error_rate"]
        print(f"\n  global__system_error_rate by label (validates SUDDEN-type fix):")
        print(f"    Failure (1) windows — mean error rate: {err_in_failure.mean():.4f}")
        print(f"    Normal (0) windows  — mean error rate: {err_in_normal.mean():.4f}")

    dataset = add_rolling_features(dataset)

    print(f"\n  ✓ Final dataset: {len(dataset):,} windows × {len(dataset.columns)} columns")

    dataset.to_csv(config.DATASET_PATH, index=False)
    print(f"  ✓ Dataset saved → {config.DATASET_PATH}")

    feature_cols = [c for c in dataset.columns if c not in ["window_start", "window_end", "label"]]
    feature_info = {
        "total_windows":         len(dataset),
        "normal_windows":        int((dataset["label"] == 0).sum()),
        "failure_windows":       int((dataset["label"] == 1).sum()),
        "feature_count":         len(feature_cols),
        "features":              feature_cols,
        "window_size_seconds":   config.WINDOW_SIZE_SECONDS,
        "prediction_horizon_s":  config.PREDICTION_HORIZON_SECONDS,
        "gradual_types":         list(GRADUAL_FAILURE_TYPES),
        "sudden_types":          list(SUDDEN_FAILURE_TYPES),
        "predictive_anchors":    len(predictive_anchors),
        "active_periods":        len(active_periods),
    }
    with open(config.FEATURE_INFO_PATH, "w") as f:
        json.dump(feature_info, f, indent=2)
    print(f"  ✓ Feature info saved → {config.FEATURE_INFO_PATH}")

    print("\n  Next step: python stage2_train_xgboost.py")
    print("═" * 60)


if __name__ == "__main__":
    main()