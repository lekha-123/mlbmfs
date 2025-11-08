#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Round-Robin (Analytical GPS-like) Baseline for MLBMFS

Reads an MLBMFS input CSV with columns:
  tenant_id, job_id, arrival_ms, compute_ms, num_gpus, [optional] qos_deadline_s

Performs a fluid, equal-share Round-Robin approximation using a global-progress
technique that runs in ~O((n_arrivals + n_completions) log n).

Outputs:
  1) rr_results.csv
     Columns:
       job_id, tenant_id, arrival_ms, finish_ms, turnaround_ms,
       num_gpus, deadline_ms, qos_violated
  2) rr_qos_summary.json
     {
       policy, cluster_gpus, total_jobs, completed_jobs,
       makespan_s, throughput_jps, latency_p95_ms,
       fairness_jain, qos_violation_rate, rebase, window
     }
  3) rr_qos_tenant.csv
     tenant_id, jobs_completed, throughput_jps

Author: MLBMFS baseline utilities
"""

import argparse
import json
import math
import heapq
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd


def load_inputs(path: str,
                window: Optional[Tuple[Optional[int], Optional[int]]] = None) -> pd.DataFrame:
    """
    Load MLBMFS inputs, optionally filter by a time window on arrival_ms.

    window: (start_ms, end_ms) – inclusive start, exclusive end.
            Pass None to skip windowing; pass (None, end) or (start, None) to one-side bound.
    """
    df = pd.read_csv(path)
    required = ["tenant_id", "job_id", "arrival_ms", "compute_ms", "num_gpus"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize types
    df["arrival_ms"] = pd.to_numeric(df["arrival_ms"], errors="coerce")
    df["compute_ms"] = pd.to_numeric(df["compute_ms"], errors="coerce")
    df["num_gpus"] = pd.to_numeric(df["num_gpus"], errors="coerce").fillna(1).clip(lower=1)
    if "qos_deadline_s" in df.columns:
        df["qos_deadline_s"] = pd.to_numeric(df["qos_deadline_s"], errors="coerce").fillna(0)
    else:
        df["qos_deadline_s"] = 0.0

    df["tenant_id"] = df["tenant_id"].astype(str)
    df["job_id"] = df["job_id"].astype(str)

    # Drop malformed rows
    df = df.dropna(subset=["arrival_ms", "compute_ms"]).copy()
    if df.empty:
        raise ValueError("No valid rows after cleaning input file.")

    # Optional windowing on arrival_ms
    if window is not None:
        start_ms, end_ms = window
        mask = pd.Series([True] * len(df))
        if start_ms is not None:
            mask &= df["arrival_ms"] >= start_ms
        if end_ms is not None:
            mask &= df["arrival_ms"] < end_ms
        df = df.loc[mask].copy()
        if df.empty:
            raise ValueError("Window filtering produced an empty dataset.")

    # Sort by arrival
    df = df.sort_values("arrival_ms").reset_index(drop=True)
    return df


def rebase_arrivals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift all arrival_ms so that min(arrival_ms) becomes 0.
    Deadlines (deadline_ms) are recomputed downstream from qos_deadline_s, so rebasing is safe.
    """
    a0 = float(df["arrival_ms"].min())
    if not math.isfinite(a0):
        return df
    df = df.copy()
    df["arrival_ms"] = df["arrival_ms"] - a0
    return df


def rr_analytical(df: pd.DataFrame, total_gpus: int) -> pd.DataFrame:
    """
    GPS-like equal-share Round-Robin in "GPU-ms work" space.

    Work for job j: w_j = compute_ms * num_gpus.
    Cluster capacity: C = total_gpus (GPU units). With K active jobs, each gets rate C/K [GPU-ms per ms].

    Returns a per-job DataFrame with:
      job_id, tenant_id, arrival_ms, finish_ms, turnaround_ms, num_gpus, deadline_ms, qos_violated
    """
    df = df.copy()
    # Precompute GPU-ms work
    df["work_gpu_ms"] = df["compute_ms"] * df["num_gpus"]

    n = len(df)
    if n == 0:
        raise ValueError("Empty input to rr_analytical")

    # Build arrival list
    arrivals = [(int(df.loc[i, "arrival_ms"]), i) for i in range(n)]
    arrivals.sort()
    arr_ptr = 0

    # Precompute deadlines in ms (absolute)
    deadline_ms = [0] * n
    for i in range(n):
        if df.loc[i, "qos_deadline_s"] > 0:
            deadline_ms[i] = int(df.loc[i, "arrival_ms"] + 1000 * df.loc[i, "qos_deadline_s"])
        else:
            deadline_ms[i] = 0

    # Simulation state
    t = arrivals[0][0]  # current time
    C = float(total_gpus)
    K = 0               # number of active jobs
    P = 0.0             # global progress

    # Min-heap of completion thresholds Tj = P_at_insert + w_j
    heap = []  # (Tj, idx)

    finish = [None] * n
    tat = [None] * n
    violated = [False] * n

    def add_arrivals_at_time(t_curr: int):
        nonlocal arr_ptr, K, P
        while arr_ptr < n and arrivals[arr_ptr][0] <= t_curr:
            _, i = arrivals[arr_ptr]
            w = float(df.loc[i, "work_gpu_ms"])
            Tj = P + w
            heapq.heappush(heap, (Tj, i))
            K += 1
            arr_ptr += 1

    # Initial arrivals at t
    add_arrivals_at_time(t)

    while arr_ptr < n or K > 0:
        t_next_arrival = arrivals[arr_ptr][0] if arr_ptr < n else None

        if K == 0:
            # Jump to next arrival
            t = t_next_arrival
            add_arrivals_at_time(t)
            continue

        # Next completion (min threshold)
        T_min, idx_min = heap[0]
        deltaP = T_min - P  # >= 0
        dt_to_complete = (deltaP * K) / C if deltaP > 0 else 0.0

        if t_next_arrival is None:
            # Only completion events remain
            t = t + dt_to_complete
            P = T_min
            # Pop all jobs tying at this P
            while heap and abs(heap[0][0] - P) <= 1e-9:
                _, i = heapq.heappop(heap)
                finish[i] = int(round(t))
                tat[i] = int(round(t - df.loc[i, "arrival_ms"]))
                if deadline_ms[i] and finish[i] > deadline_ms[i]:
                    violated[i] = True
                K -= 1
            continue
        else:
            # Compare to next arrival
            dt_to_arrival = max(0.0, t_next_arrival - t)
            if dt_to_complete <= dt_to_arrival + 1e-12:
                # Completion first (or tie -> handle completion first)
                t = t + dt_to_complete
                P = T_min
                while heap and abs(heap[0][0] - P) <= 1e-9:
                    _, i = heapq.heappop(heap)
                    finish[i] = int(round(t))
                    tat[i] = int(round(t - df.loc[i, "arrival_ms"]))
                    if deadline_ms[i] and finish[i] > deadline_ms[i]:
                        violated[i] = True
                    K -= 1
            else:
                # Arrival first: advance progress during that interval
                P = P + (C / K) * dt_to_arrival
                t = t_next_arrival
                add_arrivals_at_time(t)

    out = pd.DataFrame({
        "job_id": df["job_id"].values,
        "tenant_id": df["tenant_id"].values,
        "arrival_ms": df["arrival_ms"].astype(int).values,
        "finish_ms": finish,
        "turnaround_ms": tat,
        "num_gpus": df["num_gpus"].astype(int).values,
        "deadline_ms": deadline_ms,
        "qos_violated": violated,
    })
    return out


def compute_qos_metrics(per_job: pd.DataFrame) -> Dict:
    """Compute global QoS metrics and tenant breakdown."""
    completed = per_job.dropna(subset=["finish_ms"]).copy()
    if completed.empty:
        raise ValueError("No completed jobs to compute QoS metrics.")

    min_arrival = float(per_job["arrival_ms"].min())
    makespan_ms = float(completed["finish_ms"].max() - min_arrival)
    makespan_s = makespan_ms / 1000.0 if makespan_ms > 0 else float("nan")

    throughput_jps = float(len(completed) / makespan_s) if makespan_s and makespan_s > 0 else float("nan")
    latency_p95_ms = float(np.percentile(completed["turnaround_ms"].dropna().values, 95))

    per_tenant_counts = completed.groupby("tenant_id")["job_id"].count().astype(float)
    if makespan_s and makespan_s > 0:
        tenant_thr = per_tenant_counts.values / makespan_s
        fairness_jain = float((tenant_thr.sum() ** 2) / (len(tenant_thr) * (tenant_thr ** 2).sum()))
    else:
        fairness_jain = float("nan")

    if "deadline_ms" in completed.columns and "qos_violated" in completed.columns:
        qos_subset = completed[completed["deadline_ms"] > 0]
        qos_violation_rate = float(qos_subset["qos_violated"].astype(bool).mean()) if len(qos_subset) > 0 else 0.0
    else:
        qos_violation_rate = float("nan")

    tenant_table = per_tenant_counts.reset_index()
    tenant_table.columns = ["tenant_id", "jobs_completed"]
    tenant_table["throughput_jps"] = tenant_table["jobs_completed"] / makespan_s if makespan_s and makespan_s > 0 else np.nan

    metrics = {
        "total_jobs": int(len(per_job)),
        "completed_jobs": int(len(completed)),
        "makespan_s": makespan_s,
        "throughput_jps": throughput_jps,
        "latency_p95_ms": latency_p95_ms,
        "fairness_jain": fairness_jain,
        "qos_violation_rate": qos_violation_rate,
        "tenant_table": tenant_table
    }
    return metrics


def run_rr_analytical(input_csv: str,
                      total_gpus: int = 16,
                      out_prefix: str = "rr",
                      rebase: bool = False,
                      window: Optional[Tuple[Optional[int], Optional[int]]] = None) -> Tuple[str, str, str]:
    """
    Run the analytical RR baseline and write outputs.

    Args:
      input_csv: path to MLBMFS inputs
      total_gpus: cluster GPU capacity
      out_prefix: prefix for output files
      rebase: if True, shift min(arrival_ms) to 0 before scheduling
      window: optional (start_ms, end_ms) to filter arrivals

    Returns:
      (per_job_csv_path, summary_json_path, tenant_csv_path)
    """
    df = load_inputs(input_csv, window=window)

    # Save flags to annotate summary
    window_tuple = None
    if window is not None:
        window_tuple = (window[0] if window[0] is not None else None,
                        window[1] if window[1] is not None else None)

    if rebase:
        df = rebase_arrivals(df)

    per_job = rr_analytical(df, total_gpus=total_gpus)

    # Write per-job CSV
    per_job_csv = f"{out_prefix}_results.csv"
    per_job.to_csv(per_job_csv, index=False)

    # Compute QoS and write summaries
    metrics = compute_qos_metrics(per_job)
    summary = {
        "policy": "RoundRobin-Analytical",
        "cluster_gpus": total_gpus,
        "total_jobs": metrics["total_jobs"],
        "completed_jobs": metrics["completed_jobs"],
        "makespan_s": metrics["makespan_s"],
        "throughput_jps": metrics["throughput_jps"],
        "latency_p95_ms": metrics["latency_p95_ms"],
        "fairness_jain": metrics["fairness_jain"],
        "qos_violation_rate": metrics["qos_violation_rate"],
        "rebase": bool(rebase),
        "window": window_tuple
    }
    summary_json = f"{out_prefix}_qos_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    tenant_csv = f"{out_prefix}_qos_tenant.csv"
    metrics["tenant_table"].to_csv(tenant_csv, index=False)

    return per_job_csv, summary_json, tenant_csv


def main():
    p = argparse.ArgumentParser(description="MLBMFS Round-Robin Analytical Baseline")
    p.add_argument("--input", "-i", required=True,
                   help="Path to MLBMFS input CSV (tenant_id, job_id, arrival_ms, compute_ms, num_gpus, [qos_deadline_s])")
    p.add_argument("--gpus", "-g", type=int, default=16, help="Total GPUs in cluster (default: 16)")
    p.add_argument("--out-prefix", "-o", default="rr", help="Output file prefix (default: rr)")
    p.add_argument("--rebase", action="store_true",
                   help="Rebase arrivals so the first job starts at t=0")
    p.add_argument("--window-start-ms", type=int, default=None,
                   help="Optional arrival_ms lower bound (inclusive)")
    p.add_argument("--window-end-ms", type=int, default=None,
                   help="Optional arrival_ms upper bound (exclusive)")
    args = p.parse_args()

    window = (args.window_start_ms, args.window_end_ms) if (args.window_start_ms is not None or args.window_end_ms is not None) else None

    per_job_csv, summary_json, tenant_csv = run_rr_analytical(
        input_csv=args.input,
        total_gpus=args.gpus,
        out_prefix=args.out_prefix,
        rebase=args.rebase,
        window=window
    )

    print("✅ Done.")
    print(f"Per-job results CSV: {per_job_csv}")
    print(f"QoS summary JSON   : {summary_json}")
    print(f"Tenant breakdown   : {tenant_csv}")


if __name__ == "__main__":
    main()
