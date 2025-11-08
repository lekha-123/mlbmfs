#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build MLBMFS inputs from philly_clean.csv (Synergy integrated)

Input  : philly_clean.csv  ← from mlbmfs_preprocess.py
Output : philly_mlbmfs_inputs.csv  ← for train_and_eval.py

Columns produced:
 tenant_id        ← vc (fallback: user)
 job_id           ← jobid
 arrival_ms       ← submitted_time → epoch (ms)
 compute_ms       ← duration_s × 1000
 wait_ms          ← wait_s × 1000
 num_gpus         ← num_gpus
 gpu_util_start   ← gpu_util_start_mean
 cpu_util_start   ← cpu_util_start_mean
 mem_util_start   ← mem_util_start_mean_pct
 throughput_gpu_min ← throughput proxy
 gpu_req,cpu_req,ram_req_gb,gpu_mem_gb ← from Synergy
 job_class        ← derived from duration_s
"""

from __future__ import annotations
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- time utils ----------------
def to_epoch_ms(s: pd.Series) -> pd.Series:
    """Convert datetime column to epoch milliseconds safely."""
    return pd.to_datetime(s, errors="coerce", utc=True).view("int64") // 10**6

# ---------------- class buckets ----------------
def derive_job_class(duration_s: pd.Series) -> pd.Series:
    """
    Transparent bucketization based on observed runtime (not synthetic):
    - ≤1 min  → interactive
    - ≤10 min → short-train
    - ≤60 min → long-train
    - >60 min → batch
    """
    bins = [-np.inf, 60, 600, 3600, np.inf]
    labels = ["interactive", "short-train", "long-train", "batch"]
    return pd.cut(duration_s, bins=bins, labels=labels, right=True).astype("category")

# ---------------- main ----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--philly-clean", required=True, help="Path to philly_clean.csv")
    ap.add_argument("--output-csv", default="philly_mlbmfs_inputs.csv", help="Where to save the MLBMFS input CSV")
    args = ap.parse_args()

    src = Path(args.philly_clean)
    if not src.exists():
        sys.exit(f"[fatal] file not found: {src}")

    # Load the cleaned dataset
    df = pd.read_csv(src, parse_dates=["submitted_time","start_time","end_time"], low_memory=False)

    # Tenant: prefer VC, fallback to user
    tenant = df["vc"].astype(str)
    tenant = tenant.where(tenant.notna() & (tenant != "nan"), df["user"].astype(str))

    # Base table
    out = pd.DataFrame({
        "tenant_id": tenant,
        "job_id": df["jobid"].astype(str),
        "arrival_ms": to_epoch_ms(df["submitted_time"]),
        "compute_ms": pd.to_numeric(df["duration_s"], errors="coerce") * 1000.0,
        "wait_ms": pd.to_numeric(df["wait_s"], errors="coerce") * 1000.0,
        "num_gpus": pd.to_numeric(df["num_gpus"], errors="coerce"),
        "gpu_util_start": pd.to_numeric(df.get("gpu_util_start_mean"), errors="coerce"),
        "cpu_util_start": pd.to_numeric(df.get("cpu_util_start_mean"), errors="coerce"),
        "mem_util_start": pd.to_numeric(df.get("mem_util_start_mean_pct"), errors="coerce"),
        "throughput_gpu_min": pd.to_numeric(df.get("throughput_gpu_min"), errors="coerce"),
        "gpu_req": pd.to_numeric(df.get("gpu_req"), errors="coerce"),
        "cpu_req": pd.to_numeric(df.get("cpu_req"), errors="coerce"),
        "ram_req_gb": pd.to_numeric(df.get("ram_req_gb"), errors="coerce"),
        "gpu_mem_gb": pd.to_numeric(df.get("gpu_mem_gb"), errors="coerce"),
    })

    # Drop rows missing essentials
    out = out.dropna(subset=["arrival_ms", "compute_ms", "wait_ms", "num_gpus"])

    # Derived transparent job class
    out["job_class"] = derive_job_class(pd.to_numeric(df["duration_s"], errors="coerce"))

    # Optional synergy numerical context
    for col in ["priority","qos_deadline_s","slack_s"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    # Final cleanup
    # Final cleanup — avoid filling categorical columns with 0
    for col in out.columns:
        if str(out[col].dtype) == "category":
            out[col] = out[col].cat.add_categories(["unknown"]).fillna("unknown")
        else:
            out[col] = out[col].fillna(0)

    out.to_csv(args.output_csv, index=False)

    print(f"✔ Wrote MLBMFS inputs: {args.output_csv}  rows={len(out)}  cols={len(out.columns)}")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(out.head(8))

if __name__ == "__main__":
    sys.exit(main())
