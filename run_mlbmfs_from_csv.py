# -*- coding: utf-8 -*-
import argparse, json
import pandas as pd
from mlbmfs_core import Config, run_mlbmfs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=r"D:\Backup-May2025\Desktop\PHD\DC MEETING\MLBMFS\code\mlbmfs_philly_pipeline\philly_mlbmfs_inputs.csv")
    ap.add_argument("--model", choices=["rf","svm","logreg"], default="rf")
    args = ap.parse_args()

    cfg = Config()
    cfg.classifier = args.model
    # Your CSV deadline is in seconds — keep this True (conversion handled inside)
    cfg.deadline_in_seconds = True

    jobs, tenants, summary = run_mlbmfs(args.csv, cfg)

    print("=== MLBMFS (Machine Learning Based Multi-Level Feedback Scheduling) ===")
    print(json.dumps(summary, indent=2))
    print("\n--- Per-tenant QoS (qostenant) ---")
    print(tenants[["tenant_id","jobs","p95_latency","target_ms","qostenant","mean_slowdown"]].to_string(index=False))

    jobs.to_csv("results_jobs.csv", index=False)
    tenants.to_csv("results_tenants.csv", index=False)
    pd.DataFrame([summary]).to_csv("results_compare.csv", index=False)
    print("\nSaved: results_jobs.csv, results_tenants.csv, results_compare.csv")
