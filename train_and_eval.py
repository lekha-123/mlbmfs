#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 (final): Train DT/RF/SVM and save all predictions for Phase 3.

Input : philly_mlbmfs_inputs.csv (built from your philly_clean.csv)
Output: predictions.csv  with columns:
        tenant_id, job_id, arrival_ms, compute_ms, wait_ms, num_gpus,
        job_class, queue_level,
        DecisionTree_pred, RandomForest_pred, SVM_pred
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

QUEUE_MAP   = {"interactive": 0, "short-train": 1, "long-train": 2, "batch": 3}
QUEUE_NAMES = ["interactive", "short-train", "long-train", "batch"]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to philly_mlbmfs_inputs.csv")
    ap.add_argument("--output", required=True, help="Path to write predictions.csv")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--rand", type=int, default=42)
    # toggle to remove compute_ms later if you want a no-leakage experiment
    ap.add_argument("--no-compute", action="store_true",
                    help="Exclude compute_ms from features (for strict prediction setup)")
    args = ap.parse_args()

    # --- Load ---
    src = Path(args.input)
    df = pd.read_csv(src)
    if "job_class" not in df.columns:
        raise SystemExit("job_class not found. Rebuild inputs first.")

    df = df[df["job_class"].notna()].copy()
    df["queue_level"] = df["job_class"].map(QUEUE_MAP).astype(int)

    # --- Features ---
    enc = LabelEncoder()
    df["tenant_id_enc"] = enc.fit_transform(df["tenant_id"].astype(str))

    base_feats = ["wait_ms", "num_gpus", "tenant_id_enc"]
    feats = base_feats if args.no_compute else (["compute_ms"] + base_feats)

    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y = df["queue_level"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=args.test_size, random_state=args.rand, stratify=y
    )

    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=args.rand),
        "RandomForest": RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=args.rand),
        "SVM": SVC(kernel="rbf", C=2.0, gamma="scale"),
    }

    # --- Train & Evaluate on holdout; then fit on all and predict full dataset ---
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)

        print(f"\n{name} report:")
        print(classification_report(y_te, y_hat, target_names=QUEUE_NAMES, digits=2))
        print("Confusion matrix:\n", confusion_matrix(y_te, y_hat))

        # Fit on ALL data and write full-dataset predictions for Phase 3
        model.fit(Xs, y)
        df[f"{name}_pred"] = model.predict(Xs)

    # --- Save compact file for simulator ---
    out_cols = [
        "tenant_id", "job_id", "arrival_ms", "compute_ms", "wait_ms", "num_gpus",
        "job_class", "queue_level",
        "DecisionTree_pred", "RandomForest_pred", "SVM_pred"
    ]
    missing_preds = [c for c in ["DecisionTree_pred","RandomForest_pred","SVM_pred"] if c not in df.columns]
    if missing_preds:
        raise SystemExit(f"Internal error: missing predictions {missing_preds}")

    out = df[out_cols].copy()
    out.to_csv(args.output, index=False)
    print(f"\n✅ Saved predictions to {args.output}")
    print(out.head(10))

if __name__ == "__main__":
    main()
