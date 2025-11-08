#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Philly traces → philly_clean.csv (low-RAM, Synergy-ready)

What it does
- Reads cluster_job_log (JSON / JSONL / .gz) and builds clean rows: wait_s, duration_s, num_gpus, machines
- (Optional) Streams GPU/CPU/MEM util CSVs with native chunking (tiny RAM) and attaches per-job start snapshots
- Robustly normalizes PST/PDT timestamps to UTC (no deprecated args)
- Fallback: if util CSVs are skipped/unavailable, util columns remain NaN (still fine for MLBMFS baseline)
- Enriches Synergy fields: arrival_ts/start_ts/end_ts, resource request vector, priority, QoS, slack, multi_resource_vec
- Adds throughput proxy: throughput_gpu_min = num_gpus × duration_minutes

Usage (baseline: no util; fastest & safest):
  python mlbmfs_preprocess.py --archive "D:\\...\\philly_trace_extracted" --outcsv philly_clean.csv

Add util later (still low-RAM):
  python mlbmfs_preprocess.py --archive "D:\\...\\philly_trace_extracted" --outcsv philly_clean.csv ^
    --use-gpu --use-cpu --use-mem --chunksize 5000 --util-merge-tol-min 3
"""

from __future__ import annotations
import sys, json, gzip, io, re
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import argparse

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Philly -> philly_clean.csv (util optional, PST/PDT safe)")
    ap.add_argument("--archive", required=True, help="Path to extracted folder (the one that contains trace-data or is trace-data)")
    ap.add_argument("--outcsv", default="philly_clean.csv", help="Output CSV file")
    ap.add_argument("--chunksize", type=int, default=5000, help="Rows per chunk for util streaming (native pandas mode)")
    ap.add_argument("--use-gpu", action="store_true", help="Parse cluster_gpu_util.csv")
    ap.add_argument("--use-cpu", action="store_true", help="Parse cluster_cpu_util.csv")
    ap.add_argument("--use-mem", action="store_true", help="Parse cluster_mem_util.csv")
    ap.add_argument("--util-merge-tol-min", type=int, default=2, help="Nearest-merge tolerance minutes (default 2)")
    return ap.parse_args()

# ------------- paths & finders -------------
_TZ_DROP = re.compile(r"\s+(PST|PDT)\b", re.IGNORECASE)

def _to_dt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return pd.NaT
    return pd.to_datetime(_TZ_DROP.sub("", str(x)), errors="coerce", utc=True)

def find_by_stem(root: Path, stem: str) -> Optional[Path]:
    s = stem.lower()
    for p in root.rglob("*"):
        if p.is_file() and (p.stem.lower() == s or p.name.lower().startswith(s + ".")):
            return p
    return None

def locate_all(root: Path) -> dict[str, Optional[Path]]:
    return {
        "job_log":      find_by_stem(root, "cluster_job_log"),
        "gpu_util":     find_by_stem(root, "cluster_gpu_util"),
        "cpu_util":     find_by_stem(root, "cluster_cpu_util"),
        "mem_util":     find_by_stem(root, "cluster_mem_util"),
        "machine_list": find_by_stem(root, "cluster_machine_list"),
    }

# ------------- job log reader -------------
def read_job_log_any(path: Optional[Path]) -> list[dict]:
    if not path or not path.exists(): return []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        head = f.read(2); f.seek(0)
        if head.strip().startswith("["):   # JSON array
            return json.load(f)
        # JSONL
        return [json.loads(line) for line in f if line.strip()]

def parse_job_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    jobid = rec.get("jobid") or rec.get("jobId") or rec.get("jobID")
    submitted = _to_dt(rec.get("submitted_time") or rec.get("submitTime"))
    user, vc = rec.get("user"), rec.get("vc")
    status = rec.get("status") or rec.get("jobStatus")

    # Optional resource hints at top-level
    top_gpu   = rec.get("gpuCount") or rec.get("gpus") or rec.get("num_gpus")
    top_cpu   = rec.get("cpuCount") or rec.get("cpu") or rec.get("cpus")
    top_memMB = rec.get("memoryMB") or rec.get("memMB") or rec.get("memory_mb")
    priority  = rec.get("priority") or rec.get("jobPriority")
    qos_deadline_s = rec.get("qos_deadline_s")

    attempts = rec.get("attempts") or []
    start_times, end_times, machines, gcounts = [], [], set(), []
    cpu_req, ram_req_gb, gpu_mem_gb = None, None, None

    for att in attempts:
        st, et = att.get("start_time"), att.get("end_time")
        if st: start_times.append(_to_dt(st))
        if et: end_times.append(_to_dt(et))
        for d in (att.get("detail") or []):
            ip = d.get("ip")
            if ip: machines.add(ip)
            gpus = d.get("gpus") or []
            if gpus: gcounts.append(len(gpus))
            cpu_req = cpu_req or d.get("cpu") or d.get("cpus")
            mem_mb = d.get("memoryMB") or d.get("memMB") or d.get("memory_mb")
            if mem_mb is not None:
                try: ram_req_gb = float(mem_mb) / 1024.0
                except: pass
            gm = d.get("gpu_mem_mb") or d.get("gpuMemMB") or d.get("gpu_mem_gb")
            if gm is not None:
                try:
                    gm = float(gm)
                    gpu_mem_gb = gm / 1024.0 if gm > 256 else gm
                except: pass

    start_dt = min([t for t in start_times if pd.notna(t)], default=pd.NaT)
    end_dt   = max([t for t in end_times if pd.notna(t)], default=pd.NaT)
    duration_s = (end_dt - start_dt).total_seconds() if (pd.notna(start_dt) and pd.notna(end_dt)) else np.nan
    wait_s     = (start_dt - submitted).total_seconds() if (pd.notna(start_dt) and pd.notna(submitted)) else np.nan

    num_gpus = (max(gcounts) if gcounts else None) or top_gpu
    if num_gpus is not None:
        try: num_gpus = int(num_gpus)
        except: num_gpus = None

    if ram_req_gb is None and top_memMB is not None:
        try: ram_req_gb = float(top_memMB) / 1024.0
        except: pass
    if cpu_req is None and top_cpu is not None:
        try: cpu_req = int(top_cpu)
        except: pass

    return dict(
        jobid=jobid, submitted_time=submitted, start_time=start_dt, end_time=end_dt,
        wait_s=wait_s, duration_s=duration_s, status=status, user=user, vc=vc,
        num_gpus=num_gpus, machines=";".join(sorted(machines)) if machines else None,
        priority_raw=priority, qos_deadline_s_raw=qos_deadline_s,
        cpu_req_raw=cpu_req, ram_req_gb_raw=ram_req_gb, gpu_mem_gb_raw=gpu_mem_gb
    )

# ------------- PST/PDT → UTC parsing for util CSVs -------------
def _clean_pst_pdt_to_utc(s: pd.Series) -> pd.Series:
    """
    Convert strings like '2017-11-17 00:05:00 PST' / '2017-10-30 21:32:00 PDT' to UTC tz-aware.
    No deprecated args; avoids dateutil fallback warnings by using explicit format.
    """
    stripped = (
        s.astype(str)
        .str.replace(r"\b(PST|PDT)\b", "", regex=True)
        .str.strip()
    )
    fmt = "%Y-%m-%d %H:%M:%S"
    # Fast path parse; no inference
    dt = pd.to_datetime(stripped, format=fmt, errors="coerce", utc=False)
    # Fallback: trim stray non-time chars then re-parse with the same explicit format
    mask = dt.isna()
    if mask.any():
        cleaned = (
            stripped.loc[mask]
            .str.replace(r"[^0-9:\-\s]", "", regex=True)
            .str.strip()
        )
        dt.loc[mask] = pd.to_datetime(cleaned, format=fmt, errors="coerce", utc=False)

    pacific = "America/Los_Angeles"
    try:
        dt = dt.dt.tz_localize(pacific, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    except Exception:
        dt = dt.dt.tz_localize(pacific).dt.tz_convert("UTC")
    return dt

# --- stream-safe row-wise mean (prevents array blowups) ---
def _rowwise_mean_acc(chunk: pd.DataFrame, cols: list[str], out_col: str, dtype="float32") -> None:
    """
    Compute row-wise mean over 'cols' without materializing a large 2-D array.
    Writes into chunk[out_col]; leaves NaN where all cols are NaN.
    """
    if not cols:
        chunk[out_col] = np.nan
        return

    acc = None
    cnt = None
    for c in cols:
        if c not in chunk.columns:
            continue
        v = pd.to_numeric(chunk[c], errors="coerce").astype(dtype)
        if acc is None:
            acc = v.fillna(0)
            cnt = v.notna().astype("float32")
        else:
            acc = acc.add(v.fillna(0), fill_value=0)
            cnt = cnt.add(v.notna().astype("float32"), fill_value=0)

    if acc is None:
        chunk[out_col] = np.nan
    else:
        chunk[out_col] = acc / cnt.replace(0, np.nan)

# ------------- util streaming with native chunking -------------
def _sniff_sep(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(65536).decode("utf-8", errors="ignore")
    first = head.splitlines()[0] if head else ""
    return "," if first.count(",") >= first.count("\t") else "\t"

def load_gpu_util_stream(path: Optional[Path], chunksize: int) -> pd.DataFrame:
    if not path or not path.exists(): return pd.DataFrame()
    sep = _sniff_sep(path)
    rows = []
    for chunk in pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip",
                             low_memory=True, chunksize=chunksize):
        chunk.columns = [c.strip().replace(" ", "_").lower() for c in chunk.columns]
        if "time" not in chunk.columns: continue
        chunk["time"] = _clean_pst_pdt_to_utc(chunk["time"])
        if "machine_id" not in chunk.columns:
            if "machineid" in chunk.columns: chunk.rename(columns={"machineid":"machine_id"}, inplace=True)
            else: continue
        gcols = [c for c in chunk.columns if c.startswith("gpu") and c.endswith("_util")]
        _rowwise_mean_acc(chunk, gcols, "gpu_util_mean")  # float32, safe
        rows.append(chunk[["machine_id","time","gpu_util_mean"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["machine_id","time","gpu_util_mean"])

def load_cpu_util_stream(path: Optional[Path], chunksize: int) -> pd.DataFrame:
    if not path or not path.exists(): return pd.DataFrame()
    sep = _sniff_sep(path)
    rows = []
    for chunk in pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip",
                             low_memory=True, chunksize=chunksize):
        chunk.columns = [c.strip().replace(" ", "_").lower() for c in chunk.columns]
        if "time" not in chunk.columns: continue
        chunk["time"] = _clean_pst_pdt_to_utc(chunk["time"])
        if "machine_id" not in chunk.columns:
            if "machineid" in chunk.columns: chunk.rename(columns={"machineid":"machine_id"}, inplace=True)
            else: continue
        if "cpu_util" in chunk.columns:
            chunk["cpu_util"] = pd.to_numeric(chunk["cpu_util"], errors="coerce").astype("float32")
        else:
            cc = [c for c in chunk.columns if c.startswith("cpu") and c.endswith("_util")]
            _rowwise_mean_acc(chunk, cc, "cpu_util")
        rows.append(chunk[["machine_id","time","cpu_util"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["machine_id","time","cpu_util"])

def load_mem_util_stream(path: Optional[Path], chunksize: int) -> pd.DataFrame:
    if not path or not path.exists(): return pd.DataFrame()
    sep = _sniff_sep(path)
    rows = []
    for chunk in pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip",
                             low_memory=True, chunksize=chunksize):
        chunk.columns = [c.strip().replace(" ", "_").lower() for c in chunk.columns]
        if "time" not in chunk.columns: continue
        chunk["time"] = _clean_pst_pdt_to_utc(chunk["time"])
        if "machine_id" not in chunk.columns:
            if "machineid" in chunk.columns: chunk.rename(columns={"machineid":"machine_id"}, inplace=True)
            else: continue
        if {"mem_total","mem_free"}.issubset(chunk.columns):
            chunk["mem_total"] = pd.to_numeric(chunk["mem_total"], errors="coerce").astype("float32")
            chunk["mem_free"]  = pd.to_numeric(chunk["mem_free"], errors="coerce").astype("float32")
            chunk["mem_util_pct"] = (1 - (chunk["mem_free"]/chunk["mem_total"])) * 100.0
        elif "mem_util_pct" in chunk.columns:
            chunk["mem_util_pct"] = pd.to_numeric(chunk["mem_util_pct"], errors="coerce").astype("float32")
        else:
            chunk["mem_util_pct"] = np.nan
        rows.append(chunk[["machine_id","time","mem_util_pct"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["machine_id","time","mem_util_pct"])

# ------------- merge util @ start_time (optional) -------------
def nearest_merge(left: pd.DataFrame, right: pd.DataFrame, on: str, by: str, tol="1min"):
    if left.empty or right.empty: return pd.DataFrame()
    l = left[[by,on]].dropna().copy()
    r = right[[by,on]+[c for c in right.columns if c not in [by,on]]].dropna().copy()
    if l.empty or r.empty: return pd.DataFrame()
    l = l.sort_values([by,on]); r = r.sort_values([by,on])
    return pd.merge_asof(l, r, by=by, left_on=on, right_on=on,
                         tolerance=pd.Timedelta(tol), direction="nearest")

def attach_util_snapshots(jobs_df: pd.DataFrame,
                          gpu_df: pd.DataFrame|None,
                          cpu_df: pd.DataFrame|None,
                          mem_df: pd.DataFrame|None,
                          tol_min: int = 2) -> pd.DataFrame:
    out = jobs_df.copy()
    out["gpu_util_start_mean"] = np.nan
    out["cpu_util_start_mean"] = np.nan
    out["mem_util_start_mean_pct"] = np.nan

    df = out[out["start_time"].notna() & out["machines"].notna()].copy()
    if df.empty: return out
    df["machine_id"] = df["machines"].str.split(";")
    df = df.explode("machine_id").dropna(subset=["machine_id"])

    tol = f"{tol_min}min"

    if gpu_df is not None and not gpu_df.empty:
        g = nearest_merge(df[["jobid","machine_id","start_time"]].rename(columns={"start_time":"time"}),
                          gpu_df, on="time", by="machine_id", tol=tol)
        if not g.empty and "gpu_util_mean" in g.columns:
            g_agg = g.groupby("jobid", as_index=False)["gpu_util_mean"].mean()
            out = out.merge(g_agg.rename(columns={"gpu_util_mean":"gpu_util_start_mean"}), on="jobid", how="left")

    if cpu_df is not None and not cpu_df.empty:
        c = nearest_merge(df[["jobid","machine_id","start_time"]].rename(columns={"start_time":"time"}),
                          cpu_df, on="time", by="machine_id", tol=tol)
        if not c.empty and "cpu_util" in c.columns:
            c_agg = c.groupby("jobid", as_index=False)["cpu_util"].mean()
            out = out.merge(c_agg.rename(columns={"cpu_util":"cpu_util_start_mean"}), on="jobid", how="left")

    if mem_df is not None and not mem_df.empty:
        m = nearest_merge(df[["jobid","machine_id","start_time"]].rename(columns={"start_time":"time"}),
                          mem_df, on="time", by="machine_id", tol=tol)
        if not m.empty and "mem_util_pct" in m.columns:
            m_agg = m.groupby("jobid", as_index=False)["mem_util_pct"].mean()
            out = out.merge(m_agg.rename(columns={"mem_util_pct":"mem_util_start_mean_pct"}), on="jobid", how="left")

    return out

# -------- synergy enrich --------
DEFAULTS = {"gpu_req":1, "gpu_mem_gb":16.0, "cpu_req":8, "ram_req_gb":32.0}
def to_epoch_s(dt: pd.Timestamp) -> float:
    return float(dt.value // 10**9) if (pd.notna(dt)) else np.nan

def enrich_for_synergy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["arrival_ts"] = out["submitted_time"].apply(to_epoch_s)
    out["start_ts"]   = out["start_time"].apply(to_epoch_s)
    out["end_ts"]     = out["end_time"].apply(to_epoch_s)

    out["gpu_req"] = pd.to_numeric(out["num_gpus"], errors="coerce").fillna(DEFAULTS["gpu_req"]).astype(float)
    out["cpu_req"] = pd.to_numeric(out.get("cpu_req_raw"), errors="coerce").fillna(DEFAULTS["cpu_req"])
    out["ram_req_gb"] = pd.to_numeric(out.get("ram_req_gb_raw"), errors="coerce").fillna(DEFAULTS["ram_req_gb"])
    out["gpu_mem_gb"] = pd.to_numeric(out.get("gpu_mem_gb_raw"), errors="coerce").fillna(DEFAULTS["gpu_mem_gb"])

    # Optional bandwidth placeholders (keep NaN unless inferred elsewhere)
    for col in ["pcie_bw_gbps","nic_bw_gbps","disk_bw_mbps"]:
        if col not in out.columns:
            out[col] = np.nan

    prio_map = {"low":1, "medium":3, "med":3, "normal":3, "high":5, "urgent":7}
    out["priority"] = out.get("priority_raw").map(lambda x: prio_map.get(str(x).lower(), 3) if pd.notna(x) else 3)
    out["qos_deadline_s"] = pd.to_numeric(out.get("qos_deadline_s_raw"), errors="coerce").fillna(4*3600).astype(float)
    out["slack_s"] = out["qos_deadline_s"] - (out["wait_s"].fillna(0) + out["duration_s"].fillna(0))

    def pack_vec(row):
        return json.dumps({
            "gpu":int(row["gpu_req"]), "gpu_mem_gb":float(row["gpu_mem_gb"]),
            "cpu":int(row["cpu_req"]), "ram_gb":float(row["ram_req_gb"]),
            "pcie_gbps":float(row.get("pcie_bw_gbps") or 0.0),
            "nic_gbps":float(row.get("nic_bw_gbps") or 0.0),
            "disk_mbps":float(row.get("disk_bw_mbps") or 0.0),
        }, separators=(",", ":"))
    out["multi_resource_vec"] = out.apply(pack_vec, axis=1)
    return out

# ---------------- main ----------------
def main():
    args = parse_args()
    root = Path(args.archive)
    # Allow passing parent folder; prefer trace-data if present.
    if root.is_dir() and (root / "trace-data").exists():
        root = root / "trace-data"

    files = locate_all(root)
    print("[files]"); [print(f"  {k} :", v) for k, v in files.items()]

    jobs_raw = read_job_log_any(files["job_log"])
    if not jobs_raw:
        sys.exit("[fatal] Could not read cluster_job_log.*")

    jobs_df = pd.DataFrame([parse_job_record(r) for r in jobs_raw])

    # Optional util loads (native chunking; small memory)
    gpu_df = load_gpu_util_stream(files["gpu_util"], args.chunksize) if args.use_gpu and files["gpu_util"] else pd.DataFrame()
    cpu_df = load_cpu_util_stream(files["cpu_util"], args.chunksize) if args.use_cpu and files["cpu_util"] else pd.DataFrame()
    mem_df = load_mem_util_stream(files["mem_util"], args.chunksize) if args.use_mem and files["mem_util"] else pd.DataFrame()

    if any([not gpu_df.empty, not cpu_df.empty, not mem_df.empty]):
        jobs_df = attach_util_snapshots(jobs_df, gpu_df, cpu_df, mem_df, tol_min=args.util_merge_tol_min)
    else:
        # Ensure util columns exist (NaN)
        for col in ["gpu_util_start_mean","cpu_util_start_mean","mem_util_start_mean_pct"]:
            if col not in jobs_df.columns:
                jobs_df[col] = np.nan

    # Throughput proxy from dataset fields (gpu-minutes)
    jobs_df["throughput_gpu_min"] = (
        pd.to_numeric(jobs_df["num_gpus"], errors="coerce").fillna(0) *
        (pd.to_numeric(jobs_df["duration_s"], errors="coerce").fillna(0) / 60.0)
    ).replace({np.inf: np.nan})

    # Synergy enrich
    jobs_df = enrich_for_synergy(jobs_df)

    jobs_df.to_csv(args.outcsv, index=False)
    print(f"\n[save] CSV: {args.outcsv}  rows={len(jobs_df)}  cols={len(jobs_df.columns)}")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(jobs_df.head(6))

if __name__ == "__main__":
    sys.exit(main())
