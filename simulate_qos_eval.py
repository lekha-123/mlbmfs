#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_qos_eval.py
Phase 3 evaluator for QoS scheduling on multi-tenant (CPU/GPU/PCIe) systems.

Policies:
- RoundRobin (RR)
- SRTF (preemptive)
- MLBMFS (advanced): MLFQ + SRTF-in-level + congestion-aware quanta + aging
  (This file includes surgical upgrades:
     * slowdown/deadline-aware promotions,
     * stronger congestion-aware quanta,
     * prediction-confident initial levels,
     * SRTF tie-break with PCIe bias.)
- PQ-SemQoS+ (reference): SLA-aware priority queues (EDF/SRTF/DRR + tokens)

CSV schema (must include):
  arrival_ms, compute_ms, tenant_id
Optional columns:
  DecisionTree_pred, RandomForest_pred, SVM_pred
  xfer_bytes, rw ("read"/"write"), qos_class ("gold"/"silver"/"bronze")

Typical run:
  python simulate_qos_eval.py \
    --input predictions_mlbmfs.csv \
    --output results_compare.csv \
    --pick-busiest-min 240 \
    --compress-gaps-ms 30000 \
    --time-scale 0.02 \
    --parallel 4 \
    --quantum-ms 6500 \
    --aging-ms 1800 \
    --verbose
"""

from __future__ import annotations
import argparse, math, warnings, copy
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import deque, defaultdict
import heapq
import numpy as np
import pandas as pd

# =============================
# Utilities / preprocessing
# =============================

def log(msg: str, v: bool):
    if v:
        print(msg, flush=True)

def pick_busiest_window(arrivals_sorted: np.ndarray, minutes: int) -> Tuple[int, int]:
    """Return [i, j) indices covering the densest M-minute window by arrival count."""
    window = minutes * 60_000
    best = (0, 0); best_cnt = 0
    j = 0
    for i in range(len(arrivals_sorted)):
        start = arrivals_sorted[i]
        while j < len(arrivals_sorted) and arrivals_sorted[j] < start + window:
            j += 1
        cnt = j - i
        if cnt > best_cnt:
            best_cnt = cnt
            best = (i, j)
    return best

def apply_dense_slice(df: pd.DataFrame, minutes: Optional[int]) -> pd.DataFrame:
    if not minutes:
        return df
    a = df["arrival_ms"].to_numpy()
    idx = np.argsort(a)
    a_sorted = a[idx]
    i, j = pick_busiest_window(a_sorted, minutes)
    return df.iloc[idx[i:j]].copy().reset_index(drop=True)

def compress_gaps(a_sorted_rel: np.ndarray, cap_ms: Optional[int]) -> np.ndarray:
    """Cap inter-arrival gaps (operates on *relative* timeline)."""
    if not cap_ms or len(a_sorted_rel) == 0:
        return a_sorted_rel
    out = np.zeros_like(a_sorted_rel, dtype=float)
    out[0] = 0.0
    for k in range(1, len(a_sorted_rel)):
        gap = a_sorted_rel[k] - a_sorted_rel[k-1]
        out[k] = out[k-1] + min(float(gap), float(cap_ms))
    return out

def rebuild_arrivals(df: pd.DataFrame, cap_ms: Optional[int], scale: float) -> pd.Series:
    """Rebuild arrivals after optional gap capping + scaling (relative to min)."""
    a = df["arrival_ms"].to_numpy()
    idx = np.argsort(a)
    a_sorted = a[idx]
    rel = a_sorted - a_sorted[0]
    if cap_ms:
        rel = compress_gaps(rel, cap_ms)
    rel = (rel * float(scale)).astype(float)
    out = np.zeros_like(rel)
    out[idx] = rel
    return pd.Series(out, index=df.index, dtype=float)

def scale_compute(df: pd.DataFrame, scale: float) -> pd.Series:
    return (df["compute_ms"].astype(float) * float(scale)).astype(float)

def jain_index(values: List[float]) -> float:
    """Jain's fairness index: (sum x)^2 / (n * sum x^2)."""
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    if np.all(arr == 0):
        return 1.0
    s = arr.sum()
    s2 = (arr ** 2).sum()
    n = float(arr.size)
    return float((s * s) / (n * s2)) if s2 > 0 else float("nan")

# =============================
# Job model
# =============================

@dataclass
class Job:
    jid: int
    tenant: str
    arrival: float
    remain: float        # remaining service (compute + pcie)
    compute: float
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    level_hint: Optional[int] = None

    # QoS & PCIe fields
    qos_class: str = "silver"        # "gold" | "silver" | "bronze"
    deadline_ms: Optional[float] = None
    xfer_bytes: int = 0
    rw: str = "read"                 # "read" | "write"
    pcie_remain: float = 0.0         # remaining PCIe time (ms)

    def predicted_remaining(self) -> float:
        return float(max(self.remain, 0.0))

@dataclass(order=True)
class SRJob:
    remain: float
    arrival: float
    seq: int
    ref: Job = field(compare=False)

# =============================
# Baseline Schedulers
# =============================

def simulate_rr(jobs: List[Job], parallel: int = 4, quantum_ms: float = 10_000.0, verbose: bool=False) -> List[Job]:
    """Round-Robin with K workers, FCFS ready-queue, fixed quantum."""
    t = 0.0
    jobs_sorted = sorted(jobs, key=lambda j: j.arrival)
    ready = deque()
    workers = [None] * parallel          # holds Job or None
    next_slice_exp = [math.inf] * parallel
    i = 0
    done = 0
    total = len(jobs)

    def push_arrivals(up_to: float):
        nonlocal i
        while i < total and jobs_sorted[i].arrival <= up_to + 1e-9:
            ready.append(jobs_sorted[i]); i += 1

    if total:
        t = jobs_sorted[0].arrival

    step_print_next = t + 5_000.0  # log every 5s of simulated time

    while done < total:
        push_arrivals(t)
        # assign idle workers
        for w in range(parallel):
            if workers[w] is None and ready:
                j = ready.popleft()
                j.start_time = j.start_time or t
                workers[w] = j
                next_slice_exp[w] = t + quantum_ms

        # next interesting time
        next_times = [jobs_sorted[i].arrival] if i < total else []
        for w in range(parallel):
            if workers[w] is not None:
                j = workers[w]
                next_times.append(min(t + j.remain, next_slice_exp[w]))
        if not next_times: break
        t_next = min(next_times)
        dt = max(0.0, t_next - t)

        # run
        for w in range(parallel):
            if workers[w] is not None:
                j = workers[w]
                j.remain -= dt
                if j.remain < 0: j.remain = 0.0
        t = t_next

        # periodic progress
        if verbose and t >= step_print_next - 1e-9:
            log(f"[RR] t={t:.3f} ready={len(ready)} done={done}/{total}", True)
            step_print_next = t + 5_000.0

        # completions
        for w in range(parallel):
            j = workers[w]
            if j is not None and j.remain <= 1e-9:
                j.finish_time = t
                workers[w] = None
                next_slice_exp[w] = math.inf
                done += 1

        # quantum expirations
        for w in range(parallel):
            j = workers[w]
            if j is not None and t >= next_slice_exp[w] - 1e-9 and j.remain > 1e-9:
                workers[w] = None
                next_slice_exp[w] = math.inf
                ready.append(j)

        push_arrivals(t)
        if not ready and all(w is None for w in workers) and i < total:
            t = jobs_sorted[i].arrival

    return jobs

def simulate_srtf(jobs: List[Job], parallel: int = 4, verbose: bool=False) -> List[Job]:
    """K-way preemptive SRTF using a min-heap by remaining time."""
    t = 0.0
    jobs_sorted = sorted(jobs, key=lambda j: j.arrival)
    total = len(jobs)
    i = 0
    seq = 0
    ready: List[SRJob] = []
    workers: List[Optional[Job]] = [None] * parallel

    def push_arrivals(up_to: float):
        nonlocal i, seq
        while i < total and jobs_sorted[i].arrival <= up_to + 1e-9:
            j = jobs_sorted[i]
            heapq.heappush(ready, SRJob(remain=j.remain, arrival=j.arrival, seq=seq, ref=j))
            seq += 1; i += 1

    if total:
        t = jobs_sorted[0].arrival

    step_print_next = t + 5_000.0

    while True:
        push_arrivals(t)

        # fill workers
        for w in range(parallel):
            if workers[w] is None and ready:
                sj = heapq.heappop(ready).ref
                sj.start_time = sj.start_time or t
                workers[w] = sj

        next_times = []
        if i < total:
            next_times.append(jobs_sorted[i].arrival)
        for w in range(parallel):
            if workers[w] is not None:
                next_times.append(t + workers[w].remain)
        if not next_times: break
        t_next = min(next_times)
        dt = max(0.0, t_next - t)

        # run dt
        for w in range(parallel):
            if workers[w] is not None:
                workers[w].remain -= dt
                if workers[w].remain < 0: workers[w].remain = 0.0
        t = t_next

        if verbose and t >= step_print_next - 1e-9:
            log(f"[SRTF] t={t:.3f} ready={len(ready)}", True)
            step_print_next = t + 5_000.0

        # complete
        for w in range(parallel):
            if workers[w] is not None and workers[w].remain <= 1e-9:
                workers[w].finish_time = t
                workers[w] = None

        # preempt with arrivals: move running back into heap
        push_arrivals(t)
        running = []
        for w in range(parallel):
            if workers[w] is not None:
                running.append(workers[w])
                workers[w] = None
        for j in running:
            heapq.heappush(ready, SRJob(remain=j.remain, arrival=j.arrival, seq=seq, ref=j))
            seq += 1

        # reassign best K
        for w in range(parallel):
            if workers[w] is None and ready:
                sj = heapq.heappop(ready).ref
                sj.start_time = sj.start_time or t
                workers[w] = sj

        if all(w is None for w in workers) and i < total:
            t = jobs_sorted[i].arrival

    return jobs

# =============================
# MLBMFS (advanced MLFQ+SRTF)
# =============================

def simulate_mlbmfs_advanced(
    jobs: List[Job],
    parallel: int = 4,
    aging_ms: float = 5000.0,
    base_quanta: Tuple[float, float, float, float] = (4000.0, 10000.0, 25000.0, 60000.0),
    verbose: bool=False,
) -> List[Job]:
    """MLFQ + SRTF-in-level + congestion-aware quanta + slowdown/deadline-aware promotions."""
    def effective_quanta(q: Tuple[float, ...], qlens: List[int]) -> Tuple[float, ...]:
        # Stronger shrink under heavy load (surgical change B)
        load = int(sum(qlens))
        if load >= 64:   # very busy
            return tuple(max(x * 0.25, 1500.0) for x in q)
        if load >= 32:   # moderately busy
            return tuple(max(x * 0.55, 2500.0) for x in q)
        return q

    t = 0.0
    jobs_sorted = sorted(jobs, key=lambda j: j.arrival)
    total = len(jobs)
    i = 0

    Q: List[deque] = [deque() for _ in range(4)]
    wait_since: Dict[int, float] = {}          # jid -> last enqueued time
    workers: List[Optional[Tuple[Job, int, float]]] = [None] * parallel  # (job, lvl, expire_time)

    def initial_level(j: Job) -> int:
        if j.level_hint is not None:
            return max(0, min(3, int(j.level_hint)))
        c = j.compute
        if c <= 5_000.0: return 0
        if c <= 20_000.0: return 1
        if c <= 60_000.0: return 2
        return 3

    def push_arrivals(up_to: float):
        nonlocal i
        while i < total and jobs_sorted[i].arrival <= up_to + 1e-9:
            j = jobs_sorted[i]
            lvl = initial_level(j)
            Q[lvl].append(j)
            wait_since[j.jid] = up_to
            i += 1

    def promote_aging(now: float):
        for lvl in range(1, len(Q)):
            keep = deque()
            while Q[lvl]:
                j = Q[lvl].popleft()
                if now - wait_since.get(j.jid, now) >= aging_ms:
                    Q[lvl - 1].append(j)
                    wait_since[j.jid] = now
                else:
                    keep.append(j)
            Q[lvl] = keep

    if total:
        t = jobs_sorted[0].arrival

    done = 0
    step_print_next = t + 5_000.0

    while done < total:
        push_arrivals(t)
        promote_aging(t)

        quanta_now = effective_quanta(base_quanta, [len(q) for q in Q])

        # assign idle workers: highest non-empty level, SRTF-with-PCIe-bias within that level
        for w in range(parallel):
            if workers[w] is None:
                lvl_pick = None
                for lvl in range(4):
                    if Q[lvl]:
                        lvl_pick = lvl
                        break
                if lvl_pick is not None:
                    # SRTF tie-break with a small PCIe bias (surgical change D)
                    def sr_key(cand: Job):
                        return cand.predicted_remaining() + 0.05 * cand.pcie_remain

                    best_i, best_j = 0, None
                    for idx, cand in enumerate(Q[lvl_pick]):
                        if best_j is None or sr_key(cand) < sr_key(best_j):
                            best_i, best_j = idx, cand
                    if best_i == 0:
                        j = Q[lvl_pick].popleft()
                    else:
                        Q[lvl_pick].rotate(-best_i)
                        j = Q[lvl_pick].popleft()
                        Q[lvl_pick].rotate(best_i)
                    j.start_time = j.start_time or t
                    workers[w] = (j, lvl_pick, t + quanta_now[lvl_pick])

        # next time: next arrival, completion, or quantum expiry
        next_times = [jobs_sorted[i].arrival] if i < total else []
        for w in range(parallel):
            if workers[w] is not None:
                j, lvl, exp = workers[w]
                next_times.append(min(t + j.remain, exp))
        if not next_times: break
        t_next = min(next_times)
        dt = max(0.0, t_next - t)

        # run dt
        for w in range(parallel):
            if workers[w] is not None:
                j, lvl, exp = workers[w]
                j.remain -= dt
                if j.remain < 0: j.remain = 0.0
        t = t_next

        if verbose and t >= step_print_next - 1e-9:
            qlen_total = sum(len(q) for q in Q)
            log(f"[MLBMFS] t={t:.3f} Qlens={[len(q) for q in Q]} totalQ={qlen_total}", True)
            step_print_next = t + 5_000.0

        # completions
        for w in range(parallel):
            if workers[w] is not None:
                j, lvl, exp = workers[w]
                if j.remain <= 1e-9:
                    j.finish_time = t
                    workers[w] = None
                    done += 1

        # quantum expirations → demote unless slowdown-rescue triggers
        for w in range(parallel):
            if workers[w] is not None:
                j, lvl, exp = workers[w]
                if t >= exp - 1e-9 and j.remain > 1e-9:
                    # --- Slowdown/Deadline-aware promotion (surgical change A) ---
                    waited = t - j.arrival
                    service_budget = max(j.compute + j.pcie_remain, 1.0)
                    est_slowdown = (waited + j.remain) / service_budget

                    # default demotion target
                    demote_lvl = min(lvl + 1, 3)
                    new_lvl = demote_lvl

                    # rescue if trending to breach (threshold tuned for your dense window)
                    if est_slowdown > 5.0 and lvl > 0:
                        new_lvl = lvl - 1  # promote upward by 1 level

                    workers[w] = None
                    Q[new_lvl].append(j)
                    wait_since[j.jid] = t

        push_arrivals(t)
        promote_aging(t)

        if all(w is None for w in workers) and i < total and all(len(q) == 0 for q in Q):
            t = jobs_sorted[i].arrival

    return jobs

# =============================
# Token Bucket (for PQ-SemQoS+ bronze)
# =============================

class TokenBucket:
    def __init__(self, rate_mb_s: float, burst_mb: float):
        self.rate = rate_mb_s * 1024 * 1024    # bytes/s
        self.burst = burst_mb * 1024 * 1024
        self.bytes = self.burst
        self.last_t = 0.0

    def refill_to(self, now_sec: float):
        dt = max(0.0, now_sec - self.last_t)
        self.bytes = min(self.burst, self.bytes + self.rate * dt)
        self.last_t = now_sec

    def consume(self, amount_bytes: int) -> bool:
        if self.bytes >= amount_bytes:
            self.bytes -= amount_bytes
            return True
        return False

# =============================
# PQ-SemQoS+ (reference)
# =============================

def simulate_pq_semqos_plus(
    jobs: List[Job],
    parallel: int = 4,
    aging_ms: float = 3000.0,
    base_quanta: Tuple[float, float, float, float] = (4000.0, 10000.0, 25000.0, 60000.0),
    bronze_quantum_ms: float = 8000.0,
    verbose: bool=False,
) -> List[Job]:
    """
    Priority queues with:
      - Gold: EDF primary, SRTF tie-break (preemptive)
      - Silver: SRTF (preemptive)
      - Bronze: DRR + per-tenant token buckets (weighted fairness)
    Congestion-aware quantum shrink + slowdown-aware promotions.
    """
    # queues by class
    Q_gold, Q_silver = [], []
    Q_bronze = defaultdict(deque)
    bronze_order = deque()

    tenants = {j.tenant for j in jobs}
    buckets = {tid: TokenBucket(100.0, 512.0) for tid in tenants}

    t = min(j.arrival for j in jobs) if jobs else 0.0
    jobs_sorted = sorted(jobs, key=lambda j: j.arrival)
    i, total, done = 0, len(jobs), 0
    workers: List[Optional[Job]] = [None]*parallel
    exp: List[float] = [math.inf]*parallel

    def gold_key(j: Job):
        return (j.deadline_ms if j.deadline_ms is not None else float("inf"),
                j.predicted_remaining(), j.arrival, j.jid)

    def silver_key(j: Job):
        return (j.predicted_remaining(), j.arrival, j.jid)

    def refill_buckets(now_ms: float):
        now_sec = now_ms / 1000.0
        for b in buckets.values():
            b.refill_to(now_sec)

    def enqueue(j: Job):
        if j.qos_class == "gold":
            heapq.heappush(Q_gold, (gold_key(j), j))
        elif j.qos_class == "silver":
            heapq.heappush(Q_silver, (silver_key(j), j))
        else:
            Q_bronze[j.tenant].append(j)
            if j.tenant not in bronze_order:
                bronze_order.append(j.tenant)

    def push_arrivals(up_to: float):
        nonlocal i
        while i < total and jobs_sorted[i].arrival <= up_to + 1e-9:
            enqueue(jobs_sorted[i]); i += 1

    def effective_quanta(q, qlen_total: int):
        if qlen_total >= 64:  return tuple(max(x*0.35, 2000.0) for x in q)
        if qlen_total >= 32:  return tuple(max(x*0.60, 3000.0) for x in q)
        return q

    push_arrivals(t)

    step_print_next = t + 5_000.0

    while done < total:
        refill_buckets(t)

        # assign idle workers with strict class priority
        for w in range(parallel):
            if workers[w] is not None:
                continue
            picked = None
            if Q_gold:
                _, picked = heapq.heappop(Q_gold)
            elif Q_silver:
                _, picked = heapq.heappop(Q_silver)
            else:
                if bronze_order:
                    rot = len(bronze_order)
                    while rot > 0 and not picked:
                        tid = bronze_order[0]
                        q = Q_bronze[tid]
                        if not q:
                            bronze_order.popleft(); bronze_order.append(tid); rot -= 1
                            continue
                        j = q[0]
                        need = j.xfer_bytes
                        if buckets[tid].consume(need):
                            picked = q.popleft()
                        else:
                            bronze_order.popleft(); bronze_order.append(tid); rot -= 1
            if picked is not None:
                workers[w] = picked
                picked.start_time = picked.start_time or t
                qlen_total = len(Q_gold)+len(Q_silver)+sum(len(d) for d in Q_bronze.values())
                qtuple = effective_quanta(base_quanta, qlen_total)
                if picked.qos_class == "gold":
                    exp[w] = t + qtuple[0]
                elif picked.qos_class == "silver":
                    exp[w] = t + qtuple[1]
                else:
                    exp[w] = t + bronze_quantum_ms

        next_times = []
        if i < total:
            next_times.append(jobs_sorted[i].arrival)
        for w, j in enumerate(workers):
            if j is not None:
                next_times.append(t + j.remain)
                next_times.append(exp[w])
        if not next_times: break
        t_next = min(next_times)
        dt = max(0.0, t_next - t)

        for w, j in enumerate(workers):
            if j is not None:
                j.remain -= dt
                if j.remain < 0: j.remain = 0.0
        t = t_next

        if verbose and t >= step_print_next - 1e-9:
            log(f"[PQ] t={t:.3f}", True)
            step_print_next = t + 5_000.0

        for w, j in enumerate(workers):
            if j is not None and j.remain <= 1e-9:
                j.finish_time = t
                workers[w] = None
                exp[w] = math.inf
                done += 1

        for w, j in enumerate(workers):
            if j is not None and t >= exp[w] - 1e-9 and j.remain > 1e-9:
                workers[w] = None; exp[w] = math.inf
                if j.deadline_ms is not None:
                    waited = t - j.arrival
                    budget = max(j.deadline_ms - j.arrival, 1.0)
                    frac = waited / budget
                    if j.qos_class == "silver" and frac > 0.9:
                        j.qos_class = "gold"
                    elif j.qos_class == "bronze" and frac > 0.8:
                        j.qos_class = "silver"
                enqueue(j)

        push_arrivals(t)

        if all(x is None for x in workers) and i < total and not (Q_gold or Q_silver or any(Q_bronze.values())):
            t = jobs_sorted[i].arrival

    return jobs

# =============================
# Metrics (latency + slowdown QoS)
# =============================

def metrics(jobs: List[Job]) -> Dict[str, float]:
    arr = np.array([j.arrival for j in jobs], dtype=float)
    fin = np.array([j.finish_time for j in jobs], dtype=float)
    lat = fin - arr
    svc = np.array([max(j.compute + j.pcie_remain, 1e-3) for j in jobs], dtype=float)  # avoid /0
    slowdown = lat / svc

    makespan = fin.max() - arr.min() if len(fin) else float("nan")
    thr_jps = len(jobs) / (makespan / 1000.0) if makespan and makespan > 0 else float("nan")

    p50 = float(np.percentile(lat, 50)) if lat.size else float("nan")
    p95 = float(np.percentile(lat, 95)) if lat.size else float("nan")
    mean = float(np.mean(lat)) if lat.size else float("nan")

    sd_mean = float(np.mean(slowdown)) if slowdown.size else float("nan")
    sd_p95 = float(np.percentile(slowdown, 95)) if slowdown.size else float("nan")
    sla_violation = float(np.mean(slowdown > 4.0)) if slowdown.size else float("nan")  # SLA: 4x

    per_tenant = defaultdict(list)
    for j, s in zip(jobs, slowdown):
        per_tenant[j.tenant].append(float(s))
    tenant_slowdowns = [float(np.mean(v)) for v in per_tenant.values()]
    fairness = jain_index(tenant_slowdowns)

    return dict(
        throughput_jps=thr_jps,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_mean_ms=mean,
        slowdown_mean=sd_mean,
        slowdown_p95=sd_p95,
        qos_violation_rate=sla_violation,
        fairness_jain=fairness,
    )

# =============================
# Entry
# =============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV path (predictions or inputs)")
    ap.add_argument("--output", default="results_compare.csv", help="Results CSV path")
    ap.add_argument("--sample", type=int, help="Optional random sample size for speed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-scale", type=float, default=0.01, help="Scale for inter-arrivals and compute_ms")
    ap.add_argument("--pick-busiest-min", type=int, help="Select densest M-minute window by arrivals")
    ap.add_argument("--compress-gaps-ms", type=int, help="Cap inter-arrival gaps (ms) after sorting")
    ap.add_argument("--parallel", type=int, default=4, help="Number of workers")
    ap.add_argument("--quantum-ms", type=float, default=10000.0, help="Base quantum for MLBMFS (level-1); others derived")
    ap.add_argument("--aging-ms", type=float, default=5000.0, help="Aging threshold to promote jobs (ms)")
    ap.add_argument("--pcie-gbps", type=float, default=12.0, help="Effective PCIe GB/s for xfer-time modeling")
    ap.add_argument("--default-xfer-mb", type=float, default=64.0, help="Default transfer size if missing (MB)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    VERBOSE = args.verbose

    log(f"📦 Loading CSV: {args.input}", VERBOSE)
    df = pd.read_csv(args.input)
    required = {"arrival_ms", "compute_ms", "tenant_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    # Slice / sample / normalize
    df = df.sort_values("arrival_ms").reset_index(drop=True)
    if args.pick_busiest_min:
        log(f"🔎 Picking densest window: {args.pick_busiest_min} min", VERBOSE)
        df = apply_dense_slice(df, args.pick_busiest_min)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed).sort_values("arrival_ms").reset_index(drop=True)

    # Rebuild timeline & compute scaling
    df["arrival_adj"] = rebuild_arrivals(df, args.compress_gaps_ms, args.time_scale)
    df["compute_adj"] = scale_compute(df, args.time_scale)

    # Prediction → level_hint (surgical change C): 15/40/70 percentiles
    pred_cols = [c for c in ["DecisionTree_pred", "RandomForest_pred", "SVM_pred"] if c in df.columns]
    level_hint = None
    if pred_cols:
        preds_raw = df[pred_cols].astype(float).median(axis=1).to_numpy()
        p15, p40, p70 = np.quantile(preds_raw, [0.15, 0.40, 0.70])
        lh = np.zeros_like(preds_raw, dtype=int)
        lh[preds_raw > p15] = 1
        lh[preds_raw > p40] = 2
        lh[preds_raw > p70] = 3
        level_hint = lh
    else:
        warnings.warn("No prediction columns found; MLBMFS will bucket by compute size only.")

    # ---- PCIe modeling params ----
    PCIE_GBPS = float(args.pcie_gbps)
    RW_PENALTY = {"read": 1.0, "write": 1.15}
    DEFAULT_XFER_MB = float(args.default_xfer_mb)
    TENANT_CLASS_DEFAULT = {"1":"gold","2":"silver","3":"silver"}  # fallback map

    def xfer_ms(bytes_, rw="read"):
        gb = float(bytes_) / (1024**3)
        sec = gb / PCIE_GBPS * RW_PENALTY.get(rw, 1.0)
        return sec * 1000.0

    # Jobs (position-based to avoid index/iloc confusion)
    jobs0: List[Job] = []
    for pos, row in enumerate(df.itertuples(index=False)):
        lvl = int(level_hint[pos]) if level_hint is not None else None

        size_bytes = getattr(row, "xfer_bytes", None)
        if size_bytes is None or (isinstance(size_bytes, float) and math.isnan(size_bytes)):
            size_bytes = DEFAULT_XFER_MB * 1024 * 1024

        rw = getattr(row, "rw", "read")
        qclass = getattr(row, "qos_class", None)
        if qclass is None:
            qclass = TENANT_CLASS_DEFAULT.get(str(getattr(row, "tenant_id")), "silver")

        base_service = float(getattr(row, "compute_adj")) + xfer_ms(size_bytes, rw)
        sla_mult = {"gold": 3.5, "silver": 5.0, "bronze": 8.0}.get(qclass, 5.0)
        deadline = float(getattr(row, "arrival_adj")) + sla_mult * base_service

        jobs0.append(Job(
            jid=pos,
            tenant=str(getattr(row, "tenant_id")),
            arrival=float(getattr(row, "arrival_adj")),
            remain=base_service,                    # compute + pcie service
            compute=float(getattr(row, "compute_adj")),
            level_hint=lvl,
            qos_class=str(qclass),
            deadline_ms=float(deadline),
            xfer_bytes=int(size_bytes),
            rw=str(rw),
            pcie_remain=xfer_ms(size_bytes, rw)
        ))

    # Derive MLBMFS quanta from --quantum-ms (ratios similar to defaults)
    q1 = float(args.quantum_ms)
    quanta = (max(q1 * 0.4, 2000.0), q1, q1 * 2.5, q1 * 6.0)

    log(f"✅ Prepared jobs: {len(jobs0)}  (time-scale={args.time_scale})", VERBOSE)
    log(f"▶️  Running policies with parallel={args.parallel}", VERBOSE)

    results = []

    # Round Robin
    log("   [RoundRobin] starting …", VERBOSE)
    jobs_fin = simulate_rr(copy.deepcopy(jobs0), parallel=args.parallel, quantum_ms=q1, verbose=VERBOSE)
    results.append(dict(policy="RoundRobin", **metrics(jobs_fin)))

    # SRTF
    log("   [SRTF] starting …", VERBOSE)
    jobs_fin = simulate_srtf(copy.deepcopy(jobs0), parallel=args.parallel, verbose=VERBOSE)
    results.append(dict(policy="SRTF", **metrics(jobs_fin)))

    # MLBMFS (with surgical upgrades)
    log("   [MLBMFS] starting …", VERBOSE)
    jobs_fin = simulate_mlbmfs_advanced(copy.deepcopy(jobs0), parallel=args.parallel, aging_ms=float(args.aging_ms), base_quanta=quanta, verbose=VERBOSE)
    results.append(dict(policy="MLBMFS", **metrics(jobs_fin)))

    # PQ-SemQoS+ (reference)
    log("   [PQ-SemQoS+] starting …", VERBOSE)
    jobs_fin = simulate_pq_semqos_plus(copy.deepcopy(jobs0), parallel=args.parallel, aging_ms=max(2000.0, float(args.aging_ms)*0.6), base_quanta=quanta, verbose=VERBOSE)
    results.append(dict(policy="PQ-SemQoS+", **metrics(jobs_fin)))

    cols = [
        "policy",
        "throughput_jps",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_mean_ms",
        "slowdown_mean",
        "slowdown_p95",
        "qos_violation_rate",
        "fairness_jain",
    ]
    out_df = pd.DataFrame(results)[cols]
    out_df.to_csv(args.output, index=False)
    print("✅ Saved comparison results →", args.output)
    print(out_df.to_string(index=False))

if __name__ == "__main__":
    main()
