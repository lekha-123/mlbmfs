FIVE DATASET FILES :
| Column           | Description                                       |
| ---------------- | ------------------------------------------------- |
| `machineId`      | Unique ID for each node (`m0`, `m1`, `m29`, etc.) |
| `number of GPUs` | Number of physical GPUs in that machine           |
| `single GPU mem` | Memory per GPU (e.g., 12GB)                       |

| Column       | Description                         |
| ------------ | ----------------------------------- |
| `time`       | Timestamp (minute-level resolution) |
| `machine_id` | Machine identifier (e.g., `m29`)    |
| `cpu_util`   | CPU utilization percentage          |

| Column       | Description                   |
| ------------ | ----------------------------- |
| `time`       | Timestamp                     |
| `machine_id` | Machine identifier            |
| `mem_total`  | Total system memory           |
| `mem_free`   | Free memory at that timestamp |

| Column                    | Description                   |
| ------------------------- | ----------------------------- |
| `time`                    | Timestamp                     |
| `machineId`               | Machine identifier            |
| `gpu0_util` … `gpu7_util` | Utilization % per GPU (0–100) |

| Key                                        | Description                                 |
| ------------------------------------------ | ------------------------------------------- |
| `jobid`                                    | Unique job identifier                       |
| `status`                                   | Whether job completed (`Pass`, `Fail`)      |
| `submitted_time`, `start_time`, `end_time` | Time metrics for scheduling & latency       |
| `detail` → `ip`, `gpus`                    | Which machine(s) and GPU(s) were used       |
| `user`                                     | Tenant ID or project owner                  |
| `vc`                                       | Virtual cluster (logical grouping of users) |
PROCEDURE TO COMPLETE THE PROJECT
MAKE DATASET PREPARATION FIRST :
 MAKE SINGLE CSV WHICH COMBINES ALL INFORMATION NOTE: ITS DOESNT CONTAIN PCIEBANDWIDTH, CONTENTION INPUT, MAKE IT BLANK, WE CAN MAKE IT SYNTHETIC OR CONSTANT VALUES AT END 
PERFORM RR , SRTF, PRIORITY, RESOURCE AWARE SCHEDULING, FCFS AS BASELINES
FIND THROUGHPUT, LATENCY, RESPONSE TIME, WAITING TIME, QOS VIOLATION RATE, JAIN'S FAIRNESS INDEX,MISS RATE AS QOS METRICS 
PERFORM MLBMFS ALGORITHM - WHICH HAS RESOURCE AWARE SCHEDULING, PRIORITY, PREEMPTION, MULTILEVEL QUEUEING, TIME SLICING, FEEDACK TO ADJUST ITSELF ,
FIND QOS METRICS FOR MLBMFS ALGORITHM, AND COMPARE , VALIDATE THE PROPOSED


COMPLETED CODE: 
DO AS IT 
EXTRACT_TRACE.PY - EXTRACT FILE FILES FROM TAR.GZ
PREPROCESSOR.PY - CSV WITHOUT GPU UTIL, MEMORYUTIL, CPUUTIL INPUTS INTO CSV- (NOTE U CAN TAKE IT AS U DONT HAVE MEMORY ISSUES)
BUILD_MLBMFS_INPUT.PY- CONVERTS CSV, FIND SOME DERIVED VALUES FROM INPUT CSV WHICH IS REQUIRED FOR MLBMFS ALGORITHM 
TRAIN_AND_EVAL.PY- TRAINING USING RANDOMFOREST, SVM, REGRESSION

PSEUDOCODE:
Initialize L deques: Q[0], Q[1], ..., Q[L-1]
For every new arrival job j:
    j.level = 0                  # start at top
    j.age_ticks = 0
    enqueue Q[0].append(j)

Main loop (discrete time, quantum dt):
    # 1) Admission: add any arrivals that occurred by now

    # 2) Aging: increment age; optionally promote long-waiting jobs
    for level in 0..L-1:
        for job in Q[level]:
            job.age_ticks += 1
            if job.age_ticks > AGE_PROMOTE_THRESHOLD and level > 0:
                Q[level].remove(job)
                job.level = level - 1
                Q[job.level].append(job)
                job.age_ticks = 0

    # 3) Candidate selection: gather a small set of candidates
    candidates = []
    for level in 0..L-1:
        if Q[level] not empty:
            candidates.append(Q[level][0])   # head of each queue

    if candidates empty:
        continue to next quantum

    # 4) ML scoring / recommendation
    features = [extract_features(job, global_state) for job in candidates]
    scores = M.predict_scores(features)   # higher is better
    best_idx = argmax(scores)
    chosen_job = candidates[best_idx]

    # 5) Determine time slice
    if M.predict_timeslice_available:
        timeslice = clamp(M.predict_timeslice(features[best_idx]), min_q, max_q)
    else:
        # base timeslice scaled by level (less for lower levels)
        timeslice = time_quantum_base * (1 + (L-1 - chosen_job.level) * LEVEL_FACTOR)

    # 6) Run chosen_job for timeslice (transfer min(remaining, bw*timeslice))
    start_transfer(chosen_job, timeslice)

    # 7) Post-service update
    if chosen_job.remaining <= EPS:
        mark job completed
        remove from its queue (already popped)
    else:
        # demote if consumed full timeslice
        if used_full_timeslice:
            chosen_job.level = min(L-1, chosen_job.level + 1)
        append to appropriate queue tail

    # 8) Optional: collect sample (features, action, reward) for online training

    advance time by timeslice
