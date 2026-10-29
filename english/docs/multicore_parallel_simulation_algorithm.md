# Multi-Core Parallel Simulation Execution Algorithm

> **Scope**: Problems 1–3 of "Multi-Core Scheduling for a General-Purpose Neural Network Processor"
> **Input**: The original computational graph, the participant-submitted multi-core plan `node_to_subgraph`/`core_schedules`, and `config.txt`
> **Output**: The global Makespan, per-core Task/subgraph/op timelines, peak on-core storage, data movement volume, and (for Problem 3) Cache statistics
> **Boundary of responsibility**: Topological sorting of a single Task, insertion of cache fill/evict operations, and intra-core op ordering are covered in [Intra-Core Scheduling Algorithm](核内调度算法.md). This document only describes the parallel simulation of multiple cores, Tasks, and shared resources.

## 1. Overall Flow

The evaluator uses discrete-event simulation. It first constructs Tasks and cross-core dependencies from the multi-core plan, then invokes the intra-core scheduling algorithm to obtain each Task's extended graph and deterministic op order, and finally advances all cores in parallel on a unified timeline:

1. Validate `node_to_subgraph` and `core_schedules` to determine which subgraph each op belongs to, which core each subgraph belongs to, and the within-core ordering.
2. Construct Tasks according to the problem type, and insert `COPY_IN`/`COPY_OUT` at the necessary data boundaries.
3. Run intra-core scheduling for each Task to obtain its extended graph, op order, and newly added fill/evict ops.
4. Initialize the four Pipes per core, op states, intra-core storage states, cross-core release events, and the shared bandwidth pools.
5. Repeatedly execute "settle completion events — activate ready ops — issue ops — advance time" until all ops are complete.
6. Aggregate the global Makespan, timelines, storage peaks, data movement volume, and Cache statistics; if the simulation state cannot advance, report an error immediately.

## 2. Task Construction

### 2.1 Problem 1: One Task per Subgraph

Each subgraph independently forms a single Task and is assigned to a core according to `core_schedules`. Only one Task is active on a given core at any moment, and the activation order strictly follows that core's subgraph list.

Within a Task, private L1/UB are used. Any data read from outside the Task is paired with a DDR tensor and a `COPY_IN` inserted at the Task boundary; any data written to other Tasks or to the final outputs of the original graph gets a DDR tensor and a `COPY_OUT`. Therefore, even if adjacent subgraphs are on the same core, they do not retain on-chip data across Tasks.

A Task can only be activated when all of the following conditions are satisfied:

- All of its predecessor Tasks have completed;
- The previous Task on that core has completed;
- At least `task_same_core_wait_cycles` have elapsed since the previous same-core Task completed;
- At least `task_cross_core_wait_cycles` have elapsed since the cross-core predecessor Task completed.

The two wait-cycle values are taken from the `[multicore_scene_a]` section of `config.txt`; the evaluation configuration uses 100 and 1000 cycles, and the code sets no defaults.

### 2.2 Problems 2 and 3: One Merged Task per Core

All subgraphs on the same core are merged into a single Task, preserving the subgraph order specified by `core_schedules`. Only cross-core data edges insert a source-core `COPY_OUT` and a destination-core `COPY_IN`; adjacent same-core subgraphs directly keep on-chip data and their original dependencies.

The destination `COPY_IN` of a cross-core transfer must wait for the source `COPY_OUT` to complete, plus an additional `cross_core_copy_delay_cycles` (taken from the `[multicore_scene_b]` section of `config.txt`; the evaluation configuration uses 500 cycles). This delay only determines the earliest release time of the destination op; it does not occupy a Pipe or a bandwidth pool.

## 3. Intra-Core Op States and Resources

Each op moves through the states `pending`, `ready`, `running`, and `done` in order. The evaluator maintains separate ready queues and executors for `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_M`, and `PIPE_V` on each core; the same Pipe on the same core executes at most one op at a time, while different cores or different Pipes can run in parallel.

An op enters the ready queue only when all of the following hold:

- Its Task is active;
- All intra-core predecessors have completed;
- All cross-core predecessors have completed and any additional synchronization delay has expired;
- If the op produces an on-chip tensor, its allocation turn has arrived and the remaining L1/UB capacity is sufficient.

The ready queue is sorted by the deterministic order given by the intra-core scheduling. An op requests its output tensor before being issued; when an op completes, the remaining reference counts of its input tensors are decremented, and the corresponding on-chip space is freed immediately when the last consumer finishes. If any on-chip tensor is still resident after a Task ends, the simulation state is judged to be erroneous.

## 4. Shared DDR Bandwidth

`COPY_IN` always uses `PIPE_MTE2`, and `COPY_OUT` always uses `PIPE_MTE3`. A COPY is treated as a DDR transfer when either of its endpoints resides in DDR; an on-chip COPY between L1 and UB only occupies the corresponding Pipe and does not enter the DDR bandwidth pool.

All DDR transfers currently in flight across all cores share a single total bandwidth pool, whose total bandwidth `bandwidth` is taken from the `[bandwidth]` section of `config.txt` (the evaluation configuration uses 60 bytes/cycle). Each in-flight transfer maintains a `remaining_work`, measured in the number of cycles the transfer would still need if it exclusively owned the entire DDR bandwidth. If there are currently (n) in-flight transfers, then after an elapsed Δt, the amount of work reduced for each transfer is:

$$
\Delta w=\frac{\Delta t}{n}.
$$

Whenever a new transfer joins or an existing transfer completes, the work completed up to the current time is first settled at the old concurrency level, and then the estimated finish times of all in-flight transfers are recomputed. Sorting the remaining work as (w_1\le\cdots\le w_n), letting (w_0=0), the real time consumed to exhaust the i-th layer of work is:

$$
\Delta t_i=(w_i-w_{i-1})(n-i+1).
$$

Accumulating layer by layer yields the estimated finish time of each transfer; the final timestamp is rounded up to an integer cycle. If multiple transfers have the same remaining work, they finish at the same time and are removed from the bandwidth pool simultaneously. This computation guarantees that all active transfers fairly share the total bandwidth within any interval, and that completed transfers do not continue to occupy bandwidth due to integer rounding.

## 5. The Problem 3 FIFO Cache

Problem 3 adds a read-only Cache on top of DDR. The Cache is keyed by logical tensor id and records resident entries in FIFO order; a hit does not refresh an entry's position in that order.

- Only `COPY_IN` can query and hit the Cache.
- Miss: the op enters the DDR bandwidth pool, and upon completion is written into the Cache following the FIFO rule.
- Hit: the op is isolated from DDR accesses, does not enter the DDR bandwidth pool, and instead enters the shared `CACHE_READ` bandwidth pool.
- Insufficient capacity: entries are evicted one by one starting from the earliest-inserted entry until the new entry fits; a single tensor larger than the total Cache capacity is not cached.

The `DDR` and `CACHE_READ` bandwidth pools are independent of each other; within each pool, settlements follow the fair-sharing rules of Section 4. For each cacheable COPY_IN, the evaluator records the hit status, logical tensor id, and the actual access path, and aggregates the number of accesses, hits, hit bytes, miss bytes, hit rate computed in bytes, FIFO insertion/eviction events, and the final Cache contents.

## 6. Discrete-Event Loop

Let the current simulation time be `now`. Each round performs the following steps:

1. **Settle bandwidth pools**: update the `remaining_work` of every in-flight transfer based on `now-last_update`.
2. **Complete ops**: mark ops whose finish time is not greater than `now` as `done`, release their Pipes, update intra-core tensor references and capacity, and wake up intra-core successors.
3. **Process cross-core events**: after a source `COPY_OUT` completes, compute the expiry time of the additional delay for the destination `COPY_IN`; once it expires, the op is added to readiness evaluation.
4. **Update the Cache**: in Problem 3, after a cacheable COPY completes, insert it into the Cache following the FIFO rule and record the insertion and eviction events.
5. **Activate Tasks**: in Problem 1, activate new Tasks according to the per-core Task order and predecessor completion times; in Problems 2 and 3, the per-core merged Task participates in scheduling from time 0.
6. **Issue ops**: core by core and Pipe by Pipe, issue ops that satisfy dependency, capacity, and executor constraints; bandwidth ops join the corresponding shared pool and their estimated finish times are recomputed immediately.
7. **Advance time**: take the minimum of the estimated completion times of all running ops and the pending cross-core release times as the next time.

The pseudocode is as follows:

```text
now = 0
while unfinished_ops > 0:
    settle_bandwidth_pools(now)
    retire_finished_ops(now)
    release_due_cross_core_events(now)
    activate_ready_tasks(now)
    issue_ready_ops(now)

    candidates = running_end_times + pending_release_times
    if candidates is empty:
        fail("deadlock")
    next_now = min(candidates)
    if next_now <= now:
        fail("time does not advance")
    now = next_now
```

## 7. Termination Conditions and Failure Checks

Once all ops are complete, the global Makespan is the maximum of all op finish times. A successful result must also satisfy:

- Every Task/subgraph/op has complete and consistent start and end times;
- The per-core L1/UB peak does not exceed the configured capacity, and the final occupancy is zero;
- All cross-core dependencies and within-core orderings are satisfied;
- All bandwidth pools are empty, and in Problem 3 the Cache state is consistent with the recorded events;
- The data movement volume can be decomposed into COPYs from the original graph, COPYs newly added by graph cutting, and COPYs newly added by intra-core fill/evict.

Any of the following conditions causes an immediate failure, without returning a partial Makespan:

- There are unfinished ops, but no running ops or pending release events;
- The next event time is not greater than the current time;
- The maximum number of iterations is exceeded;
- The op, subgraph, or core mapping is incomplete, subgraph dependencies form a cycle, or the within-core ordering violates dependencies;
- Intra-core storage exceeds capacity, or tensors remain unreleased after a Task completes.

## 8. Outputs and Implementation Mapping

The evaluator outputs a result JSON, a Perfetto Trace JSON, and a concise log. The result JSON stores the full numeric values, the Trace is for viewing the multi-core parallel timeline, and the log is for quickly checking the configuration, Makespan, data movement volume, intra-core storage peaks, and Cache events.

The current implementation maps to the following scripts:

- `singlecore_evaluate.py`: standalone single-core baseline evaluation; does not read the multi-core plan.
- `multicore_cut_evaluate_problem_1.py`: Problem 1, one Task per subgraph.
- `multicore_cut_evaluate_problem_2.py`: Problem 2, one merged Task per core.
- `multicore_cut_evaluate_problem_3.py`: Problem 3, read-only FIFO Cache.

## 9. Configuration File

All parameters used by the multi-core simulation come from `config.txt`, with no defaults set in the code; if a required section or key is missing, the evaluator reports an error and exits.

| Section | Keys | Purpose | Evaluation configuration |
|---|---|---|---|
| `[capacity]` | `L1` `UB` | Per-core on-chip capacity (bytes) | 524288 / 131072 |
| `[bandwidth]` | `bandwidth` | Total DDR bandwidth (bytes/cycle) | 60 |
| `[multicore_scene_a]` | `task_cross_core_wait_cycles` `task_same_core_wait_cycles` | Task waits for Problem 1 | 1000 / 100 |
| `[multicore_scene_b]` | `cross_core_copy_delay_cycles` | Cross-core COPY wait for Problems 2/3 | 500 |
| `[problem_3]` | `cache_capacity_bytes` `cache_bandwidth_bytes_per_cycle` | Cache capacity and hit bandwidth for Problem 3 | 1048576 / 250 |

Only the five sections listed in the table above are accepted; an unknown section (e.g., a typo or the deprecated `[pipe_capacity]`) raises `unknown section` and is not silently ignored. The values of `L1`/`UB` are in bytes, consistent with `tensors[].size` in the computational graph. When `--config` is not specified, the evaluator looks for "the directory containing the computational graph/config.txt" and reports an error and exits if the file is not found.

## 10. Interface with Cost Estimation

The intra-core scheduling results provide each subgraph's fill/evict transfer volume and a reference exclusive execution time; the multi-core simulation further accounts for Task boundaries, cross-core dependencies, shared DDR/Cache bandwidth, and core parallelism to produce the final Makespan. If used to train a fast cost-prediction model, structural features such as node count, edge count, tensor size distribution, and Pipe distribution can serve as inputs, with the newly added transfer volume and time penalties as supervision targets.