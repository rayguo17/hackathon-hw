# Multi-Core Scheduling Problem: Final Evaluation Code

Evaluation code for Problem A of the "Huawei Cup" Mathematical Modeling Contest

## The Four Most Common Commands

Run from the project root directory:

```powershell
python code/stub_multicore_cut_and_schedule.py <computational_graph.json> -n 4
python code/multicore_cut_evaluate_problem_1.py <computational_graph.json> <plan.json> --config data/config.txt
python code/multicore_cut_evaluate_problem_2.py <computational_graph.json> <plan.json> --config data/config.txt
python code/multicore_cut_evaluate_problem_3.py <computational_graph.json> <plan.json> --config data/config.txt
```

When `<plan.json>` is omitted, the script reads `<computational_graph_file_name>_multicore_res.json` from the same directory as the computational graph. Each evaluator outputs a result JSON, a brief log, and a Perfetto Trace. `stub_multicore_cut_and_schedule.py` only demonstrates the plan format; it is not a baseline algorithm.

## The Complete Pipeline from Input to Makespan

1. **Read the contestant's plan.** `derive_multicore_plan` verifies that each computational operation belongs to exactly one subgraph, and that each subgraph is assigned to exactly one core. It also validates the original graph format, unique IDs, valid endpoints, non-negative cycles/size, and that the original full graph is acyclic (including COPY nodes and dependencies within the same subgraph).
2. **Form Tasks.** Problem 1 treats each subgraph as one Task; Problems 2 and 3 merge subgraphs on the same core into a single Task. Data edges crossing Tasks or cores generate COPY operations.
3. **Run intra-core scheduling.** `schedule_step1` generates the topological visitation order; `schedule_step2` inserts swap-out and swap-in operations when capacity is insufficient; `schedule_step3` completes the out-of-order arrangement and outputs a deterministic order for each Pipe.
4. **Compensate for address dependencies.** Step3 does not compute concrete addresses; instead it abstracts L1/UB capacity into divisible virtual byte credits. Before a credit is reused for a new output, all readers of the old tensor must have completed; these WAR/WAW relationships are written into the execution graph as direct op→op edges. Credits can be split and merged, so no address fragmentation occurs and no memory compaction algorithm is needed for the contest.
5. **Event simulation.** The three evaluators directly load Step3's execution graph and per-Pipe order. Before starting the event loop, Problem 1 checks the combined graph of data dependencies and same-core Task ordering; Problems 2/3 check the global operation graph composed of local execution dependencies, Pipe FIFOs, and cross-core COPYs, rejecting wait cycles. The simulator finishes completed operations, advances Pipes, releases newly satisfied true data dependencies and memory-reuse dependencies, dispatches the head-of-queue operation on each Pipe, and then jumps to the nearest next event time.
6. **Collect statistics.** The latest finish time is the makespan; the output also includes per-core timelines, on-chip memory peaks, COPY traffic, and Problem 3's Cache statistics.

## Contestant Entry Points and Error Handling

Contestants invoke the three problem evaluators, `singlecore_evaluate.py`, or the plan example generator. Invalid input returns a non-zero exit code and prints `[EVALUATION ERROR]` to stderr, without generating new success results/traces. If the plan contains a global wait cycle, a `dependency cycle` is reported before the simulation starts, along with the typed cycle edges; the `cycle` attribute of the Python exception preserves the full cycle.

The original graph must contain the three lists `ops`, `tensors`, and `edges`; op/tensor IDs are non-negative integers sharing one ID space and must be unique, edge endpoints must exist, and duplicate edges as well as direct tensor→tensor edges are forbidden. Every op must carry a `pipe` field whose value is one of the four Pipes; both the simulation and the scheduler read this field directly and never guess the Pipe from the op name. Op names themselves are not whitelist-validated. Misspelled configuration keys, duplicate keys, unknown sections, invalid values, and duplicate JSON keys all raise explicit errors with no silent fallback; an explicit `--config` pointing to a nonexistent file is also an error.

## Configuration Sources

All parameters used by the evaluation come from `config.txt`; no defaults are set in the code. If the configuration is missing, the script exits with an error rather than producing a score from fixed numbers.

|Section|Keys|Meaning|Required|
|---|---|---|---|
|`[capacity]`|`L1` `UB`|On-chip L1/UB capacity (bytes)|Required|
|`[bandwidth]`|`bandwidth`|DDR↔on-chip transfer bandwidth (bytes/cycle)|Required|
|`[multicore_scene_a]`|`task_cross_core_wait_cycles` `task_same_core_wait_cycles`|Task waits for Problem 1|Required for Problem 1|
|`[multicore_scene_b]`|`cross_core_copy_delay_cycles`|Cross-core COPY wait for Problems 2/3|Required for Problems 2/3|
|`[problem_3]`|`cache_capacity_bytes` `cache_bandwidth_bytes_per_cycle`|Cache capacity and hit bandwidth for Problem 3|Required for Problem 3|

When `--config` is not specified, the script looks for `config.txt` in the directory containing the computational graph; if it is not found, `configuration file not found` is reported. Only the five sections listed above are accepted; an unknown section (e.g., a spelling mistake, or the no-longer-supported `[pipe_capacity]`) raises `unknown section` and is never silently ignored.

`schedule_step1/2/3` are internal algorithm interfaces; the DAG, topological order, ID uniqueness, and parameter constraints guaranteed by earlier stages are documented in the function entry comments and are not validated again. Only new constraints introduced by reordering or cross-core composition require new checks; Step3 retains the output-contract check for its own newly generated memory dependencies.

## File List

|File|Purpose|Suggested Audience|
|---|---|---|
|`contest_io.py`|Unified CLI, JSON, logging, and Trace handling|Readers who want to understand input/output|
|`evaluation_validation.py`|Contestant input and global execution-dependency validation|Understanding diagnostics for infeasible plans|
|`multicore_cut_evaluate_problem_1.py`|Problem 1: one Task per subgraph|Solving Problem 1|
|`multicore_cut_evaluate_problem_2.py`|Problem 2: one Task per core|Solving Problem 2|
|`multicore_cut_evaluate_problem_3.py`|Problem 3: adds a read-only FIFO Cache|Solving Problem 3|
|`schedule_step1.py`|Intra-core visitation order for acyclic graphs|Understanding topological scheduling|
|`schedule_step2.py`|On-chip capacity and Spill insertion|Understanding memory constraints|
|`schedule_step3.py`|Out-of-order arrangement, per-Pipe order, and memory-reuse supplementary edges|Understanding execution dependencies|
|`stub_multicore_cut_and_schedule.py`|Plan format validation and random examples|First time running the code|
|`singlecore_evaluate.py`|Single-core baseline|Computing the speedup|