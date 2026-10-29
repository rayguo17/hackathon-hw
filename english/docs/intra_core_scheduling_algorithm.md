# Intra-Core Scheduling Algorithm

> **Problem**: *Multi-Core Scheduling Problem on a General-Purpose Neural Network Processor*
> **Role**: Documentation of the intra-core scheduling algorithm shared by Problems 1–3; Problem 1 requires the intra-core memory peak under this algorithm, Problem 2 builds on it to produce a swap-in/swap-out plan, and Problem 3 builds on both to produce the pipelined arrangement
> **Capacity and bandwidth**: All values are taken from `config.txt`; this algorithm embeds no default values. The evaluation configuration is L1=524288 bytes, UB=131072 bytes, DDR bandwidth 60 bytes/cycle
> **Scope boundary**: This document only describes intra-core scheduling of a single Task; cross-core dependencies, shared bandwidth, Cache, and the global makespan are covered in [*Multi-Core Parallel Simulation Execution Algorithm*](多核并行模拟执行算法.md).

## Table of Contents

0. [Algorithm Overview](#0-algorithm-overview)
   - 0.1 Role and Objectives
   - 0.2 Input Formats
   - 0.3 Output Formats
   - 0.4 Mapping to Problems 1 / 2 / 3
   - 0.5 The op-tensor bipartite graph and the op-op adjacency list (conversion happens inside Step 1; Step 2 reuses the original bipartite graph)
1. [Step 1: Multi-source reverse DFS topological sort](#1-step-1-multi-source-reverse-dfs-topological-sort)
   - 1.1 Formal definitions / 1.2 Output definition and invariants
   - 1.3 Algorithm (first, bipartite-graph preprocessing and depth precomputation / then, multi-source initialization and the main loop with **predecessor priority ordering** / finally, output)
   - 1.4 Correctness proof (sketch)
2. [Step 2: Cache swap-in/swap-out insertion](#2-step-2-cache-swap-inswap-out-insertion)
   - 2.1 Objectives / 2.2 Key definitions (uses + next_use / residency / SPILL anchors)
   - 2.3 Data structures
   - 2.4 Algorithm (first, use-sequence and lifecycle precomputation / then, sequential traversal simulation with Belady victim selection + **(a0) physical swap-back of scheduled COPY_IN** / finally, extended-sequence generation)
   - 2.5 Pseudocode / 2.6 Boundary conditions / 2.7 Complexity / 2.8 Empirical validation
3. [Step 3: Out-of-order pipelined arrangement](#3-step-3-out-of-order-pipelined-arrangement)
   - 3.1 Objectives and model (4 pipes + cycle conversion)
   - 3.2 Outputs / 3.3 Key definitions (issue conditions / memory lifecycle / issue-queue priority)
   - 3.4 Interface with the multi-core simulation
   - 3.5 Pseudocode / 3.6 Complexity / 3.7 Boundary conditions / 3.8 Measured results / 3.9 Configuration file
4. [End-to-end example](#4-end-to-end-example) (a miniature 4-op graph + a forced-overflow scenario)
5. [Summary of algorithm properties](#5-summary-of-algorithm-properties)

---

## 0. Algorithm Overview

### 0.1 Role and Objectives

This algorithm breaks "scheduling a computational graph onto SIMD AI Cores" into three steps. It is **not** a concrete optimization strategy but a **deterministic input→output pipeline**: the input of each step is the output of the previous step, and every decision is made solely by rules within that step, without relying on external empirical data. Running the full pipeline yields:

- A **scheduling sequence**: original graph nodes + SPILL nodes inserted when necessary
- A set of **swap-out/swap-in records**: for each swap, the tensor ID, byte count, source op, and target op
- (provided by Step 3) the **start time** and **end time** of each node

Relationship among the three steps:

```
        ┌─────────────────────────────────┐
        │ Step 1: reverse DFS topological │
        │  sort (find an order with the   │
        │  smallest possible peak residency) │
        └────────────────┬────────────────┘
                         │ visitation order seq
                         ▼
        ┌─────────────────────────────────┐
        │ Step 2: cache swap-in/swap-out  │
        │  insertion (simulate residency  │
        │  along seq, insert SPILL on overflow) │
        └────────────────┬────────────────┘
                         │ extended graph G' = (V∪V_spill, E∪E_spill)
                         ▼
        ┌─────────────────────────────────┐
        │ Step 3: dependency readiness +  │
        │  pipe pipelining (no memory     │
        │  state, computes start_time/end_time) │
        └────────────────┬────────────────┘
                         │ scheduling timetable
                         ▼
                  total execution time = max(end_time)
```

### 0.2 Input Formats

The algorithm accepts two input formats. Internally, both formats are abstracted uniformly into two kinds of entities: "Op nodes + Tensor buffers".

#### Format A: Explicit ALLOC / FREE

```
Nodes: [
  { Id, Op, Pipe, Cycles, Bufs },     // compute/copy Op
  { Id, Op, BufId, Size, Type },      // when Op=ALLOC, no Pipe/Cycles/Bufs
  { Id, Op, BufId },                  // when Op=FREE, no Pipe/Cycles/Bufs/Size/Type
  ...
]
Edges: [ [StartNodeId, EndNodeId], ... ]
```

ALLOC / FREE nodes exist explicitly and mark the start and end of a BufId's lifetime, respectively.

#### Format B: Bipartite graph format (the format used by the evaluation cases)

```
tensors: [ { id, pos, size } ]      // pos ∈ { DDR, L1, UB }  id ≥ 10001
ops:     [ { id, op, pipe, cycles } ] // op ∈ { COPY_IN, COPY_OUT, MATMUL, ... }  id ≥ 1
edges:   [ { source, target, data_size } ]
```

Format B has no ALLOC / FREE. Internally, the algorithm infers lifetimes by the rule "the first time a tensor is touched by an op = allocation; the last time it is touched by an op = release", which is **equivalent to** implicitly inserting ALLOC / FREE nodes (but COPY_IN / COPY_OUT nodes are only explicitly inserted when a SPILL is needed).

The `pipe` field takes one of the values `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_M`, `PIPE_V`; both the scheduler and the simulator read this field directly and never infer the Pipe from the `op` name.

> **ID space collision**: in Format B the op ID range and the tensor ID range do not overlap (ops 1~N, tensors 10001~), but **inside the algorithm** the unified namespaces `o{op_id}` / `t{tensor_id}` are used to avoid ID confusion. Newly added SPILL node IDs increment starting from max(op_id, tensor_id)+1.

### 0.3 Output Formats

```
Scheduling sequence: [n1, n2, n3, ...]      // list of node IDs: original nodes + SPILL nodes inserted when needed
SPILL table:  [ { spill_out_id, spill_in_id, tid, pos, size, ... } ]
Timetable:    { nodeId: { start, end } }    // provided by Step 3
```

Each SPILL record corresponds to concrete nodes in the sequence: the first real write-back to DDR produces both a `COPY_OUT` and a `COPY_IN` node, while reusing an existing backing produces only a `COPY_IN`.

### 0.4 Mapping to Problems 1 / 2 / 3

| Problem | Corresponding steps | Description |
|------|------|------|
| Problem 1 (intra-core memory peak) | Step 1 + the simulation part of Step 2 | Step 1 produces the topological visitation order; the Step 2 simulation yields the residency peak of each storage type under that order |
| Problem 2 (swap-in/swap-out plan) | All of Step 2 | Performs overflow detection and SPILL insertion on the Problem 1 order, producing a swap-in/swap-out plan and the additional data movement |
| Problem 3 (pipelined arrangement) | Step 3 | Performs out-of-order pipelined arrangement on Problem 2's extended graph to minimize total execution time |

### 0.5 The op-tensor bipartite graph and the op-op adjacency list

In the bipartite graph format, edges are `op → tensor` or `tensor → op`; op nodes are never directly adjacent, and dependencies are bridged through tensors. For example, `op1 → T1 → op2` consists of two edges (`op1, T1` and `T1, op2`). Format A graphs are already directly op-op adjacent.

> **Step 1 of this algorithm operates on the op-op adjacency list.** When the input is a bipartite graph, the "first" phase of Step 1 performs the `op-tensor → op-op` bridging conversion while also keeping a reference to the original bipartite graph (including the `tensors` field and the original `edges` list) — Step 2's "scan by tensor lifetime" uses the original bipartite graph directly and does not re-bridge. See Section 1.3 for the bridging rules and pseudocode.

---

## 1. Step 1: Multi-source reverse DFS topological sort

### 1.1 Formal Definitions

Let DAG $G = (V, E)$ ($V$ is the set of op nodes, $|V| = $ the number of op nodes; when the input is a bipartite graph, it is obtained by bridging in the "first" phase of Section 1.3). Define the following functions and sets:

- $\text{pred}(v) \triangleq \{u \in V \mid (u, v) \in E\}$
- $\text{succ}(v) \triangleq \{w \in V \mid (v, w) \in E\}$
- $V_{\text{in}} \triangleq \{v \in V \mid \text{pred}(v) = \varnothing\}$
- $V_{\text{out}} \triangleq \{v \in V \mid \text{succ}(v) = \varnothing\}$

**Node depth** $\text{depth}: V \to \mathbb{N}_0$:

$$
\text{depth}(v) = \begin{cases} 0, & v \in V_{\text{in}} \\ 1 + \max_{p \in \text{pred}(v)} \text{depth}(p), & \text{otherwise} \end{cases}
$$

**Predicate functions** $\text{is\_copy\_in}, \text{is\_copy\_out}: V \to \{0, 1\}$ are determined by the op field values.

**Two sort keys are defined** (ascending semantics; see the main loop in 1.3 for details):

$$
\text{key}(p) = \big( \neg \text{is\_copy\_in}(p),\ \text{depth}(p),\ -\text{id}(p) \big) \quad \text{(main loop: multi-predecessor expansion order)}
$$

$$
\text{start\_key}(p) = \big( \neg \text{is\_copy\_out}(p),\ \text{depth}(p),\ -\text{id}(p) \big) \quad \text{(start: multi-source stack-push order)}
$$

> The two keys differ in their first component: the main-loop key cares about "moving data late", while the start key cares about "bottom of the stack = last element of seq = COPY_OUT".

### 1.2 Output Definition and Invariants

The algorithm outputs a **visitation order** $\text{seq}: \{0, 1, \ldots, |V|-1\} \to V$, a bijection satisfying:

| Invariant | Formalization |
|--------|--------|
| Full coverage | $\text{seq}$ is a bijection onto $V$ (i.e., all ops are visited) |
| Topological correctness | $\forall (u, v) \in E:\ \text{seq}^{-1}(u) < \text{seq}^{-1}(v)$ |
| Stack-bottom invariant | $\text{seq}(|V|-1) = \arg\min_{v \in V_{\text{out}}} \text{start\_key}(v)$ (the last element of seq is the out node with the smallest $\text{start\_key}$; when $V_{\text{out}}$ contains $\text{COPY\_OUT}$, that node must be $\text{COPY\_OUT}$) |
| Determinism | The algorithm uses no randomness and no external parameters; the same input always yields the same output |

### 1.3 Algorithm

**First, bipartite-graph preprocessing and depth precomputation**:

1. **Bipartite graph → op-op adjacency list** (bipartite case only). Bridge the op-tensor edges into op-op adjacency according to the following rule: if $\text{op}_{\text{src}} \to \tau$ and $\tau \to \text{op}_{\text{dst}}$ (where $\tau$ is a tensor), then create the dependency edge $\text{op}_{\text{src}} \to \text{op}_{\text{dst}}$.

   ```text
   Input: bipartite graph (V_op ∪ V_tensor, E_op_tensor)
   Output: op-op adjacency lists pred_map[op] / succ_map[op] (V_op = {op.id})
   Side effect: keeps a graph_json reference (including 'tensors' and the original 'edges') for reuse by Step 2

   for op in V_op:
       pred_map[op] = set()
       succ_map[op] = set()

   for edge in E_op_tensor:
       if edge = (op_src, tensor_t):
           # op_src produces tensor_t
           for edge2 in E_op_tensor where edge2.source == tensor_t and edge2.target in V_op:
               succ_map[op_src].add(edge2.target)        # op_src is a predecessor of op_dst
       elif edge = (tensor_t, op_dst):
           # tensor_t is consumed by op_dst
           for edge2 in E_op_tensor where edge2.target == tensor_t and edge2.source in V_op:
               pred_map[op_dst].add(edge2.source)        # op_src is a predecessor of op_dst
   ```

   Format A graphs are already directly op-op adjacent, so this sub-step is skipped.

2. **Depth precomputation**. Starting from $V_{\text{in}}$, a BFS derives $\text{depth}(v) = \max_{p \in \text{pred}(v)} \text{depth}(p) + 1$ (in-nodes have $\text{depth} = 0$), with complexity $O(|V| + |E|)$.

**Then, multi-source initialization and the main loop** (with predecessor priority ordering; the core of the algorithm):

**Multi-source initialization**: let $S \triangleq \text{sort}(V_{\text{out}},\ \text{by}=\text{start\_key},\ \text{asc})$ and initialize the stack $\sigma \leftarrow S$ (smaller keys are pushed first → stack bottom; larger keys are pushed later → stack top), $\text{visited} \leftarrow \varnothing$, $\text{seq} \leftarrow \varepsilon$. Degenerate case: if $V_{\text{out}} = \varnothing$ (cycle-degenerate or isolated graph), fall back to $\{v^*\}$ where $v^* = \arg\min_v \text{id}(v)$.

**Main loop**:

Loop invariant: $|\text{seq}| = |\text{visited}|$, $\text{seq}$ is a bijection of $\text{visited}$, and $\sigma$ is the "pending stack" of currently unvisited nodes.

```
while σ ≠ ∅ ∨ visited ≠ V:
    if σ = ∅:                                          // stack empty (after all sources are exhausted, handle the rest)
        R ← V \ visited
        if R = ∅: break
        Q ← sort(R, by = key, asc)                     // remaining nodes sorted ascending by the main-loop key
        σ ← σ ∘ Q                                       // smaller keys pushed first → stack bottom

    u ← top(σ)                                          // stack top
    if u ∈ visited: σ ← pop(σ); continue               // defensive deduplication

    P ← pred(u) \ visited                               // u's unvisited predecessors
    if P ≠ ∅:
        apply_pred_priority_sort(P)                     // ★ predecessor priority ordering ★
        σ ← σ ∘ Q_sorted
    else:
        visited ← visited ∪ {u}
        seq ← seq ∘ u                                    // append u to the end of seq (visit)
        σ ← pop(σ)
```

**Predecessor priority ordering (the core step of this algorithm)**:

When $u$ has multiple unvisited predecessors $P = \text{pred}(u) \setminus \text{visited}$, this decides the **push order** of those predecessors on $\sigma$, and thereby the **visitation order** of these predecessors after $u$ has been visited. The ordering procedure:

1. **Define the key function** (already given in 1.1):

   $$\text{key}(p) = \big( \neg \text{is\_copy\_in}(p),\ \text{depth}(p),\ -\text{id}(p) \big)$$

2. **Sort $P$ ascending by key**: $Q = \text{sort}(P,\ \text{by}=\text{key},\ \text{asc})$

3. **Push in ascending order**: append $Q$ one by one to the end of $\sigma$, i.e., $\sigma \leftarrow \sigma \circ Q$.

**Why "ascending push + LIFO pop" is equivalent to "visiting in descending key order"**:

- Ascending sort $\Rightarrow$ the smallest-key $p_1$ is **first** in $Q$, and is **pushed first** $\Rightarrow$ it sits **deepest** (bottom) in $\sigma$
- Ascending sort $\Rightarrow$ the largest-key $p_k$ is **last** in $Q$, and is **pushed last** $\Rightarrow$ it sits **shallowest** (top) in $\sigma$
- LIFO popping $\Rightarrow$ $p_k$ (largest key) is **popped first** $\Rightarrow$ **visited first**
- $p_1$ (smallest key) is **popped later** $\Rightarrow$ **visited later**

That is: the larger $\text{key}(p)$ $\Rightarrow$ $p$ is visited earlier $\Rightarrow$ $p$ executes earlier in physical time.

**Design intent of the three key components**:

| Priority | Key dimension | Ascending rule (smaller pushed first → visited later) | Design intent |
|--------|----------|-----------------------------------|----------|
| First | $\neg \text{is\_copy\_in}$ | $\text{COPY\_IN}$ first (key = 0) → visited later | Data is "brought in only when needed": $\text{COPY\_IN}$ **executes late** in physical time, so the tensors it produces have short residency |
| Second | $\text{depth}$ | Shallow first (small depth) → visited later | "Handle small computation branches later": tensors produced by shallow nodes have short lifetimes, and late physical execution compresses their lifetime |
| Third | $-\text{id}$ | Larger id first (smaller $-id$) → visited later | Earlier lexicographic order → earlier physical execution: when multiple predecessors share the same depth, the visitation order follows ascending input Id |

**Finally, output**: return $\text{seq}$.

### 1.4 Correctness Proof (Sketch)

**Proposition 1.1 (Topological correctness)**. $\forall (u, v) \in E: \text{seq}^{-1}(u) < \text{seq}^{-1}(v)$.

*Proof*: $u$ is appended to $\text{seq}$ only when $P = \text{pred}(u) = \varnothing \lor \text{pred}(u) \subseteq \text{visited}$. The latter case requires every $p \in \text{pred}(u)$ to have been visited already, and $(u, v) \in E$ means $u \in \text{pred}(v)$; hence $v$'s visitation condition requires $u$ to be visited first. ∎

**Proposition 1.2 (Full coverage)**. $\text{seq}$ is a bijection onto $V$ and $|\text{seq}| = |V|$.

*Proof*: The loop condition is $\sigma \neq \varnothing \lor \text{visited} \neq V$. After each append of $u$, $\text{visited} \leftarrow \text{visited} \cup \{u\}$ (deduplicated) and $\sigma$ shrinks correspondingly, so the total number of elements is conserved. At termination $\text{visited} = V$ and $\text{seq}$ has length $|V|$. ∎

**Proposition 1.3 (Stack-bottom invariant)**. $\text{seq}(|V|-1) = \arg\min_{v \in V_{\text{out}}} \text{start\_key}(v)$.

*Proof*: By Step 1.A, the out node with $\arg\min \text{start\_key}$ is the **bottom** of $\sigma$. All subsequent push operations in the main loop ($\sigma \leftarrow \sigma \circ Q$) append new elements to the top, so the bottom never changes. Under LIFO, the bottom is popped last, i.e., it is the last element of $\text{seq}$. ∎

**Proposition 1.4 (Determinism)**. The algorithm uses no random numbers and no external parameters; all sorts are based on deterministic keys. The same input always yields the same output. ∎

---

## 2. Step 2: Cache swap-in/swap-out insertion

> **Interface**: `step2_spill_insertion(graph_json, seq, capacity)`. `capacity` is read by the caller from the `[capacity]` section of `config.txt` and passed in; this step embeds no default capacity.
>
> This step renames the output of each `COPY_IN` as a new on-chip tensor incarnation and reconnects all consumers after this swap-in to that tensor; each logical tensor gets only one persistent DDR backing, and once a backing exists, no further write-back to DDR is performed.

### 2.1 Objectives

Given:

- The visitation order $\text{seq}$ output by Step 1 (under the "visited = physically executed" semantics)
- The capacity $C_T$ of each Type (from `config.txt`: L1=524288 bytes, UB=131072 bytes)
- The original bipartite graph (Step 1's "first" phase already kept the graph_json reference)

Output:

- The extended sequence $\text{seq}_{\text{ext}}$: $\text{seq}$ plus a number of `COPY_OUT` / `COPY_IN` nodes (**inserted between ops**)
- The extended graph $G' = (V \cup V_{\text{spill}},\, E \cup E_{\text{spill}})$
- The SPILL operation record table

such that: **at any moment and for every Type, the residency $\text{resid}_T(t) \le C_T$**.

> **Key simplification**: the algorithm **does not allocate concrete addresses** (offsets); it only tracks "usage amounts".

### 2.2 Key Definitions

#### 2.2.1 Tensor use sequences (uses) and next_use

For every **non-DDR** tensor $\tau$ (DDR tensors occupy no on-chip capacity and are not handled by Step 2), define its **use sequence** as the set of ops that involve it, sorted by their seq step:

$$
\text{uses}(\tau) = \big[(\text{step}_1,\, \text{op}_1),\ (\text{step}_2,\, \text{op}_2),\ \ldots,\ (\text{step}_k,\, \text{op}_k)\big]
$$

With the convention $\text{step}_1 < \ldots < \text{step}_k$, $\text{op}_1$ is $\tau$'s producing op and the rest are consuming ops ($k=1$ means produced and consumed immediately, common for intermediate products of COPY_IN / COPY_OUT).

During the simulation, a **next_use pointer** is maintained for $\tau$: after the $i$-th use has been processed, $\text{next\_use}(\tau, i) = \text{step}_{i+1}$ (if $i < k$), otherwise $\varnothing$ ($\tau$ is no longer used and will be released at the next step). next_use provides the "**not used for the longest time in the future**" criterion for victim selection (Belady victim selection; see 2.4 (c)).

#### 2.2.2 Residency, active sets, and use handling

$$
\text{resid}_T(t) \;=\; \sum_{\tau \in \text{Active}_T(t)} \text{size}(\tau)
$$

- $T \in \{\text{L1}, \text{UB}\}$; each Type is checked independently
- $\text{Active}^{alloc}_T(t)$: all resident tensors of type $T$ at the moment when the current op's COPY_IN and outputs have all been allocated but its last input has not yet been released; this is the hard capacity-constraint checkpoint
- $\text{Active}^{steady}_T(t)$: the steady-state set after the current op has finished executing and released its last input
- $C_T$: the hardware capacity of Type $T$

The convention is that $\tau$ is **alive iff** $\text{first}(\tau) \le t < \text{last}(\tau)$ (half-open, with $\text{first} = \text{step}_1$ and $\text{last} = \text{step}_k$). Use handling branches on the current use index $i$ (the position in $\text{uses}(\tau)$):

- $i = 1$ (production): first add to $\text{Active}_T$; $\text{next\_use} = \text{step}_2$ (if $k \ge 2$) or $\varnothing$
- $1 < i < k$ (intermediate use): update $\tau$'s $\text{next\_use}$ in active to $\text{step}_{i+1}$
- $i = k$ (last use): still kept in $\text{Active}^{alloc}_T$ during the capacity check; removed only after the current op has finished executing

At each step, after allocation completes and before execute/free, the algorithm checks $\text{resid}^{alloc}_T(t) > C_T$; on overflow it triggers 2.4 (c) Belady victim selection + SPILL insertion. After the SPILL it asserts again that $\text{resid}^{alloc}_T(t) \le C_T$.

#### 2.2.3 SPILL anchors (use gaps)

After victim $\tau$ is selected, the SPILL is inserted inside $\tau$'s **use gap** (within the interval $[\text{out\_after\_step}, \text{in\_before\_step}]$, $\tau$ is not used by any op, so spilling to DDR does not conflict with active uses):

- $\text{out\_after\_step} = \text{step}_i$ (the last use step that has already occurred): the `COPY_OUT` node is inserted after $\text{seq}[\text{step}_i]$ (present only for the first real write-back)
- $\text{in\_before\_step} = \text{step}_{i+1}$ (the next use step): the `COPY_IN` node is inserted before $\text{seq}[\text{step}_{i+1}]$

> **Physical-state closure**: after the SPILL removes the victim from active, the victim must be added back to active at `in_before_step` via the swap-in node; otherwise the victim's later uses cannot find its active entry and the residency accounting becomes distorted. See the implementation in 2.4 (a0).

#### 2.2.4 Persistent DDR backing and tensor renaming

Each on-chip tensor $\tau$ establishes at most one persistent DDR backing; all subsequent swap-ins copy back from the same backing:

| Case | COPY_OUT node | DDR tensor | COPY_IN output |
|------|------|------|------|
| $\tau$ has no backing yet | Kept, writes DDR for real | Create a new persistent backing | Create new $\tau@k$, wire `backing → IN → τ@k` |
| $\tau$ already has a backing | Not created | Reuse the existing backing | Create new $\tau@k$, wire `backing → IN → τ@k` |
| $\tau$ was produced by an original COPY_IN | Not created | Reuse the original COPY_IN's DDR input | Same as above |

The original logical tensor receives a new physical incarnation after each swap-in: $\tau@0,\tau@1,\ldots$. All consumer edges between two spill boundaries are rewired to the corresponding incarnation. Step3 can therefore build reference counts directly from tensor fan-out, with no extra residency segment table.

### 2.3 Data Structures

```text
seq                : List[op_id]                     # Step 1 output
graph_json         : original bipartite graph (with ops/tensors/edges)
capacity_of        : Dict[Type → int]                # from config.txt, e.g. L1: 524288
tensor_uses        : Dict[tid → List[(step, op_id)]]  # precomputed in 2.4
tensor_lifecycle   : Dict[tid → {first, last, pos, size, uses}]
type_active[T]     : Dict[tid → (next_use, used_count)]  # per Type, for Belady victim selection
type_resid[T]      : int                             # current residency of Type T
cur_use_idx[tid]   : int                             # current use count (index of the next use)
pending_spills     : List[SpillSpec]                 # collected, inserted together at the end
overflow_log       : List[OverflowEvent]             # per-step overflow records
spill_records      : List[SpillRecord]               # output
new_ops            : List[{id, op, pipe, cycles}]    # newly added COPY_OUT / COPY_IN nodes
new_tensors        : List[{id, logical_tid?, version?, pos, size}] # DDR backing + renamed on-chip tensors
new_edges          : List[(src, dst)]                # new dependencies in the extended graph
removed_edges      : List[(src, dst)]                # old logical-tensor consumer edges that were rewired
ext_edges          : List[{source, target}]          # complete edge set after renaming
backing_by_tid     : Dict[tid → DDR tid]             # the tensor's persistent DDR backing
seq_ext            : List[node_id]                   # extended sequence
```

### 2.4 Algorithm

**First, precompute tensor use sequences and lifecycles**:

1. Traverse the graph_json bipartite graph; for each non-DDR tensor $\tau$:
   - Collect all ops involving $\tau$ (producer + all consumers)
   - Sort them by the op's step in $\text{seq}$ to obtain $\text{uses}(\tau) = [(\text{step}_1, \text{op}_1), \ldots]$
   - Record $\text{first}(\tau) = \text{step}_1$, $\text{last}(\tau) = \text{step}_k$, $\text{pos}(\tau)$, $\text{size}(\tau)$
2. Initialize: $\text{cur\_use\_idx}[\tau] = 0$ (pointing to the next entry of $\text{uses}(\tau)$), $\text{active}[T] = \varnothing$
3. Complexity $O(|V| + |E|)$, the same order as Step 1's "first" phase.

**Then, sequential traversal simulation + Belady victim selection**:

Traverse in $\text{seq}$ order with $t = 0, 1, \ldots, |\text{seq}|-1$. Each step strictly performs the five actions below:

**(a0) Handle scheduled COPY_IN** (**must come before (a)**):

For every triggered SPILL whose $\text{in\_before\_step} = t$, add the corresponding victim $\tau$ back into $\text{active}[\text{pos}(\tau)]$ (equivalent to the physical COPY_IN loading $\tau$ from DDR back on-chip). Let the victim's current use index in $\text{uses}(\tau)$ be $u$ (i.e., $\text{uses}(\tau)[u].\text{step} = t$); the new entry is recorded as $(\text{uses}(\tau)[u+1].\text{step},\ u,\ \tau)$.

> **Why it must come before (a)**: the intermediate-use branch of (a) only **updates existing entries**, it does not **add** new ones. If the victim were removed from active first and (a) then ran, the intermediate use would not find the tensor in active, so this step's next_use would not be advanced. Therefore the victim must first be added back to active in (a0), and then (a) updates next_use as usual.

> **No double counting with the first-use branch**: the `u` written by (a0) and the `u+1` written by (a) differ by 1: when entering (a0), cur_use_idx has not yet been incremented, so the victim has completed $u$ uses (uses[0..u-1]); after (a) processes uses[u], it becomes $u+1$. Both represent "the number of uses completed at the moment the event fires".

**(a) Handle this step's use alloc / next_use, with deferred last-use free**: for every $\tau$ satisfying $\text{uses}(\tau)[\text{cur\_use\_idx}[\tau]].step = t$:

- $\text{cur\_use\_idx}[\tau] = 0$ (production): add $\tau$ to $\text{active}[\text{pos}(\tau)]$ with $\text{next\_use} = \text{uses}(\tau)[1].step$ (if $k \ge 2$), otherwise $\varnothing$; even when $k=1$, it is added first so it counts toward this op's allocation peak
- $0 < \text{cur\_use\_idx}[\tau] < k-1$ (intermediate use): update $\tau$'s $\text{next\_use}$ in active to $\text{uses}(\tau)[\text{cur\_use\_idx}+1].step$
- $\text{cur\_use\_idx}[\tau] = k-1$ (last use): add to `release_after_execute`; it remains in active for now
- $\text{cur\_use\_idx}[\tau] \gets \text{cur\_use\_idx}[\tau] + 1$

**(b) Detect overflow per Type** (independently per Type):

At this point the COPY_IN, the current inputs, and the outputs are all resident, and the last input has not yet been freed. If $\text{resid}^{alloc}_T(t) > C_T$, record it in $\text{overflow\_log}$ and enter the **(c)** loop until $\text{resid}^{alloc}_T(t) \le C_T$.

**(c) Belady victim selection + triggering SPILL**:

```
victim_pool[T] = { (next_use, used_count, tid) ∈ active[T]
                   | next_use ≠ ∅ and tid is not among the current op's inputs/outputs }
if victim_pool[T] is empty:
    # unreachable under normal input constraints; never clear active to mask errors
    print step/op/type/resid/capacity/current_tids/active to stderr and the case log
    raise Step2SchedulingError
v = argmax_{v ∈ victim_pool[T]} v.next_use      # Belady
prev_use_step = uses(v.tid)[v.used_count - 1].step
next_use_step = v.next_use
prev_use_op   = uses(v.tid)[v.used_count - 1].op
next_use_op   = uses(v.tid)[v.used_count    ].op
# remove from active[T] (v.tid no longer occupies T capacity; the data is in DDR)
active[T] = active[T] \ {v}
# record the SPILL spec
pending_spills.append({
    'out_after_step': prev_use_step,
    'in_before_step': next_use_step,
    'tid': v.tid, 'pos': T, 'size': size(v.tid),
    'prev_use_op': prev_use_op, 'next_use_op': next_use_op,
})
# return to (b) and re-check
```

**(d) Posterior capacity assertion**: after the SPILL loop finishes, assert $\text{resid}^{alloc}_T(t) \le C_T$. On failure, print step/op/type/resid/capacity and raise `Step2SchedulingError`.

**(e) execute + last-use free**: after the current op finishes executing, remove in one pass from active all tensors in `release_after_execute`, yielding $\text{Active}^{steady}_T(t)$.

**Key properties**:

- Victim selection only considers active tensors with $\text{next\_use} \neq \varnothing$ that are not among the current op's inputs/outputs, guaranteeing that the write-back to DDR does not break the current op
- Finding no victim is a hard error, not a recoverable state; under the premise that "a single op's total current residency does not exceed capacity" it is theoretically unreachable, so it serves as a guard against implementation/input violations
- Belady is optimal for a single cache: swap out the one "not used for the longest time in the future", letting it return the latest

**Finally, extended-sequence generation**:

1. Every pending spill always gets a COPY_IN id and a new on-chip tensor id; only when no backing exists yet are a real COPY_OUT id and a DDR backing id allocated.
2. Wire `backing → COPY_IN → renamed_tensor`; a real COPY_OUT wires `old_incarnation → COPY_OUT → backing`.
3. Rewire all consumer edges from the next swap-in up to the next spill boundary from the logical tensor to the renamed tensor.
4. Build $\text{seq}_{\text{ext}}$: the COPY_IN is inserted before the next use; only real COPY_OUTs are inserted at their corresponding positions.

$$
\text{seq}_{\text{ext}}[i] = \begin{cases}
\text{insert\_after}[i-1] \circ \text{insert\_before}[i] \circ \text{seq}[i] & i = 0 \\
\text{insert\_before}[i] \circ \text{seq}[i] \circ \text{insert\_after}[i] & 0 < i < n \\
\text{insert\_before}[n-1] \circ \text{seq}[n-1] \circ \text{insert\_after}[n-1] & i = n-1
\end{cases}
$$

where $\text{insert\_after}[s]$ = the list of `COPY_OUT` nodes inserted after $\text{seq}[s]$ (in generation order when several share the same position), and $\text{insert\_before}[s]$ = the list of `COPY_IN` nodes inserted before $\text{seq}[s]$.

> When a backing already exists, only a COPY_IN is inserted at that position; the release of the previous incarnation is handled automatically by Step3 once all of its consumers have completed.

### 2.5 Pseudocode

```python
def step2_spill_insertion(graph_json, seq, capacity):
    # capacity is read by the caller from the [capacity] section of config.txt and passed in; no fallback defaults here

    # ---- 2.4 first: precomputation ----
    tensor_uses, op_step, _ = _build_tensor_uses(graph_json, seq)
    tensor_lifecycle = {}
    for t in graph_json['tensors']:
        if t['pos'] == 'DDR' or t['id'] not in tensor_uses:
            continue
        uses = tensor_uses[t['id']]
        tensor_lifecycle[t['id']] = {
            'pos': t['pos'], 'size': t['size'], 'uses': uses,
            'first': uses[0][0], 'last': uses[-1][0],
        }

    max_id = max(o['id'] for o in graph_json['ops'])
    max_tid = max((t['id'] for t in graph_json['tensors']), default=0)
    next_id = max(max_id, max_tid) + 1

    # ---- 2.4 then: sequential traversal + Belady ----
    active = {T: [] for T in capacity}      # list of (next_use, used_count, tid)
    cur_use_idx = {tid: 0 for tid in tensor_lifecycle}
    pending_spills = []
    overflow_log = []

    def get_resid(T):
        return sum(tensor_lifecycle[t2]['size'] for (_, _, t2) in active[T])

    for t in range(len(seq)):
        # (a0) scheduled COPY_IN: allocate the spilled victim back on-chip
        #     must run before (a), otherwise (a)'s intermediate-use branch cannot find
        #     the victim and next_use cannot be advanced
        for sp in pending_spills:
            if sp['in_before_step'] != t:
                continue
            tid = sp['tid']
            if tid not in tensor_lifecycle:   # DDR swaps do not participate
                continue
            T = tensor_lifecycle[tid]['pos']
            uses = tensor_lifecycle[tid]['uses']
            for u, (s, _) in enumerate(uses):
                if s != t:
                    continue
                # uses[u] is the victim's intermediate use at step t (after the SPILL
                # swapped it out, it is brought back via COPY_IN). At this point the
                # victim should not already be in active (it was removed at swap-out).
                new_nu = uses[u + 1][0] if u + 1 < len(uses) else None
                if not any(t2 == tid for (_, _, t2) in active[T]):
                    active[T].append((new_nu, u, tid))   # ui = u (cur_use_idx has not been incremented yet)
                break

        # (a) alloc / update next_use; last-use free is deferred until after the capacity check
        release_after_execute = []
        for tid, info in tensor_lifecycle.items():
            uses = info['uses']
            idx = cur_use_idx[tid]
            if idx < len(uses) and uses[idx][0] == t:
                T = info['pos']
                if idx == 0:           # production
                    next_nu = uses[1][0] if 1 < len(uses) else None
                    active[T].append((next_nu, 1, tid))
                elif idx < len(uses) - 1:    # intermediate use, update next_use
                    new_nu = uses[idx + 1][0] if idx + 1 < len(uses) else None
                    for j, (nu, ui, t2) in enumerate(active[T]):
                        if t2 == tid:
                            active[T][j] = (new_nu, ui + 1, tid)
                            break
                if idx == len(uses) - 1:
                    release_after_execute.append((T, tid))
                cur_use_idx[tid] = idx + 1

        # (b) check after alloc and before execute/free + (c) Belady SPILL
        for T, C in capacity.items():
            resid = get_resid(T)
            while resid > C:
                overflow_log.append({'step': t, 'type': T, 'resid': resid, 'capacity': C})
                finite = [(nu, ui, tid) for (nu, ui, tid) in active[T]
                          if nu is not None
                          and not any(s == t for s, _ in tensor_lifecycle[tid]['uses'])]
                if not finite:
                    print_error_context_to_stderr_and_case_log(...)
                    raise Step2SchedulingError(...)
                nu, ui, victim_tid = max(finite, key=lambda x: x[0])
                info = tensor_lifecycle[victim_tid]
                active[T] = [(nu2, ui2, t2) for (nu2, ui2, t2) in active[T]
                             if t2 != victim_tid]
                pending_spills.append({
                    'out_after_step': info['uses'][ui - 1][0],
                    'in_before_step': nu,
                    'tid': victim_tid, 'pos': T, 'size': info['size'],
                    'prev_use_op': info['uses'][ui - 1][1],
                    'next_use_op': info['uses'][ui][1],
                })
                resid = get_resid(T)
            assert resid <= C

        # (e) release last-use tensors only after execution completes
        for T, tid in release_after_execute:
            active[T] = [(nu, ui, t2) for (nu, ui, t2) in active[T] if t2 != tid]

    # ---- 2.4 finally: extended-sequence generation ----
    new_ops_list, new_tensors_list, new_edges, spill_records = [], [], [], []
    insert_after = defaultdict(list)   # pos -> [spill_out_id]
    insert_before = defaultdict(list)  # pos -> [spill_in_id]

    backing_by_tid = find_original_copy_in_backings(graph_json)
    current_incarnation = {tid: tid for tid in tensor_lifecycle}
    for sp in pending_spills:
        si_id = next_id; next_id += 1
        copies_data = sp['tid'] not in backing_by_tid
        from_tid = current_incarnation[sp['tid']]
        if copies_data:
            so_id = next_id; next_id += 1
            backing_tid = next_id; next_id += 1
            backing_by_tid[sp['tid']] = backing_tid
            new_tensors_list.append({'id': backing_tid, 'pos': 'DDR', 'size': sp['size']})
        else:
            so_id = None
            backing_tid = backing_by_tid[sp['tid']]
        to_tid = next_id; next_id += 1
        current_incarnation[sp['tid']] = to_tid
        new_tensors_list.append({
            'id': to_tid, 'logical_tid': sp['tid'], 'version': next_version(sp['tid']),
            'pos': sp['pos'], 'size': sp['size']})
        if copies_data:
            new_ops_list.append({'id': so_id, 'op': 'COPY_OUT', 'pipe': 'PIPE_MTE3',
                                 'cycles': max(1, sp['size'] // 64)})
        new_ops_list.append({'id': si_id, 'op': 'COPY_IN', 'pipe': 'PIPE_MTE2',
                             'cycles': max(1, sp['size'] // 64)})
        if copies_data:
            new_edges.append((from_tid, so_id))
            new_edges.append((so_id, backing_tid))
        new_edges.append((backing_tid, si_id))
        new_edges.append((si_id, to_tid))
        if copies_data:
            insert_after[sp['out_after_step']].append(so_id)
        insert_before[sp['in_before_step']].append(si_id)
        spill_records.append({
            'spill_out_id': so_id, 'spill_in_id': si_id,
            'backing_tid': backing_tid,
            'spill_out_copies_data': copies_data,
            'from_tid': from_tid, 'to_tid': to_tid,
            'tid': sp['tid'], 'pos': sp['pos'], 'size': sp['size'],
            'prev_use_op': sp['prev_use_op'], 'next_use_op': sp['next_use_op'],
            'prev_use_step': sp['out_after_step'], 'next_use_step': sp['in_before_step'],
        })

    # for each logical_tid -> consumer edge, pick the corresponding incarnation
    # by the consumer's seq step and rewire the edge to incarnation_tid -> consumer.
    ext_edges = rewire_all_consumers(graph_json.edges, spill_records) + new_edges

    seq_ext = []
    for i, op in enumerate(seq):
        if i > 0:
            for sid in insert_after[i - 1]: seq_ext.append(sid)
        for sid in insert_before[i]: seq_ext.append(sid)
        seq_ext.append(op)
    for sid in insert_after[len(seq) - 1]: seq_ext.append(sid)

    return {
        'seq_ext': seq_ext, 'spill_records': spill_records,
        'new_edges': new_edges, 'removed_edges': removed_edges,
        'ext_edges': ext_edges, 'new_ops': new_ops_list,
        'new_tensors': new_tensors_list,
        'overflow_log': overflow_log, 'capacity': capacity,
    }
```

### 2.6 Boundary Conditions

| Scenario | Handling |
|------|------|
| $k = 1$ ($\tau$ has a single use, produced and consumed immediately) | Allocate first and count it toward the current op's transient peak; free immediately after execute; it belongs to the current op and cannot be selected as a victim |
| $k = 0$ (isolated tensor with no uses) | Skipped during precomputation; never enters active and never participates in SPILL |
| $\tau$ with $\text{next\_use} = \varnothing$ | Skipped during victim selection; if capacity is still exceeded and no other valid victim exists, print the error context and raise `Step2SchedulingError` — never clear active |
| Victim's $\text{size}$ exceeds $C_T$ | A single tensor does not fit; this algorithm will not select it (logically a victim must be in $\text{active}[T]$, which means it originally fit) |
| Multiple Types overflow simultaneously | Each Type loops independently (the b/c parts of the "then" phase in 2.4 sit inside the for-T loop) |
| Multiple tensors of the same Type overflow simultaneously | Repeat the while loop until $\text{resid}_T \le C_T$ |
| The same tensor selected as victim multiple times | Each time a new COPY_IN and a renamed tensor are created; only the first performs a real write-back to DDR and creates the backing |
| Tensor produced from DDR by COPY_IN | Reuse the original DDR input as the backing; no additional COPY_OUT writing back to DDR is created |
| Position $i$ has both $\text{insert\_after}[i-1]$ and $\text{insert\_before}[i]$ | $\text{seq}_{\text{ext}}$ order: $\text{insert\_after}[i-1]$ → $\text{insert\_before}[i]$ → $\text{seq}[i]$ |
| Multiple COPY_OUTs at the same position | Serialized in generation order; they do not depend on each other and can run in parallel |
| No SPILL after $\text{seq}[i-1]$ and none before $\text{seq}[i]$ | The $\text{seq}_{\text{ext}}$ element equals $\text{seq}[i]$; no change |
| Huge $\text{step}$ gaps within $k$ | The SPILL spans many original ops; the COPY_IN is still inserted before the op corresponding to $\text{step}_{i+1}$ |

### 2.7 Complexity

- Precomputing use sequences: $O(|V| + |E|)$ (each bipartite-graph edge contributes one use record; $O(|V| \log |V|)$ after sorting)
- Main loop: $O(\log R)$ per step (victim search is a max over the list, $O(R)$; $R$ = active size, worst case $O(|V|)$). Overall worst case $O(|V|^2)$, which can be improved to $O(|V| \log |V|)$ with a heap
- **(a0) scanning scheduled COPY_INs**: $O(S)$ per step ($S$ = total number of pending_spills), worst case $|V|$. Since $\sum_t (\text{number of (a0) hits}) = O(S)$, the total cost $O(S)$ amortized over $|V|$ steps = expected $O(S/|V|)$ per step
- Number of SPILLs: worst case $O(|V|)$ (every tensor gets swapped out and swapped back)
- Total time: $O(|V|^2 + S)$ (current implementation) / $O(|V| \log |V| + S)$ (with the heap optimization)
- Total space: $O(|V| + S)$

### 2.8 Empirical Validation (4 cases, capacity L1=524288, UB=131072)

| Case | $\|V\|$ | $\#\text{edges}$ | SPILL count | $\text{peak}_{L1}$ | L1 usage | $\text{peak}_{UB}$ | UB usage |
|------|------|------|------|------|------|------|------|
| small\_50  | 43  | 110  | 0  | 352000  | 67.1%  | 50247   | 38.3%  |
| medium\_100 | 88  | 239  | 4  | 492000  | 93.8%  | 56232   | 42.9%  |
| medium\_150 | 131 | 376  | 7  | 516018  | **98.4%**  | 118049  | 90.1%  |
| large\_200  | 186 | 530  | 16 | 510475  | **97.4%**  | 127680  | **97.4%**  |

All 4 cases were independently replayed against the final $\text{seq}_{ext}$; the post-alloc L1/UB peaks are all $\le$ capacity ($\le 100\%$). For medium\_100 the post-alloc L1 peak is 492000: if the release of last-use inputs were not deferred until after the capacity check, that peak would only be counted as 452640, thereby missing the extra swap-out of t10064 (out\_after=29, in\_before=56) required before executing `op#43`.

For large\_200, the final alloc peaks are L1=510475 and UB=127680, both consistent with an independent replay of the extended sequence.

---

## 3. Step 3: Out-of-order pipelined arrangement

> **Interface**: `step3_simulation(ext_graph, capacity, bandwidth)`. Capacity and bandwidth are read by the caller from `config.txt` and passed in; this step embeds no default values.

### 3.1 Inputs, Objectives, and Outputs

Inputs:

- The extended graph produced by Step2, `ext_graph = {ops, tensors, edges, seq_ext}`
- The number of executors for the 4 Pipe types is fixed at 1 per Pipe (a hardware constraint, not configurable)
- The DDR bandwidth parameter `bandwidth` (from the `[bandwidth]` section of `config.txt`; the evaluation configuration uses 60 bytes/cycle); this step uses it to compute the reference duration of a single DDR COPY with exclusive bandwidth
- The capacity of each on-chip memory type `capacity` (from the `[capacity]` section of `config.txt`)

Objective: issue ops as early as possible while satisfying all graph dependencies and Pipe exclusivity. Outputs:

- `makespan`
- `op_start[op_id]`, `op_end[op_id]`
- `per_pipe_timeline[pipe]`, `pipe_orders[pipe]`
- `op_status` and `pred_remaining` (on a successful return, all ops are done)
- `memory_peak`, `memory_events`, `memory_used_final`
- `execution_graph` (the execution graph after memory-reuse dependencies are added) and `memory_dependencies`

Step3 never returns a partial scheduling result. On deadlock, it goes straight to the failure path in 3.5.

### 3.2 Unified Dependency Construction

Step3 only interprets the graph structure; it does not recognize node origins:

1. For `op → tensor → op` paths, build producer-to-consumer dependencies.
2. For direct `op → op` edges, build dependencies as-is.
3. DDR tensors likewise participate in data dependencies and COPY duration computation, but no memory state is built for them.
4. The COPY_IN/COPY_OUT nodes added by Step2 enter the dependency graph and Pipe queues just like the original ops.
5. The Pipe an op belongs to is taken directly from that op's own `pipe` field, never inferred from the `op` name.

Each renamed tensor node is one physical residency lifetime segment; Step3 needs no separate residency segment table.

### 3.3 Scheduling Algorithm

**Preconditions** (guaranteed by Step2's output; Step3 does not re-validate them):

- `seq_ext` has length equal to the number of ops and is a permutation of all ops;
- `ext_graph` is acyclic, and the capacity requirements of all on-chip tensors fit within `capacity`.

**Then, initialization**:

- Build `preds/succs` and `pred_remaining`;
- For each on-chip tensor, build its producer and consumer sets and `remaining_consumers`;
- On-chip inputs without a producer are pre-allocated at $t=0$;
- All ops producing on-chip outputs form the allocation order, in the same order as `seq_ext`.

**Three phases per main-loop iteration**:

1. **retire**: after an op completes, decrement the reference counts of all its input tensors; when a count reaches zero, free immediately and decrement successor dependencies.
2. **issue**: an on-chip output allocation must have its turn in the allocation order and satisfy `used + required ≤ capacity`; on success, allocate first and then issue. Ops without on-chip outputs may bypass allocation ops that are blocked by capacity. At the intra-core stage, a DDR COPY only records its reference workload with exclusive bandwidth.
3. **advance**: jump to the earliest finish time among all running ops.

Terminates successfully once all ops are done and every type has `memory_used_final = 0`. Step3 includes memory allocation, release, and capacity waiting, but no victim selection or second swap-out.

### 3.4 Interface with the Multi-Core Simulation

The intra-core step outputs the extended graph, the deterministic operation order, each COPY's data size, and its exclusive-bandwidth reference workload; it does not decide cross-core completion times within a single Task. The multi-core evaluator loads each Task's operations into the corresponding core's Pipe queues and uniformly handles cross-core dependencies, DDR/Cache shared bandwidth, and the final time advancement. See [*Multi-Core Parallel Simulation Execution Algorithm*](多核并行模拟执行算法.md) for the exact rules.

### 3.5 Deadlock Interception and Error Logging

Any of the following is treated as a scheduling failure, with no partial results returned:

- Incomplete ops remain, but all issue queues and executors are empty (`scheduler deadlock`);
- The iteration count exceeds `max_iter`;
- An op's `pipe` field is not one of the four Pipes (`missing pipe capacity`);
- A gap appears in virtual capacity credits on release dependencies (`virtual capacity credits exhausted`);
- The initial resident tensors already exceed capacity (`initial resident tensors exceed capacity`);
- The output contract check fails: the Pipe order does not cover every op exactly once, or the memory-reuse dependencies form a cycle.

On failure:

1. Print diagnostics beginning with `[STEP3 ERROR]` to stderr;
2. Raise `Step3SchedulingError` without returning partial results;
3. The evaluation entry point catches it, prints `[EVALUATION ERROR]` to stderr, and exits with a non-zero code; the case's result JSON and Trace are not written, and any log files already written contain no success results;
4. The deadlock diagnosis includes at least reason, time, iteration, status counts, each Pipe's queue/executor, and, for every incomplete op:
   - op id / type / pipe / seq_pos / status
   - `pred_remaining`
   - `unresolved_preds`
   - `in_tids/out_tids`

Under the premise that Step2 strictly guarantees memory, an acyclic extended graph, and valid Pipe fields, this failure path is theoretically unreachable and serves as a guard against input and implementation errors.

### 3.6 COPY Cycles and Pipe

| op | pipe | duration |
|---|---|---|
| COPY_IN | PIPE_MTE2 | Prefers total output tensor size / bandwidth; falls back to op.cycles when there is no explicit output tensor; participates in bandwidth sharing when an endpoint is DDR |
| COPY_OUT | PIPE_MTE3 | Prefers total input tensor size / bandwidth; falls back to op.cycles when there is no explicit input tensor; participates in bandwidth sharing when an endpoint is DDR |
| MATMUL/GEMM/CONV | PIPE_M | op.cycles |
| Other vector ops | PIPE_V | op.cycles |

Section 2 of the Step3 log additionally prints, for each `COPY_IN/COPY_OUT`:

- `full_bw_duration`: the reference duration of this COPY when it exclusively occupies the corresponding full bandwidth;
- `full_bw_end = start + full_bw_duration`: the reference end time starting from the actual issue moment with no bandwidth preemption.

The actual `end` and `duration=end-start` still include DDR preemption and the final cycle ceiling, and can be compared directly with the reference values above.

### 3.7 Complexity

- Dependency construction: $O(|V| + |E|)$
- Each op is enqueued and dequeued once: $O(|V| \log |V|)$
- retire and successor wake-ups: $O(|E|)$ in total
- Recomputing the in-flight set on each DDR COPY issue: $O(M\log M)$, where $M$ is the number of DDR transfers in flight simultaneously; currently $M\le2$ on a single core
- Space: $O(|V| + |E|)$

### 3.8 Measured Results

Using the extended graphs obtained via Step2 for the four cases above:

| Case | Ops completed | makespan | Deadlocks |
|---|---:|---:|---:|
| small_50 | 43/43 | 6815 | 0 |
| medium_100 | 96/96 | 15432 | 0 |
| medium_150 | 145/145 | 21241 | 0 |
| large_200 | 218/218 | 35542 | 0 |

All four cases finished completely; the makespan already includes the fair bandwidth preemption when MTE2/MTE3 access DDR simultaneously.

### 3.9 Configuration File

The two configuration items used by intra-core scheduling come from `config.txt`:

```text
[capacity]
L1 524288
UB 131072

[bandwidth]
bandwidth 60
```

`bandwidth` is the global total DDR bandwidth, used for the exclusive duration of DDR COPYs and for even splitting under concurrency. L1↔UB internal COPYs do not enter this shared pool. `capacity` is used jointly by Step2 and Step3: Step2 uses it for overflow detection and SPILL insertion, and Step3 uses it for allocation gating and peak statistics of on-chip tensors.

This algorithm does not read the remaining sections of `config.txt`; the per-problem wait cycles of Problems 1/2/3 and the Problem 3 Cache parameters are read by the multi-core evaluator — see [*Multi-Core Parallel Simulation Execution Algorithm*](多核并行模拟执行算法.md).

## 4. End-to-End Example

To make the three-step algorithm easy to verify, this section walks through a **miniature 4-op graph** by hand.

### 4.1 Input

```
ops:  1 (COPY_IN T1, PIPE_MTE2, 15cyc)
      2 (MATMUL a=T1, b=T2 → T3, PIPE_M, 200cyc)
      3 (RELU T3 → T4, PIPE_V, 30cyc)
      4 (COPY_OUT T4, PIPE_MTE3, 15cyc)
tensors: T1(DDR, 100), T2(DDR, 100), T3(L1, 100), T4(UB, 100)
edges: {1 → T1}, {T1 → 2}         (op 1 produces T1, consumed by op 2)
       {T2 → 2}                   (T2 already exists in DDR, no internal producer)
       {2 → T3}, {T3 → 3}         (op 2 produces T3, consumed by op 3)
       {3 → T4}, {T4 → 4}         (op 3 produces T4, consumed by op 4)
```

### 4.2 Step 1: Reverse DFS Topology

Out nodes = {4}. Stack simulation:

```
Stack: [4]                  → 4's predecessor 3 unvisited, push 3: [3, 4]
Stack: [3, 4]               → 3's predecessor 2 unvisited, push 2: [2, 3, 4]
Stack: [2, 3, 4]            → 2's predecessor 1 unvisited, push 1: [1, 2, 3, 4]
Stack: [1, 2, 3, 4]         → 1 has no predecessor, pop and visit: seq = [1]
Stack: [2, 3, 4]            → 2's predecessors (1) all visited, pop: seq = [1, 2]
Stack: [3, 4]               → 3's predecessors (2) all visited, pop: seq = [1, 2, 3]
Stack: [4]                  → 4's predecessors (3) all visited, pop: seq = [1, 2, 3, 4]
```

Output visitation order: `seq = [op1, op2, op3, op4]`

### 4.3 Step 2: Cache Swap-In/Swap-Out

**2.4.1 Lifecycle scan**:

| tensor | First occurrence (first) | Last occurrence (last) | Type |
|--------|-------------------|-------------------|------|
| T1 | 0 (op1, COPY_IN) | 1 (op2 input) | DDR (occupies no L1/UB) |
| T2 | 1 (op2 input) | 1 (op2 input) | DDR |
| T3 | 1 (op2 output) | 2 (op3 input) | L1 |
| T4 | 2 (op3 output) | 3 (op4 input) | UB |

The DDR type does not count against L1/UB capacity; the algorithm only tracks residency of the L1/UB types.

**2.4.2 Sequential traversal** (per `config.txt`: L1=524288, UB=131072; the tensors in this graph are tiny, far below the capacity limits):

```
t=0 (op1): involves T1 (DDR) — no effect on L1/UB residency
t=1 (op2): involves T2 (DDR, transient), T1 (DDR, released)
            involves T3 (L1): allocate +100 → L1 residency = 100
t=2 (op3): involves T3 (L1, released) → L1 residency = 0
            involves T4 (UB): allocate +100 → UB residency = 100
t=3 (op4): involves T4 (UB, released) → UB residency = 0
```

**No overflow**, no SPILL needed. SPILL records = empty.

### 4.4 Step 3: Pipelined Arrangement

```
FIFO per pipe:
  PIPE_MTE2: op1 (0..15)
  PIPE_M:    op2 (15..215)  # waits for op1 to finish (pred_end=15) and the MTE2 queue to empty (15)
  PIPE_V:    op3 (215..245) # waits for op2 to finish (215)
  PIPE_MTE3: op4 (245..260) # waits for op3 to finish (245)
```

Total execution time = 260 cycles.

### 4.5 Overflow and Eviction Criteria

Enlarging the tensor sizes in the example above is enough to trigger eviction. The decision procedure follows steps (b)(c) of 2.4; there are three key points:

1. **Overflow threshold**: at some step, after allocation completes and before execute/free, if $\text{resid}^{alloc}_T(t) > C_T$, record it in `overflow_log` and enter the eviction loop until residency is back within capacity.
2. **The victim must "not be in use right now"**: the candidate set excludes all tensors used at step $t$ (i.e., entries in `uses` that contain step $t$). Otherwise the swap-out node could only be placed after that op, and the op itself would still transiently exceed capacity.
3. **The victim must "still have future uses"**: tensors with $\text{next\_use} = \varnothing$ are skipped (they are about to be released; swapping them out is pointless). Among candidates, the one with the largest $\text{next\_use}$ is preferred — the Belady criterion: evict the one that will not be needed for the longest time in the future.

If the candidate set is empty (for example, the sum of the current op's inputs and outputs already exceeds capacity, or there is no tensor "still having future uses"), the algorithm does not silently clear active; instead it prints the step/op/type/resid/capacity/current_tids/active context and raises `Step2SchedulingError`, which the evaluation entry point converts into an `[EVALUATION ERROR]` exit.

After a successful eviction, the tensor's record looks like:

```
{ tid, size, prev_use_step, next_use_step, prev_use_op, next_use_op,
  spill_out_id, spill_in_id, spill_out_copies_data, from_tid, to_tid, backing_tid }
```

- `prev_use_step` / `next_use_step`: the swap-out is placed after the op corresponding to `prev_use_step`, and the swap-in before the op corresponding to `next_use_step`;
- `spill_out_copies_data = true` means this is the tensor's first write-back to DDR, producing both a `COPY_OUT` and a `COPY_IN` node plus a persistent backing; when `false`, the existing backing is reused and only a `COPY_IN` is produced.

### 4.6 Output

Reusing the $260$-cycle result from 4.4; if the example above experienced one eviction (T3's first write-back), the extended sequence and eviction record would look like:

```
Scheduling sequence: [op1, op2, op3, COPY_OUT(T3), ..., COPY_IN(T3), op5, ...]
Eviction record: [
  { tid: T3, size, prev_use_step, next_use_step,
    spill_out_copies_data: true, spill_out_id, spill_in_id, backing_tid },
]
Timetable: (provided by Step 3)
```

---

## 5. Summary of Algorithm Properties

| Property | Description |
|------|------|
| **Deterministic** | All three steps are deterministic; the same input always yields the same output (no randomness, no learning, no external parameters) |
| **Topologically correct** | Step 1 strictly guarantees the topological order; when Step 2 inserts SPILLs it adds edges along "predecessor→successor", preserving the original topology |
| **Capacity satisfied** | Step 2 performs overflow detection independently for each Type; in the output extended graph, each Type's residency ≤ capacity |
| **SPILL minimization (single cache)** | Step 2 selects victims with the Belady criterion, which is optimal for a single cache; treating multiple caches independently is an approximation |
| **No address allocation** | Step 2 only tracks usage amounts and does not output offsets; offsets are computed independently on the extended graph by the "cache allocation" step of Problem 2 in the original contest statement |
| **Usable as ground truth** | The total time complexity of the three steps is $O(|V| \log |V| + S)$; graphs at the scale of 40k nodes (Conv_Case1) run in milliseconds |

---