"""
schedule_step1.py — Step 1: multi-source reverse DFS topological sort

Reference document: docs/intra_core_scheduling_algorithm.md § 1

Input interface:
    step1_schedule(graph_json):
        graph_json: dict containing 'ops' (with 'id', 'op', 'pipe', 'cycles') and 'edges' (with 'source', 'target')
                    optional 'tensors' (needed for the bipartite graph); without tensors, edges are assumed to be direct op-op edges

    step1_from_adj(pred_map, succ_map, V, ops=None):
        pred_map: Dict[v_id -> Set[pred_id]]
        succ_map: Dict[v_id -> Set[succ_id]]
        V: List[v_id] node set
        ops: List[{id, op, ...}] optional, provides the op field used to decide is_copy_in / is_copy_out

Output:
    seq: List[v_id] schedule sequence (visit order = physical execution order)

Invariants (guaranteed by Proposition 1.4):
    1. Complete coverage: len(seq) == len(V), seq is a bijection of V
    2. Topological correctness: ∀ (u,v) ∈ E: pos(u) < pos(v)
    3. Stack-bottom invariant: seq[-1] = argmin_{v ∈ V_out} start_key(v) (V_out = {v | succ(v) = ∅})
    4. Determinism: the same input always yields the same output
"""

from collections import deque



# ==================== Step 1.0: depth precomputation ====================

def precompute_depth(pred_map, succ_map, V):
    """
    Step 1.0: node depth precomputation

    depth(v) = 0                                 if pred(v) = ∅
             = 1 + max_{p ∈ pred(v)} depth(p)   otherwise

    Complexity: O(|V| + |E|)

    Input:
        pred_map: dict[v_id -> set of pred_id]
        succ_map: dict[v_id -> set of succ_id]
        V: list of v_id
    Output:
        depth: dict[v_id -> int]
    Constraint: the adjacency table is a DAG obtained by contracting the
    validated original graph; step1_from_adj passes it in as-is, so cycle
    detection is not repeated here. Callers must not pass in arbitrary
    adjacency tables that have not been entry-validated.
    """
    depth = {}
    queue = deque()
    for v in V:
        if len(pred_map[v]) == 0:
            depth[v] = 0
            queue.append(v)
    while queue:
        u = queue.popleft()
        for w in succ_map[u]:
            new_depth = depth[u] + 1
            if w not in depth or new_depth > depth[w]:
                depth[w] = new_depth
                queue.append(w)
    return depth


# ==================== Section 0.5: bipartite graph -> op-op adjacency table ====================

def bipartite_to_op_adj(graph_json):
    """
    Convert the attachment compute graph's op-tensor bipartite graph into an
    op-op adjacency table.

    Input: graph_json contains 'ops', 'tensors', 'edges'; both node ID classes are unique and non-negative
    Output: the (pred_map, succ_map, V) triple
    Constraint: graph IDs are unique, endpoints exist, and there are no cycles;
    the contestant entry validate_graph already checked this. The conversion
    keeps all tensor producers and direct op edges, so the adjacency table is
    equivalent to the original dependencies.
    """
    ops = graph_json['ops']
    V = [o['id'] for o in ops]
    op_id_set = set(V)

    # tensor -> producer op
    tensor_producer = {}
    for e in graph_json['edges']:
        if e['source'] in op_id_set and e['target'] not in op_id_set:
            tensor_producer.setdefault(e['target'], set()).add(e['source'])

    pred_map = {v: set() for v in V}
    succ_map = {v: set() for v in V}
    for e in graph_json['edges']:
        if e['source'] in op_id_set and e['target'] in op_id_set:
            pred_map[e['target']].add(e['source'])
            succ_map[e['source']].add(e['target'])
        if e['source'] not in op_id_set and e['target'] in op_id_set:
            for producer in tensor_producer.get(e['source'], ()):
                pred_map[e['target']].add(producer)
                succ_map[producer].add(e['target'])

    return pred_map, succ_map, V


# ==================== Step 1 main algorithm ====================

def step1_from_adj(pred_map, succ_map, V, ops=None):
    """
    Step 1: multi-source reverse DFS topological sort (core algorithm)

    Reference: Section 1.3, Step 1.0 / 1.A / 1.B / 1.C

    Sort keys:
        key(p)       = (¬is_copy_in(p),  depth(p), -id(p))   # main loop: expand multiple predecessors
        start_key(p) = (¬is_copy_out(p), depth(p), -id(p))   # start: push multiple sources

    Push in ascending order -> pop LIFO -> larger keys are visited first
    (executed earlier in physical time)

    Input:
        pred_map: dict[v_id -> set of pred_id]
        succ_map: dict[v_id -> set of succ_id]
        V: list of v_id
        ops: optional, list of {id, op, ...}  # provides the op field used to decide is_copy_in / is_copy_out
    Output:
        seq: list of v_id, the schedule sequence
    """
    # Input constraints: V is unique, predecessors/successors mirror each
    # other, and there are no cycles. The entry already validated the original
    # graph; bipartite_to_op_adj adds dependencies in pairs and contracting
    # tensors creates no new cycles, so no repeated checks.
    if not V:
        return []
    # ---------- Step 1.0 depth precomputation ----------
    depth = precompute_depth(pred_map, succ_map, V)

    # Decision helpers
    is_copy_in = {}
    is_copy_out = {}
    if ops is not None:
        is_copy_in = {o['id']: o['op'] == 'COPY_IN' for o in ops}
        is_copy_out = {o['id']: o['op'] == 'COPY_OUT' for o in ops}
    else:
        is_copy_in = {v: False for v in V}
        is_copy_out = {v: False for v in V}

    def key(v):
        return (not is_copy_in[v], depth[v], -v)

    def start_key(v):
        return (not is_copy_out[v], depth[v], -v)

    # ---------- Step 1.A multi-source start ----------
    out_nodes = [v for v in V if len(succ_map[v]) == 0]
    if not out_nodes:
        # Degenerate: no out nodes (cycle or isolated graph) -> fall back to
        # the smallest id
        out_nodes = [min(V)]
    sorted_starts = sorted(out_nodes, key=start_key)
    # Push in ascending order: smaller keys enter the stack first -> stack
    # bottom; larger keys enter later -> stack top
    stack = list(sorted_starts)

    visited = set()
    seq = []

    # ---------- Step 1.B main loop ----------
    while stack or len(visited) < len(V):
        if not stack:
            # Stack empty (all sources drained) -> handle the remainder
            remaining = [v for v in V if v not in visited]
            if not remaining:
                break
            for v in sorted(remaining, key=key):
                stack.append(v)
            continue

        u = stack[-1]
        if u in visited:
            stack.pop()
            continue

        unvisited_preds = [p for p in pred_map[u] if p not in visited]
        if unvisited_preds:
            for p in sorted(unvisited_preds, key=key):
                stack.append(p)
        else:
            visited.add(u)
            seq.append(u)
            stack.pop()

    return seq


def step1_schedule(graph_json):
    """
    Step 1 entry: accepts graph_json in the attachment compute graph format
    and returns the schedule sequence.

    Internally: Section 0.5 preprocessing -> Step 1 main algorithm
    Input constraint: the graph format is correct and it is a DAG. Guarantee
    source: the contestant entry already validated the original graph; Task
    construction keeps local dependencies and replaces cross-core edges with
    source/sink COPYs, introducing no local cycle.
    This function is an internal algorithm interface and does not re-scan
    the input.
    """
    pred_map, succ_map, V = bipartite_to_op_adj(graph_json)
    ops = graph_json['ops']
    return step1_from_adj(pred_map, succ_map, V, ops)


# ==================== CLI ====================


def _check_topo(graph_json, seq):
    """Topological correctness check"""
    op_id_set = {o['id'] for o in graph_json['ops']}
    pred_map, succ_map, _ = bipartite_to_op_adj(graph_json)
    pos = {v: i for i, v in enumerate(seq)}
    for u in seq:
        for v in succ_map[u]:
            if pos[u] > pos[v]:
                return False
    return True
