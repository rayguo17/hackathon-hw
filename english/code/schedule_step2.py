"""
schedule_step2.py — Step 2: cache spill-in/spill-out insertion (SPILL insertion)

Reference document: docs/intra_core_scheduling_algorithm.md § 2

Input interface:
    step2_spill_insertion(graph_json, seq, capacity):
        graph_json: dict containing 'ops' (id >= 1), 'tensors' (id >= 10001), 'edges' (op-tensor bipartite graph)
        seq: List[op_id], the visit order output by Step 1
        capacity: dict[Type -> int], from config.txt, containing only L1 and UB

Output:
    dict {
        'seq_ext': the extended sequence (including SPILL node ids),
        'spill_records': list of {spill_out_id?, spill_in_id, tid, from_tid,
            to_tid, pos, size, backing_tid, spill_out_copies_data, ...},
        'new_edges': list of (src, dst), dependency edges newly added to the extended graph,
        'removed_edges': logical tensor consumer edges replaced by renaming,
        'ext_edges': the complete edge set after renaming,
        'new_ops': list of {id, op, pipe, cycles}, SPILL nodes newly added to the extended graph,
        'new_tensors': persistent DDR backing and renamed on-chip tensors,
        'overflow_log': list of {step, type, resid, capacity}, SPILL trigger events,
        'capacity': the capacity configuration used this run,
    }

Invariants (guaranteed by Proposition 1.4 plus this step's overflow handling):
    1. Topological correctness: at any moment the resident amount of each Type <= that Type's capacity
    2. Data flow preserved: every tensor remains accessible to all original consumers while spilled (via the DDR trampoline)
    3. seq_ext is an extension of the original seq (original ops + inserted SPILL nodes)
    4. Each logical tensor has at most one SPILL_OUT DDR write-back; afterwards only COPY_IN is generated
"""

from contest_io import emit_error
from collections import defaultdict

class Step2SchedulingError(RuntimeError):
    """Raised when Step2 cannot complete an op's alloc within the hardware
    capacity."""

def _build_tensor_uses(graph_json, seq):
    """For each tensor, give a (step, op_id) list sorted by seq step"""
    n = len(seq)
    op_id_set = {o['id'] for o in graph_json['ops']}
    op_step = {op_id: i for i, op_id in enumerate(seq)}

    tensor_uses = defaultdict(list)
    for e in graph_json['edges']:
        if e['source'] in op_id_set and e['target'] not in op_id_set:
            tid, op = e['target'], e['source']
        elif e['source'] not in op_id_set and e['target'] in op_id_set:
            tid, op = e['source'], e['target']
        else:
            continue
        if op in op_step:
            tensor_uses[tid].append((op_step[op], op))
    for tid in tensor_uses:
        tensor_uses[tid].sort()
        # Deduplicate: multiple uses at the same step (several ops using the
        # same tensor at the same step) keep only one. Otherwise mid-use
        # handling misaligns: for example with uses=[(72),(79),(119),(119),(169)],
        # the first mid use at step 119 would wrongly set new_nu to
        # uses[3][0]=119 (it should be 169)
        new_list = []
        last_step = -1
        for (s, op) in tensor_uses[tid]:
            if s != last_step:
                new_list.append((s, op))
                last_step = s
        tensor_uses[tid] = new_list
    return tensor_uses, op_step, op_id_set

def _find_copy_in_backings(graph_json):
    """Find on-chip tensors copied in by COPY_IN from a unique DDR tensor."""
    op_by_id = {op['id']: op for op in graph_json['ops']}
    tensor_by_id = {tensor['id']: tensor for tensor in graph_json['tensors']}
    op_ids = set(op_by_id)
    in_tids = defaultdict(list)
    out_tids = defaultdict(list)
    for edge in graph_json['edges']:
        src, dst = edge['source'], edge['target']
        if src not in op_ids and dst in op_ids:
            in_tids[dst].append(src)
        elif src in op_ids and dst not in op_ids:
            out_tids[src].append(dst)

    candidates = defaultdict(set)
    for op_id, op in op_by_id.items():
        if op.get('op') != 'COPY_IN':
            continue
        ddr_inputs = [
            tid for tid in in_tids[op_id]
            if tid in tensor_by_id and tensor_by_id[tid].get('pos') == 'DDR'
        ]
        if len(ddr_inputs) != 1:
            continue
        for tid in out_tids[op_id]:
            if tid in tensor_by_id and tensor_by_id[tid].get('pos') != 'DDR':
                candidates[tid].add(ddr_inputs[0])
    return {
        tid: next(iter(backings))
        for tid, backings in candidates.items() if len(backings) == 1
    }

def step2_spill_insertion(graph_json, seq, capacity):
    """
    Belady SPILL insertion
    Input constraints: graph_json is a validated DAG; seq is complete and
    topologically correct; capacity is valid.
    Guarantee source: the contestant entry validates the graph and capacity,
    and Step1 generates a complete topological order by predecessor visits;
    the subgraph priority reordering of Problems 2/3 is checked separately
    for topological validity, so the above constraints are not re-validated
    here.
    capacity must be read from config.txt by the caller and passed in; this
    module no longer has built-in default capacity.
    """
    n = len(seq)
    tensor_uses, op_step, op_id_set = _build_tensor_uses(graph_json, seq)
    original_copy_in_backing = _find_copy_in_backings(graph_json)

    # Per-tensor lifecycle
    tensor_lifecycle = {}
    for t in graph_json['tensors']:
        tid = t['id']
        pos = t['pos']
        if pos == 'DDR':
            continue
        if tid not in tensor_uses:
            continue
        uses = tensor_uses[tid]
        tensor_lifecycle[tid] = {
            'pos': pos,
            'size': t['size'],
            'uses': uses,
            'first': uses[0][0],
            'last': uses[-1][0],
        }

    # Replace "scanning all tensor lifecycles at every step" with a one-time
    # event index. The table is built following tensor_lifecycle's insertion
    # order to preserve the original algorithm's processing order within a
    # step.
    uses_at_step = [[] for _ in range(n)]
    for tid, info in tensor_lifecycle.items():
        for use_idx, (step, _) in enumerate(info['uses']):
            uses_at_step[step].append((tid, use_idx))

    # id allocation
    max_id = max(o['id'] for o in graph_json['ops'])
    max_tid = max((t['id'] for t in graph_json['tensors']), default=0)
    next_id = max(max_id, max_tid) + 1

    # Simulation state
    # active[T][tid] = (next_use_step, used_count)
    # The dict provides O(1) lookup/removal while keeping the same insertion
    # order as the original list.
    type_active = {T: {} for T in capacity}
    type_resid = {T: 0 for T in capacity}
    # pending_spills: list of (out_after_step, in_before_step, tid, T, size, prev_use_op, next_use_op)
    pending_spills = []
    spill_in_at_step = defaultdict(list)
    overflow_log = []
    debug_over_count = []

    def get_resid(T):
        return type_resid[T]

    def trigger_one_spill(t, T, current_step_tids):
        """Trigger one Belady SPILL at step t; return (victim_tid, freed_size)
        or None. Excludes buffers currently being used at step t (i.e.,
        victim.uses contains a use at step t) — otherwise the victim is still
        being read by the op, SPILL_OUT would have to come after the op, and
        the op itself would transiently exceed capacity.
        """
        # Choose only among candidates whose next_use is not None
        # (i.e., they still have a future use)
        finite = [
            (nu, ui, tid)
            for tid, (nu, ui) in type_active[T].items()
            if nu is not None
        ]
        # Exclude buffers currently used at this step (a use in the uses list
        # has step == t)
        finite = [item for item in finite if item[2] not in current_step_tids]
        if not finite:
            return None
        finite.sort(key=lambda x: -x[0])  # max next_use first
        nu, ui, victim_tid = finite[0]
        info = tensor_lifecycle[victim_tid]
        v_size = info['size']

        # Basis for the prev_use / next_use values:
        #   active[T] stores (next_use, use_index), where use_index counts how
        #   many times the victim has already been used. The overflow check
        #   happens after step t's use processing, so ui already includes
        #   step t among the processed uses.
        #   Therefore: the next use = uses[ui] (exists and is not None, see the
        #   filtering above); the previous use = uses[ui-1] (when ui > 0),
        #   otherwise fall back to the first alloc step.
        prev_use_step = info['uses'][ui - 1][0] if ui > 0 else info['first']
        next_use_step = nu  # next use, guaranteed not None
        prev_use_op = info['uses'][ui - 1][1] if ui > 0 else None
        next_use_op = info['uses'][ui][1]  # uses[ui] is next use, op is index 1

        # Remove the victim from active
        type_active[T].pop(victim_tid)
        type_resid[T] -= v_size

        pending_spills.append({
            'out_after_step': prev_use_step,
            'in_before_step': next_use_step,
            'tid': victim_tid,
            'pos': T,
            'size': v_size,
            'prev_use_op': prev_use_op,
            'next_use_op': next_use_op,
        })
        spill_in_at_step[next_use_step].append((victim_tid, ui))
        return victim_tid, v_size

    # Main simulation loop
    for t in range(n):
        op = seq[t]

        current_step_tids = {T: set() for T in capacity}
        for tid, _ in uses_at_step[t]:
            current_step_tids[tensor_lifecycle[tid]['pos']].add(tid)

        # 0. Handle scheduled COPY_INs (physical spill-in before step t).
        # Re-alloc the victim into L1/UB, aligned with step t's op consumption.
        for tid, u_idx in spill_in_at_step.get(t, ()):
            info = tensor_lifecycle[tid]
            T = info['pos']
            uses = info['uses']
            new_nu = uses[u_idx + 1][0] if u_idx + 1 < len(uses) else None
            # Add only if the victim is not in type_active (avoid duplicates)
            if tid not in type_active[T]:
                type_active[T][tid] = (new_nu, u_idx)
                type_resid[T] += info['size']

        # 1. Handle this step's use alloc / next_use updates, but defer the
        #    last-use free. The physical semantics must be alloc -> capacity
        #    check/SPILL -> execute -> free; otherwise the transient peak of
        #    "output already allocated, last input not yet released" is missed.
        release_after_execute = []
        for tid, idx in uses_at_step[t]:
            info = tensor_lifecycle[tid]
            uses = info['uses']
            T = info['pos']
            if idx == 0:
                # First use (producer) = allocate memory
                next_nu = uses[1][0] if 1 < len(uses) else None
                type_active[T][tid] = (next_nu, 1)
                type_resid[T] += info['size']
            elif idx < len(uses) - 1:
                # Mid use: update next_use. Assigning to an existing key does not
                # change dict insertion order.
                new_nu = uses[idx + 1][0]
                active_item = type_active[T].get(tid)
                if active_item is not None:
                    _, used_count = active_item
                    type_active[T][tid] = (new_nu, used_count + 1)
            # A k=1 tensor is both first and last use: alloc first so it joins
            # the transient capacity check, then release after execution.
            # Ordinary last-use inputs are likewise deferred until after the
            # capacity check.
            if idx == len(uses) - 1:
                release_after_execute.append((T, tid))

        # 1.5 After alloc and before execute/free, run the unified Belady SPILL.
        #     trigger_one_spill excludes all tensors used by the current op, so
        #     it only evicts resident entries the current op does not need.
        #     The generated SPILL_OUT is anchored after the victim's last use
        #     and physically occurs before the current op in the extended
        #     sequence.
        for T, C in capacity.items():
            while get_resid(T) > C:
                resid_now = get_resid(T)
                overflow_log.append({'step': t, 'type': T, 'resid': resid_now, 'capacity': C})
                result = trigger_one_spill(t, T, current_step_tids[T])
                if result is None:
                    current_tids = sorted(current_step_tids[T].intersection(type_active[T]))
                    active_desc = sorted(
                        (tid, tensor_lifecycle[tid]['size'], nu)
                        for tid, (nu, _) in type_active[T].items()
                    )
                    message = (
                        '[STEP2 ERROR] no spill victim: step={step} op={op} type={type_} '
                        'alloc_resid={resid} capacity={capacity} current_tids={current} '
                        'active(tid,size,next_use)={active}'
                    ).format(
                        step=t, op=op, type_=T, resid=resid_now, capacity=C,
                        current=current_tids, active=active_desc,
                    )
                    emit_error(message)
                    raise Step2SchedulingError(message)

            # This is the current op's true post-alloc peak; both inputs and outputs
            # are still resident at this point.
            resid_after_alloc = get_resid(T)
            if resid_after_alloc > C:
                # Post-assertion for the while loop, preventing future edits from
                # reintroducing silent overflow.
                message = (
                    '[STEP2 ERROR] alloc peak still exceeds capacity after spill: '
                    'step={step} op={op} type={type_} resid={resid} capacity={capacity}'
                ).format(step=t, op=op, type_=T, resid=resid_after_alloc, capacity=C)
                emit_error(message)
                raise Step2SchedulingError(message)

        # 2. After the current op finishes executing, release this step's
        # last-use inputs/outputs.
        for T, tid in release_after_execute:
            if type_active[T].pop(tid, None) is not None:
                type_resid[T] -= tensor_lifecycle[tid]['size']

    # 3. Build the extended graph after renaming.
    # Every COPY_IN creates a new on-chip tensor incarnation; it is itself a
    # physical residency lifetime. Release-only COPY_OUTs are no longer
    # created; genuine DDR write-backs are still kept.
    new_ops_list = []
    new_tensors_list = []
    new_edges = []
    removed_edges = []
    spill_records = []

    # Collect insertion points
    insert_after = defaultdict(list)  # step -> list of real spill_out
    insert_before = defaultdict(list)

    backing_by_tid = dict(original_copy_in_backing)
    backing_origin = {
        tid: 'original_copy_in' for tid in original_copy_in_backing
    }
    current_incarnation = {
        tid: tid for tid in tensor_lifecycle
    }
    incarnation_version = defaultdict(int)
    spill_records_by_tid = defaultdict(list)

    for sp in pending_spills:
        logical_tid = sp['tid']
        from_tid = current_incarnation[logical_tid]
        spill_out_copies_data = sp['tid'] not in backing_by_tid
        spill_out_id = None
        if spill_out_copies_data:
            spill_out_id = next_id
            next_id += 1
        spill_in_id = next_id
        next_id += 1
        if spill_out_copies_data:
            backing_tid = next_id
            next_id += 1
            backing_by_tid[sp['tid']] = backing_tid
            backing_origin[sp['tid']] = 'spill_out'
            new_tensors_list.append({
                'id': backing_tid,
                'pos': 'DDR',
                'size': sp['size'],
            })
            backing_source = 'new_spill_out'
        else:
            backing_tid = backing_by_tid[sp['tid']]
            backing_source = (
                'original_copy_in'
                if backing_origin[sp['tid']] == 'original_copy_in'
                else 'reused_spill_out'
            )

        incarnation_version[logical_tid] += 1
        to_tid = next_id
        next_id += 1
        current_incarnation[logical_tid] = to_tid
        new_tensors_list.append({
            'id': to_tid,
            'logical_tid': logical_tid,
            'version': incarnation_version[logical_tid],
            'pos': sp['pos'],
            'size': sp['size'],
        })

        if spill_out_copies_data:
            new_ops_list.append({
                'id': spill_out_id,
                'op': 'COPY_OUT',
                'pipe': 'PIPE_MTE3',
                'cycles': max(1, sp['size'] // 64),
                'transfer_bytes': sp['size'],
                'spill_logical_tid': logical_tid,
            })
        new_ops_list.append({
            'id': spill_in_id,
            'op': 'COPY_IN',
            'pipe': 'PIPE_MTE2',
            'cycles': max(1, sp['size'] // 64),
            'transfer_bytes': sp['size'],
            'spill_logical_tid': logical_tid,
        })

        # Data edges directly express the production and consumption of the
        # physical incarnation. COPY_OUT is also a consumer of from_tid;
        # Step3's reference counting releases from_tid only after all
        # consumers finish.
        if spill_out_copies_data:
            new_edges.append((from_tid, spill_out_id))
            new_edges.append((spill_out_id, backing_tid))
        new_edges.append((backing_tid, spill_in_id))
        new_edges.append((spill_in_id, to_tid))

        if spill_out_id is not None:
            insert_after[sp['out_after_step']].append(spill_out_id)
        insert_before[sp['in_before_step']].append(spill_in_id)

        record = {
            'spill_out_id': spill_out_id,
            'spill_in_id': spill_in_id,
            'swap_tid': backing_tid,
            'backing_tid': backing_tid,
            'backing_source': backing_source,
            'spill_out_copies_data': spill_out_copies_data,
            'tid': logical_tid,
            'logical_tid': logical_tid,
            'from_tid': from_tid,
            'to_tid': to_tid,
            'version': incarnation_version[logical_tid],
            'pos': sp['pos'],
            'size': sp['size'],
            'prev_use_op': sp['prev_use_op'],
            'next_use_op': sp['next_use_op'],
            'prev_use_step': sp['out_after_step'],
            'next_use_step': sp['in_before_step'],
        }
        spill_records.append(record)
        spill_records_by_tid[logical_tid].append(record)

    # Reconnect each logical tensor -> consumer edge to the incarnation that
    # consumer belongs to. producer -> original tensor edges stay unchanged;
    # COPY_IN -> renamed tensor edges were already added above.
    op_step = {op_id: step for step, op_id in enumerate(seq)}
    consumer_incarnation = {}
    for logical_tid, records in spill_records_by_tid.items():
        record_index = 0
        physical_tid = logical_tid
        for step, consumer_op in tensor_uses[logical_tid]:
            while (record_index < len(records)
                   and step >= records[record_index]['next_use_step']):
                physical_tid = records[record_index]['to_tid']
                record_index += 1
            consumer_incarnation[(logical_tid, consumer_op)] = physical_tid
    rewired_edges = []
    for edge in graph_json['edges']:
        src, dst = edge['source'], edge['target']
        replacement = None
        if src in spill_records_by_tid and dst in op_step:
            physical_tid = consumer_incarnation.get((src, dst), src)
            if physical_tid != src:
                replacement = (physical_tid, dst)
        if replacement is None:
            rewired_edges.append(dict(edge))
        else:
            removed_edges.append((src, dst))
            rewired_edges.append({
                **edge,
                'source': replacement[0],
                'target': replacement[1],
            })

    ext_edges = rewired_edges + [
        {'source': src, 'target': dst, 'data_size': 0}
        for src, dst in new_edges
    ]

    # Build seq_ext
    seq_ext = []
    for i, op in enumerate(seq):
        # 'after (i-1)' event
        if i > 0:
            for sid in insert_after[i - 1]:
                seq_ext.append(sid)
        # 'before i' event
        for sid in insert_before[i]:
            seq_ext.append(sid)
        seq_ext.append(op)
    # 'after (n-1)' event (after the loop ends)
    for sid in insert_after[n - 1]:
        seq_ext.append(sid)

    return {
        'seq_ext': seq_ext,
        'spill_records': spill_records,
        'new_edges': new_edges,
        'removed_edges': removed_edges,
        'ext_edges': ext_edges,
        'new_ops': new_ops_list,
        'new_tensors': new_tensors_list,
        'overflow_log': overflow_log,
        'capacity': capacity,
        '_debug_over_count': debug_over_count,
    }


# ==================== CLI entry ====================
def _build_extended_graph(graph_json, result):
    """Assemble the Step2 output without re-validating.

    Input constraint: result comes from step2_spill_insertion on the same
    original graph. Step2 allocates new IDs uniformly and inserts SPILLs at
    the original topological visit points, so ext_edges matches seq_ext with
    complete coverage. This function changes neither order nor dependencies;
    it only assembles these two related outputs.
    """
    ops = list(graph_json['ops'])
    ops.extend(result['new_ops'])
    tensors = list(graph_json['tensors'])
    tensors.extend(result['new_tensors'])
    edges = list(result.get('ext_edges', graph_json['edges']))
    return {
        'ops': ops,
        'tensors': tensors,
        'edges': edges,
        'seq_ext': result['seq_ext'],
    }

def _simulate_occupancy(ext_graph, result, seq_orig):
    """Replay seq_ext residency using producer/consumer reference counts of
    renamed tensors."""
    del result, seq_orig
    ops = ext_graph['ops']
    tensors = ext_graph['tensors']
    edges = ext_graph['edges']
    seq = ext_graph['seq_ext']
    n = len(seq)
    tensor_by_id = {t['id']: t for t in tensors}
    op_id_set = {o['id'] for o in ops}
    in_tids = {o['id']: [] for o in ops}
    out_tids = {o['id']: [] for o in ops}
    for e in edges:
        if e['source'] in op_id_set and e['target'] not in op_id_set:
            out_tids[e['source']].append(e['target'])
        elif e['target'] in op_id_set and e['source'] not in op_id_set:
            in_tids[e['target']].append(e['source'])

    consumer_count = defaultdict(int)
    producers = set()
    for op_id in out_tids:
        producers.update(out_tids[op_id])
    for op_id, tids in in_tids.items():
        for tid in set(tids):
            consumer_count[tid] += 1

    l1_occ_alloc = [0] * n
    ub_occ_alloc = [0] * n
    l1_occ = [0] * n
    ub_occ = [0] * n
    events_per_step = [[] for _ in range(n)]
    active = set()
    used = {'L1': 0, 'UB': 0}

    # On-chip graph inputs without a producer are resident before step 0.
    for tid, count in consumer_count.items():
        tensor = tensor_by_id.get(tid)
        if (count and tid not in producers and tensor is not None
                and tensor.get('pos') in used):
            active.add(tid)
            used[tensor['pos']] += tensor['size']
            if n:
                events_per_step[0].append(
                    (tid, tensor['pos'], tensor['size'], 'initial_alloc'))

    for i, op_id in enumerate(seq):
        for tid in set(out_tids[op_id]):
            tensor = tensor_by_id.get(tid)
            if tensor is not None and tensor.get('pos') in used and tid not in active:
                active.add(tid)
                used[tensor['pos']] += tensor['size']
                events_per_step[i].append(
                    (tid, tensor['pos'], tensor['size'], 'alloc'))

        l1_occ_alloc[i], ub_occ_alloc[i] = used['L1'], used['UB']

        for tid in set(in_tids[op_id]):
            if tid not in consumer_count:
                continue
            consumer_count[tid] -= 1
            if consumer_count[tid] == 0 and tid in active:
                tensor = tensor_by_id[tid]
                active.remove(tid)
                used[tensor['pos']] -= tensor['size']
                events_per_step[i].append(
                    (tid, tensor['pos'], tensor['size'], 'free'))
        for tid in set(out_tids[op_id]):
            if consumer_count.get(tid, 0) == 0 and tid in active:
                tensor = tensor_by_id[tid]
                active.remove(tid)
                used[tensor['pos']] -= tensor['size']
                events_per_step[i].append(
                    (tid, tensor['pos'], tensor['size'], 'free'))

        l1_occ[i], ub_occ[i] = used['L1'], used['UB']

    l1_peak = max(l1_occ_alloc) if l1_occ_alloc else 0
    ub_peak = max(ub_occ_alloc) if ub_occ_alloc else 0
    l1_peak_idx = l1_occ_alloc.index(l1_peak) if l1_peak > 0 else -1
    ub_peak_idx = ub_occ_alloc.index(ub_peak) if ub_peak > 0 else -1
    return l1_occ_alloc, ub_occ_alloc, l1_occ, ub_occ, events_per_step, l1_peak, l1_peak_idx, ub_peak, ub_peak_idx
