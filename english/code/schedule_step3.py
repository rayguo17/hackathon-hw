"""
schedule_step3.py — Step 3: out-of-order pipeline arrangement
(out-of-order scheduling)

Reference document: docs/intra_core_scheduling_algorithm.md § 3

Responsibility boundaries:
    - Step2 handles on-chip memory capacity, spill-in/spill-out nodes, and
      the extended graph;
    - Step3 only reads the extended graph and arranges timing by dependencies
      and the 4 pipes;
    - all COPYs touching DDR share the total bandwidth; L1<->UB COPYs use
      internal bandwidth;
    - Step3 performs alloc/free and capacity gating by consumer reference
      counts of renamed tensors;
    - no concrete addresses are allocated. Capacity is abstracted into
      divisible virtual byte credits;
    - when credits are reused, a synchronization dependency is added between
      all readers of the old tensor and the new producer, compensating the
      WAR/WAW false data dependencies that real address reuse would create;
    - outputs the deterministic per-Pipe order and the edge-augmented
      execution graph for direct use by the three multi-core evaluators.

If ops remain unfinished while no runnable/running op exists, the extended
graph or the scheduler has an error. In that case a full diagnostic is
printed and Step3SchedulingError is raised; the CLI also writes the
diagnostic to the corresponding log.
"""

import heapq
import math
from copy import deepcopy
from collections import Counter, defaultdict

from contest_io import emit_error

# Hardware model: four intra-core instruction Pipes, each with at most 1
# in-flight instruction at any moment. Both are fixed constraints rather than
# contestant-tunable parameters, so they are hard-coded and not accepted from
# config.txt.
PIPES = ('PIPE_MTE2', 'PIPE_MTE3', 'PIPE_M', 'PIPE_V')
PIPE_SLOTS = 1

class Step3SchedulingError(RuntimeError):
    """Raised when the extended graph cannot issue further but unfinished ops
    remain."""

def op_pipe(op):
    """The Pipe is taken directly from the node's own pipe field, not guessed
    from the op name."""
    return op['pipe']

def _build_graph_views(ops, edges):
    """Uniformly parse tensor-mediated edges and direct op-op edges; DDR
    tensors participate in dependencies too."""
    op_ids = {o['id'] for o in ops}
    in_tids = {o['id']: [] for o in ops}
    out_tids = {o['id']: [] for o in ops}
    direct_edges = []
    for edge in edges:
        src, dst = edge['source'], edge['target']
        src_is_op = src in op_ids
        dst_is_op = dst in op_ids
        if src_is_op and not dst_is_op:
            out_tids[src].append(dst)
        elif not src_is_op and dst_is_op:
            in_tids[dst].append(src)
        elif src_is_op and dst_is_op and src != dst:
            direct_edges.append((src, dst))

    producers = defaultdict(set)
    for op_id, tids in out_tids.items():
        for tid in tids:
            producers[tid].add(op_id)

    preds = {o['id']: set() for o in ops}
    succs = {o['id']: set() for o in ops}
    for op_id, tids in in_tids.items():
        for tid in tids:
            for producer in producers.get(tid, ()):
                if producer != op_id:
                    preds[op_id].add(producer)
                    succs[producer].add(op_id)
    for src, dst in direct_edges:
        preds[dst].add(src)
        succs[src].add(dst)
    return in_tids, out_tids, preds, succs

def _op_duration(op, in_tids, out_tids, tensor_by_id, bandwidth):
    """Regular ops use cycles; COPY ops are converted from the moved tensor
    size when available."""
    op_type = op['op']
    if op_type in ('COPY_IN', 'COPY_OUT'):
        tids = out_tids[op['id']] if op_type == 'COPY_IN' else in_tids[op['id']]
        sizes = [tensor_by_id[tid]['size'] for tid in tids if tid in tensor_by_id]
        if sizes:
            return max(1, math.ceil(sum(sizes) / bandwidth))
    return max(1, op.get('cycles', 1))

def _uses_ddr_bandwidth(op, in_tids, out_tids, tensor_by_id):
    """Only COPYs with a DDR endpoint enter the shared DDR bandwidth pool."""
    if op['op'] not in ('COPY_IN', 'COPY_OUT'):
        return False
    tids = in_tids[op['id']] + out_tids[op['id']]
    return any(
        tid in tensor_by_id and tensor_by_id[tid].get('pos') == 'DDR'
        for tid in tids
    )

def step3_simulation(ext_graph, capacity, bandwidth, max_iter=100000):
    """Schedule the extended graph by dependencies, pipes, DDR bandwidth, and
    renamed tensor lifecycles.

    Input constraints: seq_ext is a complete topological order of all ops in
    the extended graph; IDs/endpoints are valid; capacity and bandwidth are
    valid values already checked at the entry.
    Guarantee source: Step2 inserts SPILLs along Step1's topological order and
    allocates new IDs, constructing ext_edges and seq_ext together after
    renaming; _build_extended_graph assembles this output as-is.
    The input is not re-checked; output contract checks for the memory
    dependencies and execution order newly generated in this phase are kept.
    capacity and bandwidth are read from config.txt by the caller and passed
    in; this function no longer has built-in default configuration. The
    hardware model limits each Pipe to PIPE_SLOTS in-flight instructions at
    any moment.
    """
    capacity = dict(capacity)

    ops = ext_graph['ops']
    tensors = ext_graph['tensors']
    edges = ext_graph['edges']
    seq = ext_graph['seq_ext']
    op_by_id = {o['id']: o for o in ops}
    tensor_by_id = {t['id']: t for t in tensors}

    in_tids, out_tids, preds, succs = _build_graph_views(ops, edges)
    tensor_producers = defaultdict(set)
    tensor_consumers = defaultdict(set)
    for op_id, tids in out_tids.items():
        for tid in tids:
            tensor_producers[tid].add(op_id)
    for op_id, tids in in_tids.items():
        for tid in tids:
            tensor_consumers[tid].add(op_id)

    # Each renamed on-chip tensor is one physical residency lifetime. It is
    # allocated before the producing op issues and freed after all consumers
    # finish; a genuine COPY_OUT is also an ordinary consumer.
    managed_tensors = {
        tid for tid, tensor in tensor_by_id.items()
        if tensor.get('pos') in capacity
    }
    remaining_consumers = {
        tid: len(tensor_consumers.get(tid, ())) for tid in managed_tensors
    }
    resident_tensors = set()
    memory_used = {pos: 0 for pos in capacity}
    memory_peak = {pos: 0 for pos in capacity}
    memory_events = []
    # free_credits are "reusable capacity" holding only a byte count and the
    # synchronization sources of the previous tensor, with no offset. Credits
    # may be split/merged, which is equivalent to performing one cost-free
    # memory compaction before execution and introduces no address
    # fragmentation problem.
    free_credits = {pos: [] for pos in capacity}
    memory_dependency_parts = defaultdict(lambda: {
        'bytes': 0, 'tensor_ids': set(), 'kinds': set(), 'positions': set()})

    def add_free_credit(tid):
        tensor = tensor_by_id[tid]
        consumers = sorted(tensor_consumers.get(tid, ()))
        producers = sorted(tensor_producers.get(tid, ()))
        # With readers, it must wait for all readers to finish (WAR); a dead
        # output with no readers must at least wait for the old write to
        # finish before reusing the same credit (WAW).
        sources = consumers if consumers else producers
        free_credits[tensor['pos']].append({
            'bytes': tensor['size'],
            'sources': tuple(sources),
            'tid': tid,
            'kind': 'WAR' if consumers else 'WAW',
        })

    def consume_free_credit(tid, producer_op):
        tensor = tensor_by_id[tid]
        pos = tensor['pos']
        remaining = tensor['size']
        credits = free_credits[pos]
        while remaining > 0 and credits:
            credit = credits[0]
            taken = min(remaining, credit['bytes'])
            if credit['sources']:
                for source_op in credit['sources']:
                    if source_op == producer_op:
                        continue
                    part = memory_dependency_parts[(source_op, producer_op)]
                    part['bytes'] += taken
                    part['tensor_ids'].add(credit['tid'])
                    part['kinds'].add(credit['kind'])
                    part['positions'].add(pos)
            credit['bytes'] -= taken
            remaining -= taken
            if credit['bytes'] == 0:
                credits.pop(0)
        if remaining:
            message = '[STEP3 ERROR] virtual capacity credits exhausted: tid={} missing={}'.format(
                tid, remaining)
            emit_error(message)
            raise Step3SchedulingError(message)

    def allocate_tensor(tid, now, producer_op=None, kind='alloc'):
        if tid not in managed_tensors or tid in resident_tensors:
            return
        tensor = tensor_by_id[tid]
        pos, size = tensor['pos'], tensor['size']
        if producer_op is not None:
            consume_free_credit(tid, producer_op)
        memory_used[pos] += size
        resident_tensors.add(tid)
        memory_peak[pos] = max(memory_peak[pos], memory_used[pos])
        memory_events.append({
            'time': now, 'kind': kind, 'tid': tid,
            'logical_tid': tensor.get('logical_tid', tid),
            'pos': pos, 'size': size, 'op_id': producer_op,
            'used_after': memory_used[pos],
        })

    def release_tensor(tid, now, consumer_op=None):
        if tid not in resident_tensors:
            return
        tensor = tensor_by_id[tid]
        pos, size = tensor['pos'], tensor['size']
        resident_tensors.remove(tid)
        memory_used[pos] -= size
        add_free_credit(tid)
        memory_events.append({
            'time': now, 'kind': 'free', 'tid': tid,
            'logical_tid': tensor.get('logical_tid', tid),
            'pos': pos, 'size': size, 'op_id': consumer_op,
            'used_after': memory_used[pos],
        })

    # Tensors without an on-chip producer are considered resident before the
    # graph starts.
    for tid in sorted(managed_tensors):
        if not tensor_producers.get(tid) and tensor_consumers.get(tid):
            allocate_tensor(tid, 0, kind='initial_alloc')
    initial_overflow = {
        pos: memory_used[pos] for pos in capacity
        if memory_used[pos] > capacity[pos]
    }
    if initial_overflow:
        message = (
            '[STEP3 ERROR] initial resident tensors exceed capacity: used={} capacity={}'
        ).format(initial_overflow, capacity)
        emit_error(message)
        raise Step3SchedulingError(message)
    for pos in capacity:
        unused = capacity[pos] - memory_used[pos]
        if unused:
            free_credits[pos].append({
                'bytes': unused, 'sources': (), 'tid': None,
                'kind': 'VIRGIN',
            })

    def allocation_requirement(op_id):
        required = {pos: 0 for pos in capacity}
        tids = []
        for tid in set(out_tids[op_id]):
            if tid not in managed_tensors or tid in resident_tensors:
                continue
            tensor = tensor_by_id[tid]
            required[tensor['pos']] += tensor['size']
            tids.append(tid)
        return required, sorted(tids)

    def can_allocate_outputs(op_id):
        required, _ = allocation_requirement(op_id)
        return all(
            memory_used[pos] + required[pos] <= capacity[pos]
            for pos in capacity
        )

    def allocate_outputs(op_id, now):
        for tid in sorted(set(out_tids[op_id])):
            allocate_tensor(tid, now, producer_op=op_id)

    def consume_inputs(op_id, now):
        for tid in set(in_tids[op_id]):
            if tid not in remaining_consumers:
                continue
            if op_id not in tensor_consumers.get(tid, ()):
                continue
            remaining_consumers[tid] -= 1
            if remaining_consumers[tid] < 0:
                message = '[STEP3 ERROR] negative tensor refcount: tid={} op={}'.format(
                    tid, op_id)
                emit_error(message)
                raise Step3SchedulingError(message)
            if remaining_consumers[tid] == 0:
                release_tensor(tid, now, consumer_op=op_id)

    def release_dead_outputs(op_id, now):
        for tid in set(out_tids[op_id]):
            if tid in remaining_consumers and remaining_consumers[tid] == 0:
                release_tensor(tid, now, consumer_op=op_id)

    seq_pos = {op_id: i for i, op_id in enumerate(seq)}
    # seq_ext is the deterministic topological order already validated by
    # Step2. Step3 uses its projection on each Pipe as the fixed issue order;
    # this way, after the multi-core phase adds cross-core COPY waits, the
    # order stays consistent with the intra-core scheduling result and Pipes
    # are not rearranged due to differing local ready times.
    planned_pipe_orders = {pipe: [] for pipe in PIPES}
    for op_id in seq:
        planned_pipe_orders[op_pipe(op_by_id[op_id])].append(op_id)
    pipe_cursor = {pipe: 0 for pipe in PIPES}
    allocation_order = [
        op_id for op_id in seq
        if any(tid in managed_tensors for tid in out_tids[op_id])
    ]
    allocation_rank = {
        op_id: rank for rank, op_id in enumerate(allocation_order)
    }
    next_allocation_rank = 0
    op_status = {op_id: 'pending' for op_id in op_by_id}
    pred_remaining = {op_id: len(preds[op_id]) for op_id in op_by_id}
    op_start, op_end = {}, {}
    issue_queues = {pipe: [] for pipe in PIPES}
    allocation_ready = set()
    executors = {pipe: [] for pipe in PIPES}
    ddr_transfer = {
        op_id: _uses_ddr_bandwidth(op, in_tids, out_tids, tensor_by_id)
        for op_id, op in op_by_id.items()
    }

    # DDR fair-sharing state. remaining_work is measured in "how many cycles
    # are still needed if the transfer had all DDR bandwidth to itself".
    # After the n-th new transfer joins, the existing transfers' bandwidth
    # share drops from 1/(n-1) to 1/n, and the remaining time lengthens by
    # n/(n-1). Below, sorting by remaining_work and repeatedly removing the
    # earliest-finishing item equivalently implements the specified recurrence
    # and naturally supports the future multi-core 2N MTE pipes.
    ddr_remaining_work = {}
    ddr_last_update = 0.0
    ddr_contention_log = []

    def advance_ddr_work(now):
        nonlocal ddr_last_update
        elapsed = now - ddr_last_update
        # The projected end is rounded up to whole cycles. If a transfer finishes
        # inside a cycle, its remaining fractional time must be shared by the
        # other transfers immediately; a finished item must not keep occupying
        # bandwidth.
        while elapsed > 1e-9:
            active = [
                op_id for op_id, work in ddr_remaining_work.items()
                if work > 1e-9
            ]
            if not active:
                break
            min_work = min(ddr_remaining_work[op_id] for op_id in active)
            time_to_first_finish = min_work * len(active)
            if time_to_first_finish >= elapsed - 1e-9:
                share = elapsed / len(active)
                for op_id in active:
                    ddr_remaining_work[op_id] = max(
                        0.0, ddr_remaining_work[op_id] - share)
                break
            for op_id in active:
                ddr_remaining_work[op_id] = max(
                    0.0, ddr_remaining_work[op_id] - min_work)
            elapsed -= time_to_first_finish
        ddr_last_update = now

    def reschedule_ddr_ends(now):
        """Remove the earliest item one by one, computing provisional ends,
        contention penalties, and final ends."""
        if not ddr_remaining_work:
            return {}, {}, []
        ordered = sorted(
            (max(0.0, work), op_id)
            for op_id, work in ddr_remaining_work.items()
        )
        projected = {}
        provisional = {}
        stages = []
        base_cursor = float(now)
        adjusted_cursor = float(now)
        previous_work = 0.0
        active_count = len(ordered)
        i = 0
        while i < len(ordered):
            work = ordered[i][0]
            delta_work = work - previous_work
            if active_count > 1:
                # New ops join one at a time: first get the provisional interval t
                # on the n-1-way timeline before joining, then stretch that
                # interval by n/(n-1). The extra penalty is adjusted - base.
                base_delta = delta_work * (active_count - 1)
                adjusted_delta = base_delta * active_count / (active_count - 1)
                factor = active_count / (active_count - 1)
            else:
                base_delta = delta_work
                adjusted_delta = delta_work
                factor = 1.0
            base_start = base_cursor
            adjusted_start = adjusted_cursor
            base_cursor += base_delta
            adjusted_cursor += adjusted_delta
            j = i
            while j < len(ordered) and abs(ordered[j][0] - work) <= 1e-9:
                provisional[ordered[j][1]] = base_cursor
                # The scheduling time unit is a cycle; piecewise accounting allows
                # fractions, but the final end written back must be rounded up
                # to avoid meaningless float timestamps.
                projected[ordered[j][1]] = int(math.ceil(adjusted_cursor - 1e-9))
                j += 1
            stages.append({
                'active_ops': sorted(op_id for _, op_id in ordered[i:]),
                'n': active_count,
                'base_start': base_start,
                'base_end': base_cursor,
                'adjusted_start': adjusted_start,
                'adjusted_before_penalty_end': adjusted_start + base_delta,
                'adjusted_end': adjusted_cursor,
                'factor': factor,
                'penalty': adjusted_delta - base_delta,
                'completed_ops': sorted(ordered[k][1] for k in range(i, j)),
            })
            active_count -= j - i
            previous_work = work
            i = j

        for op_id, end in projected.items():
            op_end[op_id] = end
        for pipe in executors:
            executors[pipe] = [
                (op_id, projected.get(op_id, end))
                for op_id, end in executors[pipe]
            ]
        return projected, provisional, stages

    def queue_if_ready(op_id):
        if op_status[op_id] == 'pending' and pred_remaining[op_id] == 0:
            pipe = op_pipe(op_by_id[op_id])
            if pipe not in issue_queues:
                message = '[STEP3 ERROR] missing pipe capacity: op={} pipe={}'.format(op_id, pipe)
                emit_error(message)
                raise Step3SchedulingError(message)
            order = planned_pipe_orders[pipe]
            cursor = pipe_cursor[pipe]
            if cursor >= len(order) or order[cursor] != op_id:
                return
            op_status[op_id] = 'ready'
            if op_id in allocation_rank:
                allocation_ready.add(op_id)
            else:
                heapq.heappush(issue_queues[pipe], (seq_pos[op_id], op_id))

    for op_id in seq:
        queue_if_ready(op_id)

    t_now = 0
    iteration = 0

    def retire_step():
        advance_ddr_work(t_now)
        retired_ddr = False
        for pipe in executors:
            still_running = []
            for op_id, end in executors[pipe]:
                if end <= t_now + 1e-9:
                    op_status[op_id] = 'done'
                    consume_inputs(op_id, t_now)
                    release_dead_outputs(op_id, t_now)
                    completed_pipe = op_pipe(op_by_id[op_id])
                    completed_cursor = pipe_cursor[completed_pipe]
                    pipe_cursor[completed_pipe] += 1
                    completed_order = planned_pipe_orders[completed_pipe]
                    if completed_cursor + 1 < len(completed_order):
                        queue_if_ready(completed_order[completed_cursor + 1])
                    if op_id in ddr_remaining_work:
                        ddr_remaining_work.pop(op_id, None)
                        retired_ddr = True
                    for succ_id in succs[op_id]:
                        pred_remaining[succ_id] -= 1
                        queue_if_ready(succ_id)
                else:
                    still_running.append((op_id, end))
            executors[pipe] = still_running
        if retired_ddr:
            reschedule_ddr_ends(t_now)

    def issue_step():
        nonlocal next_allocation_rank
        # Issuing is handled strictly one at a time. Even when multiple ops share
        # the same t_now, the current op's join, provisional end computation,
        # and the update of all ends complete before the next op is popped;
        # there is no batch simultaneous joining.
        issued_in_pass = True
        while issued_in_pass:
            issued_in_pass = False
            for pipe in PIPES:
                while len(executors[pipe]) < PIPE_SLOTS:
                    # All on-chip allocations strictly follow Step2's seq_ext order;
                    # execution may still overlap across pipes. This prevents
                    # later branches from filling memory early and forming a
                    # resource deadlock.
                    op_id = None
                    if next_allocation_rank < len(allocation_order):
                        candidate = allocation_order[next_allocation_rank]
                        candidate_pipe = op_pipe(op_by_id[candidate])
                        if (candidate in allocation_ready
                                and candidate_pipe == pipe
                                and can_allocate_outputs(candidate)):
                            allocation_ready.remove(candidate)
                            op_id = candidate
                    if op_id is None and issue_queues[pipe]:
                        _, op_id = heapq.heappop(issue_queues[pipe])
                    if op_id is None:
                        break
                    issued_in_pass = True
                    rank = allocation_rank.get(op_id)
                    if rank is not None:
                        next_allocation_rank += 1
                    op_status[op_id] = 'running'
                    allocate_outputs(op_id, t_now)
                    duration = _op_duration(
                        op_by_id[op_id], in_tids, out_tids,
                        tensor_by_id, bandwidth)
                    op_start[op_id] = t_now
                    op_end[op_id] = t_now + duration
                    executors[pipe].append((op_id, op_end[op_id]))
                    if ddr_transfer[op_id]:
                        advance_ddr_work(t_now)
                        previous_count = len(ddr_remaining_work)
                        ddr_remaining_work[op_id] = float(duration)
                        active_count = len(ddr_remaining_work)
                        slowdown_factor = (
                            active_count / previous_count if previous_count else 1.0
                        )
                        projected, provisional, stages = reschedule_ddr_ends(t_now)
                        ddr_contention_log.append({
                            'time': t_now,
                            'issued_op': op_id,
                            'issue_order': len(ddr_contention_log) + 1,
                            'exclusive_end': t_now + duration,
                            'provisional_end': provisional[op_id],
                            'slowdown_factor': slowdown_factor,
                            'active_ops': sorted(ddr_remaining_work),
                            'stages': stages,
                            'projected_ends': {
                                active_id: projected[active_id]
                                for active_id in sorted(projected)
                            },
                        })

    def raise_deadlock(reason):
        status_count = Counter(op_status.values())
        queue_snapshot = {
            pipe: [op_id for _, op_id in sorted(queue)]
            for pipe, queue in issue_queues.items()
        }
        executor_snapshot = {
            pipe: sorted(running, key=lambda item: item[1])
            for pipe, running in executors.items()
        }
        lines = [
            '[STEP3 ERROR] scheduler deadlock: reason={} time={} iteration={} status={}'.format(
                reason, t_now, iteration, dict(status_count)),
            'queues={}'.format(queue_snapshot),
            'executors={}'.format(executor_snapshot),
            'ddr_remaining_work={}'.format(dict(sorted(ddr_remaining_work.items()))),
            'memory_used={} capacity={} resident_tensors={}'.format(
                memory_used, capacity, sorted(resident_tensors)),
            'allocation_front={} allocation_ready={}'.format(
                allocation_order[next_allocation_rank]
                if next_allocation_rank < len(allocation_order) else None,
                sorted(allocation_ready)),
        ]
        for op_id in seq:
            if op_status[op_id] == 'done':
                continue
            unresolved = sorted(
                pred for pred in preds[op_id] if op_status.get(pred) != 'done')
            lines.append(
                'op={} type={} pipe={} seq_pos={} status={} pred_remaining={} '
                'unresolved_preds={} in_tids={} out_tids={} alloc_required={}'.format(
                    op_id, op_by_id[op_id]['op'],
                    op_pipe(op_by_id[op_id]), seq_pos[op_id],
                    op_status[op_id], pred_remaining[op_id], unresolved,
                    in_tids[op_id], out_tids[op_id],
                    allocation_requirement(op_id)))
        message = '\n'.join(lines)
        emit_error(message)
        raise Step3SchedulingError(message)

    while True:
        iteration += 1
        if iteration > max_iter:
            raise_deadlock('max_iter_exceeded')
        retire_step()
        if all(status == 'done' for status in op_status.values()):
            break
        issue_step()
        next_times = [end for running in executors.values() for _, end in running]
        if not next_times:
            raise_deadlock('no_running_or_issuable_op')
        t_now = min(next_times)

    makespan = max(op_end.values(), default=0)
    per_pipe_timeline = {pipe: [] for pipe in PIPES}
    for op_id, start in op_start.items():
        pipe = op_pipe(op_by_id[op_id])
        per_pipe_timeline[pipe].append((op_id, start, op_end[op_id]))
    for timeline in per_pipe_timeline.values():
        timeline.sort(key=lambda item: (item[1], seq_pos[item[0]]))

    pipe_orders = {
        pipe: list(order) for pipe, order in planned_pipe_orders.items()
    }

    memory_dependencies = []
    for (source, target), part in sorted(memory_dependency_parts.items()):
        memory_dependencies.append({
            'source': source,
            'target': target,
            'kind': '+'.join(sorted(part['kinds'])),
            'positions': sorted(part['positions']),
            'reused_bytes': part['bytes'],
            'previous_tensor_ids': sorted(
                tid for tid in part['tensor_ids'] if tid is not None),
        })

    # The multi-core simulation only trusts graph dependencies. Virtual credit
    # reuse relations are written as direct op->op edges, keeping metadata for
    # review; _build_graph_views parses them together with ordinary data
    # dependencies.
    execution_graph = deepcopy(ext_graph)
    existing_edges = {
        (edge['source'], edge['target']) for edge in execution_graph['edges']
    }
    for dep in memory_dependencies:
        pair = (dep['source'], dep['target'])
        if pair in existing_edges:
            continue
        execution_graph['edges'].append({
            'source': dep['source'], 'target': dep['target'],
            'dependency': 'MEMORY_REUSE',
        })
        existing_edges.add(pair)

    # Validate the contract once before handing results to the multi-core
    # simulation. What is validated here is the generated result itself; the
    # multi-core event loop need not re-check memory capacity or re-derive the
    # false dependencies.
    scheduled_ids = {
        op_id for order in pipe_orders.values() for op_id in order
    }
    if scheduled_ids != set(op_by_id) or sum(map(len, pipe_orders.values())) != len(op_by_id):
        raise Step3SchedulingError('Step3 pipe orders do not cover every op exactly once')
    exec_in, exec_out, exec_preds, exec_succs = _build_graph_views(
        execution_graph['ops'], execution_graph['edges'])
    del exec_in, exec_out
    for target, target_preds in exec_preds.items():
        for source in target_preds:
            if op_end[source] > op_start[target]:
                raise Step3SchedulingError(
                    'Step3 dependency timing violation: {} -> {}'.format(
                        source, target))
    indegree = {op_id: len(exec_preds[op_id]) for op_id in op_by_id}
    ready = [op_id for op_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    visited = 0
    while ready:
        source = heapq.heappop(ready)
        visited += 1
        for target in exec_succs[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    if visited != len(op_by_id):
        raise Step3SchedulingError('memory reuse dependencies introduced a cycle')

    unfinished_refs = {
        tid: count for tid, count in remaining_consumers.items() if count != 0
    }
    if unfinished_refs or resident_tensors or any(memory_used.values()):
        message = (
            '[STEP3 ERROR] tensor lifetime did not close: remaining_refs={} '
            'resident_tensors={} memory_used={}'
        ).format(unfinished_refs, sorted(resident_tensors), memory_used)
        emit_error(message)
        raise Step3SchedulingError(message)

    return {
        'makespan': makespan,
        'op_start': op_start,
        'op_end': op_end,
        'per_pipe_timeline': per_pipe_timeline,
        'pipe_orders': pipe_orders,
        'execution_graph': execution_graph,
        'memory_dependencies': memory_dependencies,
        'op_status': op_status,
        'completed_count': len(op_start),
        'pred_remaining': pred_remaining,
        'in_tids': in_tids,
        'out_tids': out_tids,
        'ddr_transfer': ddr_transfer,
        'ddr_contention_log': ddr_contention_log,
        'memory_peak': memory_peak,
        'memory_used_final': memory_used,
        'memory_events': memory_events,
        'remaining_consumers': remaining_consumers,
        'resident_tensors_final': sorted(resident_tensors),
        'execution_contract_validated': True,
    }

def prepare_step3_execution(ext_graph, capacity, bandwidth):
    """Run Step3 and return a Task description the multi-core evaluators can
    load directly.

    The multi-core phase may change ops' absolute start/end times but must
    keep the per-Pipe order given here, and must treat the memory reuse edges
    in execution_graph as ordinary completion dependencies.
    Input constraints and their source are the same as step3_simulation; the
    return value directly references its validated graph and Pipe orders
    without a second coverage/local-DAG validation. New cycles introduced by
    cross-core COPYs are checked by the evaluation entry.
    Capacity and bandwidth are read from config.txt by the caller and passed
    in.
    """
    result = step3_simulation(
        ext_graph, capacity=capacity, bandwidth=bandwidth)
    graph = result['execution_graph']
    in_tids, out_tids, op_preds, op_succs = _build_graph_views(
        graph['ops'], graph['edges'])
    return {
        'graph': graph,
        'seq': graph['seq_ext'],
        'seq_pos': {op_id: index
                    for index, op_id in enumerate(graph['seq_ext'])},
        'op_by_id': {op['id']: op for op in graph['ops']},
        'tensor_by_id': {tensor['id']: tensor for tensor in graph['tensors']},
        'in_tids': in_tids,
        'out_tids': out_tids,
        'op_preds': op_preds,
        'op_succs': op_succs,
        'pipe_ops': {pipe: list(order)
                     for pipe, order in result['pipe_orders'].items()},
        'step3': result,
    }
