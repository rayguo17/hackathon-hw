"""Single-core execution evaluation entry; no multi-core cut/schedule plan
needed."""

import argparse
import json
import os

from contest_io import (
    cli_errors, _read_json,
    emit,
    format_data_movement_log,
    format_multicore_result_log,
    format_scene_a_trace_json,
    write_text,
)
from multicore_cut_evaluate_problem_1 import (
    evaluate_scene_a,
    read_scene_a_config,
)
from evaluation_validation import read_evaluation_config
from stub_multicore_cut_and_schedule import EXCLUDED_COPY_TYPES

def build_singlecore_plan(graph_json):
    """Merge all non-COPY ops into one subgraph scheduled to core 0."""
    # Full original-graph validation is performed later by the
    # evaluate_scene_a entry; here we only require that IDs can be enumerated,
    # avoiding KeyError/TypeError from invalid JSON before reaching that entry.
    try:
        eligible = sorted(
            op['id'] for op in graph_json.get('ops', [])
            if op.get('op') not in EXCLUDED_COPY_TYPES)
    except (AttributeError, KeyError, TypeError):
        raise ValueError('graph.ops must be a list of op objects with integer ids') from None
    return {
        'node_to_subgraph': {
            str(op_id): 0 for op_id in eligible
        },
        'core_schedules': [[0] if eligible else []],
    }

def evaluate_singlecore(graph_json, bandwidth, capacity,
                        cross_core_wait=0, same_core_wait=0):
    """Run the deterministic single-core evaluation and return the full result.

    Bandwidth and capacity are read by the CLI from config.txt and passed in;
    this function no longer provides configuration defaults.
    """
    plan = build_singlecore_plan(graph_json)
    result = evaluate_scene_a(
        graph_json,
        plan,
        bandwidth=bandwidth,
        capacity=capacity,
        cross_core_wait=cross_core_wait,
        same_core_wait=same_core_wait,
    )
    result['execution_mode'] = 'singlecore'
    result['input_plan'] = None
    return result


@cli_errors
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='evaluate the single-core execution time; no multi-core '
                    'schedule result file needed')
    parser.add_argument('graph', help='original compute graph JSON')
    parser.add_argument('--config',
                        help='config.txt; defaults to the config next to the compute graph')
    parser.add_argument('-o', '--output', help='evaluation result JSON path')
    parser.add_argument('--trace-output', help='Perfetto Trace JSON path')
    parser.add_argument('--log-output', help='concise result log path')
    args = parser.parse_args(argv)
    if args.config and not os.path.isfile(args.config):
        raise FileNotFoundError(f'configuration file not found: {args.config}')

    case_path, _ = os.path.splitext(args.graph)
    config_path = args.config or os.path.join(
        os.path.dirname(args.graph), 'config.txt')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            'configuration file not found: {} (defaults to <graph dir>/config.txt; '
            'pass --config to choose another file)'.format(config_path))
    output_path = args.output or case_path + '_singlecore_res.json'
    trace_path = args.trace_output or case_path + '_singlecore_trace.json'
    log_path = args.log_output or case_path + '_singlecore_log.txt'

    graph = _read_json(args.graph)
    # Capacity and bandwidth come uniformly from config.txt; built-in code
    # defaults are no longer used.
    settings = read_evaluation_config(config_path)
    waits = read_scene_a_config(config_path)
    result = evaluate_singlecore(
        graph,
        bandwidth=settings['bandwidth'],
        capacity=settings['capacity'],
        cross_core_wait=waits['task_cross_core_wait_cycles'],
        same_core_wait=waits['task_same_core_wait_cycles'],
    )

    result['input_graph'] = os.path.basename(args.graph)
    write_text(output_path, json.dumps(
        result, ensure_ascii=False, indent=2) + '\n')
    write_text(trace_path, format_scene_a_trace_json(
        os.path.basename(args.graph), result) + '\n')
    write_text(log_path, format_multicore_result_log(
        os.path.basename(args.graph), result))
    emit(format_data_movement_log(result))
    emit('OK: singlecore makespan={:,} cycles; result={}; trace={}; log={}'.format(
        result['makespan'], output_path, trace_path, log_path))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
