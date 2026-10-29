"""生成简化题样例计算图：从易到难，全部符合评估器校验规则。

规则要点（与 evaluation_validation.validate_graph 一致）：
- tensors: id/op 共享唯一非负 id, pos ∈ {DDR, L1}, size 为非负整数
- ops:     op 名任意（COPY_IN/COPY_OUT 为搬运，其余为计算），pipe ∈ 四条 Pipe,
           cycles 为非负整数（COPY 的实际时长由仿真按带宽动态计算）
- edges:   只允许 Tensor→Op 或 Op→Tensor，无重复边，整图为 DAG
- 题目约定: 图输入在 DDR 且每个 DDR 输入只搬入一次；最终结果搬出一次；
            单个 op 的输入+输出总量不超过 L1 容量

用法: python3 generate_samples.py [输出目录]（默认本目录）
"""

import json
import sys
from pathlib import Path

# 与 data/config.txt 一致
L1_CAPACITY = 1048576
BANDWIDTH = 60


def _tensor(tid, pos, size):
    return {'id': tid, 'pos': pos, 'size': size}


def _op(oid, name, pipe, cycles):
    return {'id': oid, 'op': name, 'pipe': pipe, 'cycles': cycles}


def _graph(tensors, ops, edges):
    return {'tensors': tensors, 'ops': ops, 'edges': edges}


def copy_in(tid, size):
    """DDR -> L1 搬入节点及其 DDR 源 tensor。

    返回 (DDR源tensor, COPY_IN op, 下一个可用 id)。
    """
    return _tensor(tid, 'DDR', size), _op(tid + 1, 'COPY_IN', 'PIPE_MTE2', 1), tid + 2


def copy_out(oid, src_tid, size):
    """L1 -> DDR 搬出节点及其 DDR 目标 tensor。

    返回 (tensor节点列表, COPY_OUT op, 边列表, 下一个可用 id)。
    """
    return ([_tensor(oid, 'DDR', size)],
            _op(oid + 1, 'COPY_OUT', 'PIPE_MTE3', 1),
            [{'source': src_tid, 'target': oid + 1},
             {'source': oid + 1, 'target': oid}],
            oid + 2)


def sample1_single_op():
    """最简单：一次搬入、一次向量加、一次搬出（对应赛题 B.7 示例）。"""
    tensors, ops, edges = [], [], []
    t_in, op_in, nid = copy_in(1, 16)
    tensors.append(t_in)
    ops.append(op_in)
    edges += [{'source': 1, 'target': op_in['id']},
              {'source': op_in['id'], 'target': nid}]
    tensors.append(_tensor(nid, 'L1', 16))
    t_mid = nid
    nid += 1
    ops.append(_op(nid, 'ADD', 'PIPE_V', 4))
    edges += [{'source': t_mid, 'target': nid},
              {'source': nid, 'target': nid + 1}]
    tensors.append(_tensor(nid + 1, 'L1', 16))
    t_out_src = nid + 1
    nid += 2
    t_out, op_out, e_out, nid = copy_out(nid, t_out_src, 16)
    tensors += t_out
    ops.append(op_out)
    edges += e_out
    return _graph(tensors, ops, edges)


def sample2_chain():
    """串行流水：MTE2 -> V -> M -> MTE3 交替链，测试同 Pipe 串行。"""
    tensors, ops, edges = [], [], []
    nid = 1
    t_in, op_in, nid = copy_in(nid, 4096)
    tensors.append(t_in)
    ops.append(op_in)
    edges.append({'source': t_in['id'], 'target': op_in['id']})
    edges.append({'source': op_in['id'], 'target': nid})
    tensors.append(_tensor(nid, 'L1', 4096))
    prev = nid
    nid += 1
    stages = [('ACTIVATE', 'PIPE_V', 20), ('MATMUL', 'PIPE_M', 100),
              ('ACTIVATE', 'PIPE_V', 20), ('MATMUL', 'PIPE_M', 100),
              ('ACTIVATE', 'PIPE_V', 20)]
    for name, pipe, cycles in stages:
        tensors.append(_tensor(nid, 'L1', 4096))
        ops.append(_op(nid + 1, name, pipe, cycles))
        edges.append({'source': prev, 'target': nid + 1})
        edges.append({'source': nid + 1, 'target': nid})
        prev = nid
        nid += 2
    t_out, op_out, e_out, nid = copy_out(nid, prev, 4096)
    tensors += t_out
    ops.append(op_out)
    edges += e_out
    return _graph(tensors, ops, edges)


def sample3_diamond():
    """并行分叉：共享 DDR 输入分给两条独立分支，结果汇合后搬出。

    适合测试切图并行度：两分支放不同核可并行，但输出汇合引入跨核同步。
    """
    tensors, ops, edges = [], [], []
    nid = 1
    # 共享输入 x（DDR），两个 COPY_IN 分别读入（重复读，计入 D）
    x_tid = 1000
    tensors.append(_tensor(x_tid, 'DDR', 8192))
    branch_tails = []
    for branch in range(2):
        t_in, op_in, nid = copy_in(nid, 8192)
        tensors.append(t_in)
        ops.append(op_in)
        edges.append({'source': x_tid, 'target': op_in['id']})
        edges.append({'source': op_in['id'], 'target': nid})
        tensors.append(_tensor(nid, 'L1', 8192))
        a_tid = nid
        nid += 1
        ops.append(_op(nid, 'MATMUL', 'PIPE_M', 200 + 50 * branch))
        tensors.append(_tensor(nid + 1, 'L1', 4096))
        edges.append({'source': a_tid, 'target': nid})
        edges.append({'source': nid, 'target': nid + 1})
        branch_tails.append(nid + 1)
        nid += 2
    # 汇合：ADD 消费两条分支结果
    ops.append(_op(nid, 'ADD', 'PIPE_V', 30))
    for tail in branch_tails:
        edges.append({'source': tail, 'target': nid})
    tensors.append(_tensor(nid + 1, 'L1', 4096))
    edges.append({'source': nid, 'target': nid + 1})
    t_out_src = nid + 1
    nid += 2
    t_out, op_out, e_out, nid = copy_out(nid, t_out_src, 4096)
    tensors += t_out
    ops.append(op_out)
    edges += e_out
    return _graph(tensors, ops, edges)


def sample4_shared_input():
    """重复读压力：一个 DDR 输入被多个子图反复消费（抬升 D 指标）。"""
    tensors, ops, edges = [], [], []
    nid = 1
    x_tid = 2000
    block = 12288  # 12 KB
    tensors.append(_tensor(x_tid, 'DDR', block))
    tails = []
    for k in range(6):
        t_in, op_in, nid = copy_in(nid, block)
        tensors.append(t_in)
        ops.append(op_in)
        edges.append({'source': x_tid, 'target': op_in['id']})
        edges.append({'source': op_in['id'], 'target': nid})
        tensors.append(_tensor(nid, 'L1', block))
        a_tid = nid
        nid += 1
        pipe, name = ('PIPE_M', 'MATMUL') if k % 2 == 0 else ('PIPE_V', 'ACTIVATE')
        ops.append(_op(nid, name, pipe, 150))
        tensors.append(_tensor(nid + 1, 'L1', 4096))
        edges.append({'source': a_tid, 'target': nid})
        edges.append({'source': nid, 'target': nid + 1})
        tails.append(nid + 1)
        nid += 2
    # 逐对累加成一棵小归约树
    while len(tails) > 1:
        nxt = []
        for i in range(0, len(tails) - 1, 2):
            ops.append(_op(nid, 'ADD', 'PIPE_V', 10))
            for tail in tails[i:i + 2]:
                edges.append({'source': tail, 'target': nid})
            tensors.append(_tensor(nid + 1, 'L1', 4096))
            edges.append({'source': nid, 'target': nid + 1})
            nxt.append(nid + 1)
            nid += 2
        if len(tails) % 2 == 1:
            nxt.append(tails[-1])
        tails = nxt
    t_out, op_out, e_out, nid = copy_out(nid, tails[0], 4096)
    tensors += t_out
    ops.append(op_out)
    edges += e_out
    return _graph(tensors, ops, edges)


def sample5_vgg16(layers=6, h_blocks=6, w=28, c=16):
    """VGG16 式卷积链：特征图沿 H 轴按粒度 2 分块，层内交错依赖。

    每层每输出块 conv 消费相邻两个输入块（3x3 感受野造成的 halo），
    再接一个 PIPE_V 激活；层间串联。切分 H 轴即可解耦层内依赖。
    """
    tensors, ops, edges = [], [], []
    nid = 1
    block_bytes = 2 * w * c * 2  # 2 行 × W × C × fp16
    # 权重：每层一份 DDR 权重，被该层所有输出块共享读
    weight_tids = {}
    for l in range(layers):
        tensors.append(_tensor(5000 + l, 'DDR', c * c * 2))
        weight_tids[l] = 5000 + l
    feats = {}
    for h in range(h_blocks):
        t_in, op_in, nid = copy_in(nid, block_bytes)
        tensors.append(t_in)
        ops.append(op_in)
        edges.append({'source': t_in['id'], 'target': op_in['id']})
        edges.append({'source': op_in['id'], 'target': nid})
        tensors.append(_tensor(nid, 'L1', block_bytes))
        feats.setdefault(0, {})[h] = nid
        nid += 1
    for l in range(layers):
        nxt = {}
        for h in range(h_blocks):
            ops.append(_op(nid, 'CONV', 'PIPE_M', 300))
            for src_h in (h, h + 1):
                if src_h < h_blocks:
                    edges.append({'source': feats[l][src_h], 'target': nid})
            edges.append({'source': weight_tids[l], 'target': nid})
            tensors.append(_tensor(nid + 1, 'L1', block_bytes))
            edges.append({'source': nid, 'target': nid + 1})
            conv_out = nid + 1
            nid += 2
            ops.append(_op(nid, 'ACTIVATE', 'PIPE_V', 40))
            edges.append({'source': conv_out, 'target': nid})
            tensors.append(_tensor(nid + 1, 'L1', block_bytes))
            edges.append({'source': nid, 'target': nid + 1})
            nxt[h] = nid + 1
            nid += 2
        feats[l + 1] = nxt
    # 最终输出：最后一层每个块单独搬出（也测试多输出）
    for h in range(h_blocks):
        t_out, op_out, e_out, nid = copy_out(nid, feats[layers][h], block_bytes)
        tensors += t_out
        ops.append(op_out)
        edges += e_out
    return _graph(tensors, ops, edges)


def matmul_case(k_blocks=1, i_blocks=2, j_blocks=2, mm_cycles=300,
                block_bytes=32768):
    """分块矩阵乘（对应赛题 Matmul 结构）：C_ij = Σ_k A_ik · B_kj。

    左右矩阵按块搬入，A 块被一列 C 块重复读、B 块被一行 C 块重复读，
    正是题面强调的重复搬运带宽瓶颈。k_blocks>1 时用 ADD 累加部分和。
    """
    tensors, ops, edges = [], [], []
    nid = 1
    # 输入矩阵 A (I×K 块) 与 B (K×J 块)
    a_tids, b_tids = {}, {}
    for i in range(i_blocks):
        for k in range(k_blocks):
            tensors.append(_tensor(nid, 'DDR', block_bytes))
            a_tids[i, k] = nid
            nid += 1
    for k in range(k_blocks):
        for j in range(j_blocks):
            tensors.append(_tensor(nid, 'DDR', block_bytes))
            b_tids[k, j] = nid
            nid += 1
    # 每个 A/B 块一个 COPY_IN（每块只搬入一次）
    a_local, b_local = {}, {}
    for tid in sorted(list(a_tids.values()) + list(b_tids.values())):
        ops.append(_op(nid, 'COPY_IN', 'PIPE_MTE2', 1))
        edges.append({'source': tid, 'target': nid})
        tensors.append(_tensor(nid + 1, 'L1', block_bytes))
        edges.append({'source': nid, 'target': nid + 1})
        nid += 2
    for (i, k), tid in a_tids.items():
        a_local[i, k] = tid + 1
    for (k, j), tid in b_tids.items():
        b_local[k, j] = tid + 1
    # C 块：k 累加
    c_out = {}
    for i in range(i_blocks):
        for j in range(j_blocks):
            acc = None
            for k in range(k_blocks):
                ops.append(_op(nid, 'MATMUL', 'PIPE_M', mm_cycles))
                edges.append({'source': a_local[i, k], 'target': nid})
                edges.append({'source': b_local[k, j], 'target': nid})
                tensors.append(_tensor(nid + 1, 'L1', block_bytes))
                edges.append({'source': nid, 'target': nid + 1})
                mm_out = nid + 1
                nid += 2
                if acc is None:
                    acc = mm_out
                else:
                    ops.append(_op(nid, 'ADD', 'PIPE_V', 50))
                    edges.append({'source': acc, 'target': nid})
                    edges.append({'source': mm_out, 'target': nid})
                    tensors.append(_tensor(nid + 1, 'L1', block_bytes))
                    edges.append({'source': nid, 'target': nid + 1})
                    acc = nid + 1
                    nid += 2
            c_out[i, j] = acc
    # 每个输出块一个 COPY_OUT
    for (i, j), acc in sorted(c_out.items()):
        t_out, op_out, e_out, nid = copy_out(nid, acc, block_bytes)
        tensors += t_out
        ops.append(op_out)
        edges += e_out
    return _graph(tensors, ops, edges)


def sample7_memory_pressure(h_blocks=10, block_bytes=393216):
    """L1 容量压力：中间块约 384KB×多块驻留，逼出换入换出（SPILL）。"""
    tensors, ops, edges = [], [], []
    nid = 1
    feats = []
    for h in range(h_blocks):
        t_in, op_in, nid = copy_in(nid, block_bytes)
        tensors.append(t_in)
        ops.append(op_in)
        edges.append({'source': t_in['id'], 'target': op_in['id']})
        edges.append({'source': op_in['id'], 'target': nid})
        tensors.append(_tensor(nid, 'L1', block_bytes))
        feats.append(nid)
        nid += 1
    # 长链消费：每个 op 读最旧的两个未消费块，拉长驻留时间
    tails = []
    queue = list(feats)
    while len(queue) > 1:
        a, b = queue.pop(0), queue.pop(0)
        ops.append(_op(nid, 'MATMUL', 'PIPE_M', 500))
        edges.append({'source': a, 'target': nid})
        edges.append({'source': b, 'target': nid})
        tensors.append(_tensor(nid + 1, 'L1', block_bytes // 2))
        edges.append({'source': nid, 'target': nid + 1})
        queue.append(nid + 1)
        nid += 2
    t_out, op_out, e_out, nid = copy_out(nid, queue[0], block_bytes // 2)
    tensors += t_out
    ops.append(op_out)
    edges += e_out
    return _graph(tensors, ops, edges)


SAMPLES = {
    'sample1_single_op.json': sample1_single_op,
    'sample2_chain.json': sample2_chain,
    'sample3_diamond.json': sample3_diamond,
    'sample4_shared_input.json': sample4_shared_input,
    'sample5_vgg16.json': sample5_vgg16,
    'matmul_case.json': matmul_case,
    'sample6_matmul_large.json': lambda: matmul_case(
        k_blocks=2, i_blocks=4, j_blocks=4, mm_cycles=2000, block_bytes=32768),
    'sample7_memory_pressure.json': sample7_memory_pressure,
}


def main(argv):
    out_dir = Path(argv[1]) if len(argv) > 1 else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))
    from evaluation_validation import validate_graph
    for name, builder in SAMPLES.items():
        graph = builder()
        validate_graph(graph)  # 生成即校验，坏了立刻报错
        path = out_dir / name
        path.write_text(json.dumps(graph, ensure_ascii=False, indent=1) + '\n',
                        encoding='utf-8')
        n_copy = sum(1 for op in graph['ops'] if op['op'] in ('COPY_IN', 'COPY_OUT'))
        print('{}: {} tensors, {} ops ({} COPY), {} edges'.format(
            name, len(graph['tensors']), len(graph['ops']), n_copy,
            len(graph['edges'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
