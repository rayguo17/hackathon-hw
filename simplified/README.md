# 简化题：多核调度评估代码

基于 `../chinese/` 原三问题评估代码改造，对应简化版赛题
（"17-Multi-Core Scheduling of Computational Graphs under Limited Memory
Bandwidth - alternative.docx"）：单一问题、单一 L1 存储、无场景区分、无 L2 Cache。

与原版的主要差异：

- 只有 **L1 = 1 MiB** 一个片上存储（原版 L1 512 KB + UB 128 KB）；
  tensor `pos` 只允许 `DDR` / `L1`。
- 不区分场景 A/B，也没有 Task 切换等待；每核所有子图合并为一个
  Task，跨核数据经 DDR 中转，目标 COPY_IN 在源 COPY_OUT 结束
  **1000 周期**后才能发射。
- 移除了 L2 只读 Cache 与 Cache 命中率指标。
- 搬运量指标是**总 COPY 字节数**（`total_copy_bytes`），不是相对原图的增量。
- 核内 op 顺序仍由评估器固定启发式（子图调度序为主键）生成；选手通过
  `core_schedules` 间接调整，方案格式未定义直接的 op 顺序字段。

## 常用命令

在 `simplified/` 目录下运行：

```bash
python3 code/stub_multicore_cut_and_schedule.py <计算图.json> -n 4
python3 code/multicore_evaluate.py <计算图.json> <方案.json> --config data/config.txt
python3 code/singlecore_evaluate.py <计算图.json> --config data/config.txt
```

省略 `<方案.json>` 时读取与计算图同目录的 `<计算图名>_multicore_res.json`。
输出结果 JSON、简短日志与 Perfetto Trace（`*_multicore_res/trace/log`）。
`singlecore_evaluate.py` 给出速度比基线 `T_base`。

## 配置（data/config.txt）

| 段 | 键 | 含义 |
|---|---|---|
| `[capacity]` | `L1` | 片上 L1 容量（字节），固定 1048576 |
| `[bandwidth]` | `bandwidth` | 共享 DDR 总带宽（字节/周期），固定 60 |
| `[sync]` | `cross_core_sync_cycles` | 跨核同步等待（周期），固定 1000 |

配置缺失或拼写错误直接报错，不使用代码内置默认值。

## 文件列表

| 文件 | 作用 |
|---|---|
| `code/contest_io.py` | 统一命令行、JSON、日志与 Perfetto Trace |
| `code/evaluation_validation.py` | 图/方案校验与配置解析（仅 L1） |
| `code/multicore_evaluate.py` | 统一多核评估器（每核合并 Task + 跨核 COPY + 事件模拟） |
| `code/singlecore_evaluate.py` | 单核基线（速度比基准 `T_base`） |
| `code/schedule_step1.py` | 核内反向 DFS 拓扑序 |
| `code/schedule_step2.py` | L1 容量溢出与换入换出插入 |
| `code/schedule_step3.py` | 乱序流水排布、逐 Pipe 顺序与内存复用补边 |
| `code/stub_multicore_cut_and_schedule.py` | 方案格式校验与随机示例生成 |
| `data/config.txt` | 固定评估配置 |

原 `multicore_cut_evaluate_problem_{1,2,3}.py` 已删除；问题 1/2/3 的
场景区分、Task 切换等待、L2 Cache 逻辑不再存在。
