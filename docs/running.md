# 运行手册

本文讲 SFT、RL、独立 VBench 的通用命令、输入、产物与排障。**要在当前服务器开跑，请从[远端运行手册](server.md)开始**：共享盘对应关系、解释器、Git 更新以及六台训练机各自执行的命令只在那里维护。历史变更见[修复日志](changelog.md)，历史分数见[归档](history/vbench-results-through-20261125.md)，新实验见[总表](results.md)。

## 配置与审核

常用入口是 `sh scripts/sft.sh`、`sh scripts/rl.sh` 和 `sh scripts/vbench.sh`。脚本自动定位源码；直接调用 Python 时先设 `export PYTHONPATH=src` 或安装项目。模型、数据、运行目录和凭据不入 Git。

[启动配置](../configs/launch.example.json)按子命令记录 `python`、`args`、可选的 `cuda_visible_devices`；SFT 训练还可指定 `torchrun` 参数。示例的路径和摘要是占位值，不代表实际资产。配置用 JSON 数组传参，不执行 shell 表达式；所有相对业务路径以该启动配置所在目录为准。显式 `--profile` 优先于 `SCRL_PROFILE`；命令行附加参数排在配置参数之后，可覆盖普通标量选项，位置参数不要重复提供。未配置 `python` 时使用启动解释器，可由 `SCRL_PYTHON` 选择。API 密钥和集群 RPC token 经私有环境变量提供；SSH 登录信息只在控制机的私有节点清单维护，二者不能混用。

| 要改什么 | 修改位置 |
|---|---|
| 教师模型、请求参数、SFT 学习率和 LoRA 参数 | [sft.json](../configs/sft.json)，通过 `--config` 选择 |
| 联合训练采样、两端学习率、奖励权重和审核条件 | 新 24 卡使用 [rl-whole24.json](../configs/rl-whole24.json)；旧四格配方为 [rl.json](../configs/rl.json)，请显式传 `--recipe` |
| VBench 种子、帧率、生成预算、评分公式 | [vbench.json](../configs/vbench.json)，通过 `--config` 选择 |
| 权重、数据、节点和 GPU 布局 | [路径示例](../configs/paths.example.json)和[集群示例](../configs/cluster.example.json)，通过 `--paths`、`--cluster` 选择 |

```sh
# 预览示例，不启动进程；真正运行时选择自己的 profile，去掉 --print-command
sh scripts/sft.sh train --profile configs/launch.example.json --print-command
sh scripts/rl.sh train --profile configs/launch.example.json --print-command
sh scripts/vbench.sh generate --profile configs/launch.example.json --shard 0 --shards 4 --print-command
```

联合训练 `--approvals` 是独立审核记录，包含 `teacher_dataset`、`teacher_manifest_sha256`、`official_eval_metadata_sha256`、`sft_run`、`planner_adapter_sha256` 和 `executor_policy_sha256`。SFT 的审核摘要与 Executor 策略缺失时，`rl prepare` 不创建 run。实际启动前应先用目标环境短程验收采样、两端梯度和断点重载。

默认要求官方元数据摘要列入教师数据清单的排除项。本轮有意使用官方原题：教师请求和审核稿导出均须传 `--include-official-metadata "$VBENCH_METADATA"`，其他真正的留出集仍用 `--exclude-metadata`。进入 RL 前，审核人须在 `--approvals` 的 `approved_official_training_metadata_sha256` 数组中独立确认所有纳入训练的官方元数据 SHA-256，必须与教师清单的 `official_training_metadata_sha256` 集合一致。未审批、纳入与排除相冲突、或官方文件身份不匹配都不能开训。

| 阶段 | 入口与产物 |
|---|---|
| SFT | `scrl sft teacher --help` 生成待审分段；独立人工审核后用 `scrl sft prepare --help` 导出分组切分和清单；`scrl sft train --help` 训练并保存 adapter。只按本次声明的提示词范围请求教师。 |
| RL | `scrl rl check --recipe configs/rl-whole24.json --candidate CANDIDATE --approvals APPROVALS` 检查候选输入。实际 prepare、train、watch 的执行位置与命令见[远端运行手册](server.md)。静态通过不等于实卡验收。 |
| 评估 | `scrl eval vbench RUN --arm ARM` 对已有视频评分；加 `--dimension NAME` 只跑一项。质量/语义/总分均可在不加载任何模型的情况下从完整评分收据复算（见下）。 |

## SFT 选样

按 [sft-selection.json](../configs/sft-selection.json) 选择原始提示词，解释器须提供 NumPy、SciPy。以下命令只选样，不请求教师：

```sh
sh scripts/select_sft_prompts.sh --recipe configs/sft-selection.json \
    --vbench "$VBENCH_METADATA" --vbench-long "$VBENCH_LONG_METADATA" \
    --rl "$RL_PROMPTS_JSON" --output "$NEW_SELECTION_DIR"
```

`--vbench-long` 可传普通文件，或 `归档路径::成员路径`。新目录的 `prompts.jsonl` 保留原文、来源行号、审核分组与时间线索；`manifest.json` 固定输入、配方、实现和输出摘要。线索、词面聚类和场景分组只用于覆盖，不是人工认可的动作数或时长标签。

## 独立评估

周期监控按数字顺序读取训练已发布且保留的评估断点，不要求训练暂停。间隔与保留策略见[当前配方](../configs/rl-whole24.json)。失败项留在队首，30 秒后重试；重启时复查已有报告，`--once` 只处理当前积压。

```bash
scrl eval prepare --run "$NEW_RUN" --arm baseline --paths "$PATHS" \
  --official-metadata "$VBENCH_METADATA" --vbench-source "$VBENCH_SRC" --vbench-weights "$VBENCH_WEIGHTS"
scrl eval generate --run "$NEW_RUN" --arm baseline
scrl eval vbench "$NEW_RUN" --arm baseline --output "$NEW_REPORT"
```

prepare 不启动模型或覆盖已有目录。默认使用官方元数据全部提示；可用 `--prompts` 指定已覆盖 16 维的子集（字符串列表，或历史 prompts.json），历史子集沿用原 seed。无训练点时按路径加载初始模型，有 checkpoint 时显式传入。

无 planner 的原生 LongLive 对照：给 `eval prepare` 加 `--native-chunks 7`，用另一新目录，生成时保持原 prompt 连续不切换，按原生块宽逐块去噪，固定 81 帧。它不加载 planner、训练用 executor adapter 或 RL checkpoint。只改 planner 模式的段内去噪请用 `--denoising-mode causal-chunkwise` 而非 native。自由时长与固定预算的差不能全部归因于 RL。

`RUN/config.json` 配置本地官方 VBench 源码、权重及各 metric 源文件摘要（`vbench_source`、`vbench_weights`、`metric_modules_sha256`）；`RUN/inputs/tasks.json` 和 `metadata.json` 固定任务、seed 和官方维度。每个视频有 `RUN/records/ARM/TASK_ID.json`，记录视频绝对路径、SHA-256、prompt 身份和 seed。GPU 评分先校验视频与任务的一一对应；无效维度不填零。无需 RL checkpoint 或训练配置。[默认公式与种子](../configs/vbench.json)用于新评估；历史 run 保留自己的评分公式。

```bash
# 已有 16 个维度的评分 JSON，读存档，无 GPU；不覆盖已有输出
python -m scrl eval vbench "$RUN" --arm "$ARM" --summary-only --output "$NEW_REPORT"
# 同 plan/seed 的两种去噪模式：95% prompt 配对区间（CPU）
python -m scrl.evaluation.paired "$JOINT_RUN" "$CHUNK_RUN" --joint-arm "$ARM" --chunk-arm "$ARM"
```

如需生成视频，独立的 [generate](../src/evaluation/generate.py) 使用评估 run 的 `assets`、`arms`（无训练点时 `checkpoint: null, version: 0`）、`plan`、`max_new_tokens`、`fps`、`protocol_sha256` 和任务；可用 `python -m scrl.evaluation.generate --run "$RUN" --arm "$ARM" --shard 0 --shards 1`。不同 GPU 各用不同 shard，共用同一 `shards`。生成与真实评分必须在已验证运行环境和获准空闲 GPU 上执行；不要在控制机盲目运行。

## 运行目录与保留

下表是当前实际布局，不是三阶段已经统一的目录规范。`RUN` 表示各命令自己的输出位置；SFT 的教师草稿、审核后数据集和训练作业是三个目录。**不提交 Git 不等于可以删除**：可复核的原件应留在受控存储，仓库只提交代码、示例配置和脱敏结果说明。

| 阶段 | 保留的原件与收据 | 可重建或不必发布 | 如何判定成功 |
|---|---|---|---|
| SFT | 教师目录的 `manifest.json`、`review.jsonl`；单独保留人工审核稿；数据集的 `manifest.json`、`train.jsonl`、`validation.jsonl`；训练目录的 `adapter/`、`comparison.json`，以及要审计的 `base-*.json`、`sft-*.json` | 教师 `cache/` 供断点补跑，私存不公开；API 再调用不保证得到原计划。已保留审核稿和清单后可重新导出数据集 | 草稿产出不等于审核通过。核对 split 摘要、训练收据记录的数据清单摘要、验证覆盖与 adapter 可重载；格式通过不代表画质提升。 |
| RL | `RUN/config.json`、`status.json`、`batches/*/report.json`；联合断点 `checkpoints/checkpoint-*/{state.json,planner.pt,executor.pt}`。配置了持久化目录时以该处已发布的断点为准 | 未提交的断点暂存、节点缓存和调试日志不进 Git；不能把半轮数据当作恢复点 | 核对目标轮次、游标、两个策略版本及文件摘要，并独立重载。进程退出或 `drained` 只表示停下，不等于目标轮已完成。 |
| VBench | `RUN/config.json`、`inputs/`、`plans/ARM/`、`records/ARM/`、`evaluation/ARM/` 的分项与覆盖，以及 `--output` 指定的报告；需重新评分时还要原始 `videos/ARM/` | 分项齐全可仅用 CPU 重算总分；文本条件可用固定 plan、编码器和环境重建。视频不入 Git，但缩减视频会失去独立重评依据 | 核对同一模型、协议和任务的覆盖；16 维均有可计分结果、汇总为 `complete` 才记作完整分数。仅有视频或总分不能重建配对区间，见[实验总表与记录](results.md)。 |

发布前检查源数据/视频的授权、提示词与审阅者身份、内部路径与主机信息，以及命令日志中的凭据。不要直接公开整个 `RUN/`、私有 profile 或审批文件；对外导出脱敏副本，保留原始摘要与内部验证记录。

## 日志与排障

RL 的首份全局故障记录在 `RUN/FAIL.json`。先读其中的 role、node、rank，再看对应 worker 的原始日志；主控的 EOF 可能是后果。单视频复审查对应 Executor 的 `score_reviews/` 和该批的 `jobs.jsonl`。复审耗尽不更新不完整批次，不能手动补零或删除锁文件继续运行。

RL coordinator 和 worker 的 UTC 事件写入 `RUN/logs/<host>-<pid>.jsonl`。设置 `SCRL_LOG_DIR` 可启用 SFT、RL、评估的命令级日志；并发任务使用不同目录。每进程 8 MiB 轮转，最多保留 3 个备份。不要记录凭据、提示词或私有配置全文。

联合训练中各节点约每两秒向 `RUN/nodes/node-N/utilization.jsonl` 追加一次采样，只查询分配到本项目的物理卡。按 `unix`、节点和物理 GPU UUID 对齐：`utilization` 是 GPU 忙碌时间百分比；`memory_busy_pct` 是显存发生读写的时间百分比，不是带宽 GB/s；`memory_mib`、`memory_total_mib` 和 `memory_occupied_pct` 是容量。主机 CPU busy、I/O wait 及每网卡收发字节/秒从第二个样本开始提供，属于整机，不是训练进程独占；没有采功耗或温度。

对照同轮 `batches/*/report.json` 的实际视频数、采集/更新时间，以及 `jobs.jsonl` 每个 Executor 的生成、等评分和反向时间。用物理视频数除以轮次秒数比较吞吐。GPU 空闲且打分等待长，先查裁判吞吐；单 rank 长时间落后则查该卡、网络与 CPU。仅凭显存占用高不能判定带宽瓶颈。
