# 运行手册

本文说明团队内的代码同步、环境、启动和排障。历史变更见[修复日志](changelog.md)，历史分数和局限见[结果](results.md)。

## 多机目录与设备

| 位置 | 用途与约束 |
|---|---|
| 控制机 `/data/260010081/project/semantic-chunk-rl` | 唯一开发树。私有 `configs/cluster.json` 权限为 `0600`，只留在控制机。不要打印、提交或复制 SSH 凭据。 |
| 共享工作区 `W` | 配置统一使用 `/data_test/202611/ryt_workspace`。`/hbt_share02/202611/ryt_workspace` 是同一共享盘的另一入口，但不是每台机器都有此别名；只需同步一次。 |
| 共享 `$W/semantic-chunk-rl/` | 代码发布目录；新作业使用 runs/ 下的独立子目录。 |
| 共享 `$W/semantic-chunk-dependencies/` | 权重、数据、上游代码和运行环境。配置中的源码、解释器、模型、run 绝对路径必须在相关节点指向同一份内容；本机 `/root`、`/tmp` 不是共享目录。 |
| benchmark 本地盘 | 周期评测写入 `/var/tmp/semantic-chunk-vbench/<run名>-<配置摘要前16位>/step-XXXXX/`；只从共享盘读取输入与已发布断点。不会自动清理。 |

按节点索引查询登录入口，不输出密码：

```sh
jq -er --argjson i 2 '. as $c | .nodes[] | select(.index == $i) | $c.workspace_by_password[.password]' configs/cluster.json
```

当前 24 卡 RL 布局：训练运行时索引 0–5 对应清单索引 0、1、2、3、4、6，分配 14 张 Executor、8 张评分卡、2 张 Planner/Actor 卡；清单索引 5 是独立 benchmark，不在此启动 RL。以新 `RUN/config.json` 的映射和 GPU 白名单为准。启动前须逐节点校验 SSH 主机密钥和卡上进程；零利用率不能代替资源授权。两台评分机各启动四个评分副本，须检查 cgroup 内存和 `/dev/shm`。

## 解释器与模型

不要用系统默认 Python 混跑。已知环境在 `$W/semantic-chunk-dependencies/runtime/`；换节点、驱动或模型后重新验证导入、权重分片、真实前向和跨机 NCCL。

| 角色 | 共享环境 | 已安装 Torch / Transformers | 必需输入（相对依赖目录） |
|---|---|---|---|
| 4B Planner / RL Actor | `planner-joint-v1/bin/python` | 2.13.0 / 5.18.0；PEFT 0.21.0、verl 0.5.0 | `models/Qwen3.5-4B` 和 `models/planner-sft-4b-2ep/adapter`，两者缺一不可 |
| Executor / 8B Judge | `train-v1/bin/python` | 2.8.0+cu128 / 4.57.6 | LongLive、Wan、策略 adapter；评分节点另读 `models/Qwen3-VL-8B-Instruct`，不要求 vLLM |
| VBench | `vbench-v1/bin/python` | 2.8.0+cu128 / 4.37.2 | `third_party/vbench`、`models/VBench`；独立全维评估用 `models/VBench-eval-v1` |

仓库的 `planner` 与 `models` extras 分别对应 Planner 和 Executor/Judge 的 Python 依赖，不能一起安装；它们不是 CUDA、上游代码或完整集群环境的安装器。现网优先使用表中已有前缀；三者不互相升级。定版 SFT 数据在本地 `datasets/sft/teacher-2000-v1/`，远端是依赖目录的 `datasets/teacher-sft/`；过程稿及 API 缓存不作为训练集。RL 提示词另用 `data/prompts5000.json`。

Executor 的 `upstream/LongLive-1.3B/`、`models/Wan2.1-T2V-1.3B/`、`models/LongLive-1.3B/`、`assets/executor_reference/` 及 `data/prompts5000.json` 均相对共享资产根配置。样例路径不是现网配置；准备作业时核对实际文件摘要。

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
| RL | `scrl rl check --recipe configs/rl-whole24.json --candidate CANDIDATE --approvals APPROVALS` 检查候选输入。新作业按下方 prepare 命令准备，六个训练节点分别启动训练，评测节点单独 watch。静态通过不等于实卡验收。 |
| 评估 | `scrl eval vbench RUN --arm ARM` 对已有视频评分；加 `--dimension NAME` 只跑一项。质量/语义/总分均可在不加载任何模型的情况下从完整评分收据复算（见下）。 |

## SFT 选样

按 [sft-selection.json](../configs/sft-selection.json) 选择原始提示词，解释器须提供 NumPy、SciPy。以下命令只选样，不请求教师：

```sh
sh scripts/select_sft_prompts.sh --recipe configs/sft-selection.json \
    --vbench "$VBENCH_METADATA" --vbench-long "$VBENCH_LONG_METADATA" \
    --rl "$RL_PROMPTS_JSON" --output "$NEW_SELECTION_DIR"
```

`--vbench-long` 可传普通文件，或 `归档路径::成员路径`。新目录的 `prompts.jsonl` 保留原文、来源行号、审核分组与时间线索；`manifest.json` 固定输入、配方、实现和输出摘要。线索、词面聚类和场景分组只用于覆盖，不是人工认可的动作数或时长标签。

## Git 同步

GitHub 私库只同步代码、测试、模板和文档。真实节点清单、运行时拓扑、审批文件、机密、权重、数据和运行产物不入 Git。每次提交前在根目录运行 `python3 tools/check_git_sync.py` 检查**整个暂存快照**，并人工检查 `git diff --cached`；工具不扫描历史，不代替人工检查。

**首次切换（只做一次）。** 现有本地 Git 历史包含旧节点清单，留在控制机。先审查待暂存文件，暂存当前变更（包括旧文件的删除），通过上述检查后，用该树建立无父提交的同步分支。不推旧分支、标签或镜像。以下操作仅用于切换，不是日常提交流程：

```sh
# 前提：已暂存要同步的完整工作树，且暂存检查通过；team-sync 是新分支名
git branch team-sync "$(git commit-tree "$(git write-tree)" -m 'Initial team snapshot')"
git switch team-sync
git rev-list --count HEAD   # 首次应为 1
```

首次推送时只将审核后的 `team-sync` 设为私库的同步分支。远端旧目录是文件拷贝，不是 clone：先停作业与 watch，在同盘临时目录 clone 此分支并核对 commit；盘点旧目录中的 `configs/cluster.runtime.json`、在用的 `runs/` 以及依赖兼容链接，切换时只迁入仍需的项目。保留原路径指向同一依赖；不要拉入旧 `src/`、`tools/` 或 `tests/`。新 checkout 在原路径验过真实配置后才清理旧副本。不要在旧目录直接 `git init` 或对共享盘执行 `git clean -fdx`。

**日常更新。** 控制机提交，远端只拉代码，不在远端改源码。确认作业和 watch 已停、工作树无意外改动后，执行 `git pull --ff-only`，对照两端的 `git rev-parse HEAD`。每次只更新这一份共享 checkout，不逐节点重复拉。无 Git 的离线交付才用 `sh scripts/stage_code.sh NEW_DIRECTORY` 和 `sha256sum --check --status SHA256SUMS`；该脚本不是覆盖同步工具。

## 准备与启动

```sh
W=/data_test/202611/ryt_workspace
cd "$W/semantic-chunk-rl"
D="$W/semantic-chunk-dependencies"
PATHS="$D/configs/paths.json"
export SCRL_PYTHON="$D/runtime/train-v1/bin/python"
: "${RUN:?设为共享代码目录 runs/ 下尚不存在的新作业绝对路径}"
sh scripts/rl.sh prepare --recipe configs/rl-whole24.json \
    --paths "$PATHS" --cluster configs/cluster.runtime.json \
    --approvals "$D/configs/pins.json" --run "$RUN"
```

prepare 只写新目录。请在 `pwd -P` 为所有节点均可访问的规范路径的节点执行，核对生成配置的 `source_root`、`run`、解释器和策略身份。换奖励协议不能直接续旧断点。新评分协议上板前先用[缓存探针](../tools/cache_contention_probe.py)验证同键单写，再核对评分、双端梯度和断点重载。

新 24 卡作业须在上述 prepare 命令中显式加 `--recipe configs/rl-whole24.json`。短测用独立新目录并加 `--rounds 1`；每台训练机只提交一次自己的 `--node`，不启动 benchmark。监控 `status.json`、`FAIL.json`、`nodes/*/utilization.jsonl` 和评分进程日志。整轮 report 及两端权重就绪后执行 `sh scripts/rl.sh diagnose --help` 所示的完整性检查，并核对端到端耗时、实际视频数、排队、拒评和显存。首轮包含预热，不能直接归因为布局加速。失败不自动重试。通过后才在另一个新目录去掉 `--rounds`、用相同协议 `--resume` 已验证的联合断点启动一次长期作业（新配方跑完一遍数据集）。如需改变采样、打分或其他训练身份，不得借用旧断点静默继续。

```sh
# 在六台训练机分别运行；N 是该机在 RUN/config.json 中的运行时索引
sh scripts/rl.sh train --run "$RUN" --node "$N"
# 仅在 benchmark 上，用同一份发布，在持久终端或服务管理器下运行
sh scripts/rl.sh watch --run "$RUN"
```

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
| VBench | `RUN/config.json`、`inputs/`、`plans/ARM/`、`records/ARM/`、`evaluation/ARM/` 的分项与覆盖，以及 `--output` 指定的报告；需重新评分时还要原始 `videos/ARM/` | 分项齐全可仅用 CPU 重算总分；文本条件可用固定 plan、编码器和环境重建。视频不入 Git，但缩减视频会失去独立重评依据 | 核对同一模型、协议和任务的覆盖；16 维均有可计分结果、汇总为 `complete` 才记作完整分数。仅有视频或总分不能重建配对区间，见[结果及收据](results.md)。 |

发布前检查源数据/视频的授权、提示词与审阅者身份、内部路径与主机信息，以及命令日志中的凭据。不要直接公开整个 `RUN/`、私有 profile 或审批文件；对外导出脱敏副本，保留原始摘要与内部验证记录。

## 日志与排障

RL 的首份全局故障记录在 `RUN/FAIL.json`。先读其中的 role、node、rank，再看对应 worker 的原始日志；主控的 EOF 可能是后果。单视频复审查对应 Executor 的 `score_reviews/` 和该批的 `jobs.jsonl`。复审耗尽不更新不完整批次，不能手动补零或删除锁文件继续运行。

RL coordinator 和 worker 的 UTC 事件写入 `RUN/logs/<host>-<pid>.jsonl`。设置 `SCRL_LOG_DIR` 可启用 SFT、RL、评估的命令级日志；并发任务使用不同目录。每进程 8 MiB 轮转，最多保留 3 个备份。不要记录凭据、提示词或私有配置全文。

联合训练中各节点约每两秒向 `RUN/nodes/node-N/utilization.jsonl` 追加一次采样，只查询分配到本项目的物理卡。按 `unix`、节点和物理 GPU UUID 对齐：`utilization` 是 GPU 忙碌时间百分比；`memory_busy_pct` 是显存发生读写的时间百分比，不是带宽 GB/s；`memory_mib`、`memory_total_mib` 和 `memory_occupied_pct` 是容量。主机 CPU busy、I/O wait 及每网卡收发字节/秒从第二个样本开始提供，属于整机，不是训练进程独占；没有采功耗或温度。

对照同轮 `batches/*/report.json` 的实际视频数、采集/更新时间，以及 `jobs.jsonl` 每个 Executor 的生成、等评分和反向时间。用物理视频数除以轮次秒数比较吞吐。GPU 空闲且打分等待长，先查裁判吞吐；单 rank 长时间落后则查该卡、网络与 CPU。仅凭显存占用高不能判定带宽瓶颈。
