# 普通 VBench 全量评测

本页维护操作步骤。用同一个 UTC 实验 ID 关联 `docs/experiments/<ID>.md`、`$C/runs/<ID>/<ARM>/` 和[结果总表](results.md)。同一物理卡上，上一组的生成进程退出后才交接给下一组。实际分片、批量和启动时间以[实验记录](experiments/20261126-vbench-full-4b-base-vs-sft-longlive-v1.md)为准；不要重复提交正在运行的分片。

## 准备一次

先按[服务器手册](server.md)设置共享目录。下面在新目录为裸 4B 创建清单；同名作业已存在时只能检查，不能覆盖。普通全量不传 `--prompts`。

```bash
W=/data_test/202611/ryt_workspace
C="$W/semantic-chunk-rl"
D="$W/semantic-chunk-dependencies"
ID=20261126-vbench-full-4b-base-vs-sft-longlive-v1
ARM=base
RUN="$C/runs/$ID/$ARM"
cd "$C" || exit 1
export SCRL_PYTHON="$D/runtime/train-v1/bin/python"
VSRC="$D/third_party/vbench"
sh scripts/vbench.sh prepare --run "$RUN" --arm "$ARM" \
    --config configs/vbench.json --paths "$D/configs/paths.json" \
    --official-metadata "$VSRC/vbench/VBench_full_info.json" \
    --vbench-source "$VSRC" --vbench-weights "$D/models/VBench-eval-v1" \
    --planner-variant pre_sft --base-executor --denoising-mode segment-joint \
    --text-batch-size 8 --video-batch-size 2 --planner-python "$D/runtime/planner-joint-v1/bin/python"
```

裸 4B 应为 `pre_sft`，无 RL checkpoint，`generation_identity.text_batch_size` 为 8。SFT 使用独立运行目录、`--planner-variant sft` 和同一个 base executor。核对两组 tasks、元数据、种子及评分协议一致，不复用裸 4B 的 plan。

## 26 卡生成

仅在已分配的物理卡上启动。按本项目资源约定，指定卡上约 3 GiB 常驻且计算利用率为 0 时，判作可用，不要求显存为零；活跃计算、明显超出常驻基线或已有本作业分片时不重复提交，也不停止其他进程。

| 清单节点 | 物理 GPU（按顺序） | 全局 shard（按顺序） |
|---|---|---|
| 0 / 1 / 2 / 3 / 4 | 各自 0–3 | 0–3 / 4–7 / 8–11 / 12–15 / 16–19 |
| 5 | 2、3 | 20、21 |
| 6 | 0–3 | 22–25 |

每个分片仅用一张卡，在所属节点重复设置上面的 `C/D/ID/ARM/RUN/SCRL_PYTHON`，按表填 `GPU` 和 `SHARD`。T5 的每任务条件张量必须写该节点的 `/var/tmp`，不写共享盘；各节点可用相同的路径字符串，但内容不是共享的。同分片恢复要回原节点并保留路径；空间不够时不要改成共享缓存。

```bash
: "${GPU:?本机物理卡号}" "${SHARD:?全局分片号}"
export SCRL_EVAL_PLANNER_BATCH_SIZE=16  # 单模型多提示词；设为 1 仅供调试
export SCRL_EVAL_CONDITIONS="/var/tmp/semantic-chunk-vbench/$ID/conditions"
mkdir -p "$SCRL_EVAL_CONDITIONS"
df -h "$SCRL_EVAL_CONDITIONS"
CUDA_VISIBLE_DEVICES="$GPU" sh scripts/vbench.sh generate \
    --run "$RUN" --arm "$ARM" --shard "$SHARD" --shards 26
```

26 个生成进程并非分别固定充当 Planner/T5/Executor：每卡只加载一个 Planner 实例，每次解码前向批量处理最多 16 个提示词，各行独立使用任务 seed 的随机流、自己的语法约束与 EOS 终止状态，完成行不再参加采样。之后 T5 对不同 caption 微批（8），LongLive 则只将同一分片内完整有序段长序列相同的任务两两合批（视频 batch 2）。三者独立，不能用同卡多个模型实例代替 Planner 的真正 batch。每个分片另用一个后台线程编码 MP4，最多两个未收尾任务；`records/<ARM>/*.json` 在成片和 SHA 写好后才发布。看 `workers/<ARM>/generate-*/status.json` 区分加载、编码、生成及失败，不能把多个 PID 当成重复占卡或把计划数当作视频数。

非连续分片：为完整任务清单生成互不重叠的 ID 分组，保留空槽，各 worker 加 `--task-ids-file "$RUN/inputs/assignment.json"`；`--shards` 仍为清单的总槽数。排除一张卡不等于排除它原来的提示词。断点恢复沿用原分配文件、原两类 seed 和既有计划，不重建 prepare 目录或静默重抽。

## 评分

每组只提交一个监控。将已退出生成的卡显式交给评分，一张卡同一时间执行一个维度。`--incremental` 会先处理所有所需视频均已发布的维度。静态过滤先于闪烁评分。对于已知缺样、只需处理当前就绪维度的组，可加 `--yield-when-idle`；部分维度不是完整总分。生成期间不要清理本地缓存或共享的视频。

```bash
: "${SCORE_GPU:?已分配的评分物理卡号}"
export SCRL_VBENCH_PYTHON="$D/runtime/vbench-v1/bin/python"
CUDA_VISIBLE_DEVICES="$SCORE_GPU" sh scripts/vbench_score_when_ready.sh \
    "$RUN" --arm "$ARM" --shards 26 --expected-tasks 6220 --incremental
```

非连续 worker 用 `--shards <活跃数量> --shard-indices <逗号分隔的实际编号>`。看 `evaluation/<ARM>/watch-status.json` 和逐维日志；已有监控时不要删锁重开。只有全量视频和 16 维都核验通过，才发布 `evaluation/<ARM>/summary.json` 并回填质量、语义和总分。
