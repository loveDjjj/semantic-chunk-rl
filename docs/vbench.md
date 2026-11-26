# 普通 VBench 全量评测

本页维护操作步骤。每个具体实验用同一个 UTC 实验 ID 关联 `docs/experiments/<ID>.md`、`$C/runs/<ID>/<ARM>/` 和[结果总表](results.md)。[本次实验](experiments/20261126-vbench-full-4b-base-vs-sft-longlive-v1.md)先用全部 26 卡做 `base`，不分成 13/13 两组，更不是 RL 的 14/8/2。**本次不得自动衔接或手工启动 SFT 组**；以下生成命令只用于新作业首次提交，不能在已经启动的分片上重复执行。

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
    --text-batch-size 8 --planner-python "$D/runtime/planner-joint-v1/bin/python"
```

核对 `config.json` 的 arm 为 `pre_sft`、无 RL checkpoint，`generation_identity.text_batch_size` 为 8。若以后另行授权 SFT，必须用新的 arm 目录、`--planner-variant sft` 和同一个 base executor，核对 tasks、元数据、种子及评分协议一致，但不要复用裸 4B 的 plan。

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
export SCRL_EVAL_CONDITIONS="/var/tmp/semantic-chunk-vbench/$ID/conditions"
mkdir -p "$SCRL_EVAL_CONDITIONS"
df -h "$SCRL_EVAL_CONDITIONS"
CUDA_VISIBLE_DEVICES="$GPU" sh scripts/vbench.sh generate \
    --run "$RUN" --arm "$ARM" --shard "$SHARD" --shards 26
```

26 个生成进程并非分别固定充当 Planner/T5/Executor：各进程先调用同卡 Planner 子进程采样，再对窗口中的不同 caption 做 T5 微批（8，不是同时生成 8 个视频），最后加载 LongLive 逐视频生成。每个分片另用一个后台线程编码 MP4，最多两个未收尾任务；`records/<ARM>/*.json` 在成片和 SHA 写好后才发布。看 `workers/<ARM>/generate-*/status.json` 区分加载、编码、生成及失败，不能把多个 PID 当成重复占卡或把计划数当作视频数。

## 评分一次

在选定评分节点只提交一个评分监控（已有则不重复）。它先在 CPU 上等 26 个分片成功且 6,220 条视频记录齐全，再在指定 GPU 执行完整评分，不占用单独 13 张评分卡。生成期间不要清理本地缓存或共享的视频记录。

```bash
: "${SCORE_GPU:?已分配的评分物理卡号}"
export SCRL_VBENCH_PYTHON="$D/runtime/vbench-v1/bin/python"
CUDA_VISIBLE_DEVICES="$SCORE_GPU" sh scripts/vbench_score_when_ready.sh \
    "$RUN" --arm "$ARM" --shards 26 --expected-tasks 6220
```

看 `evaluation/<ARM>/watch-status.json` 和 `score.log`。若分片失败，先调查该分片再恢复，不能发布部分结果；已有监控时不要删锁重开。成功报告是 `evaluation/<ARM>/summary.json`：核验 16 维覆盖、质量/语义/总分、协议及报告摘要后回填对应实验页和总表。该流程不启动下一组。
