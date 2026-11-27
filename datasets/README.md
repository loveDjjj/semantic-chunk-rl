# 数据集目录

本地定版数据集按 `datasets/<用途>/<名称>-v<版本>/` 存放。内容或切分有变化就建新版本，不覆盖旧版。数据文件不入 Git；生成草稿、缓存和日志留在 `runs/`。

| 本地目录 | 用途 | 远端训练副本 |
|---|---|---|
| `sft/teacher-2000-v1/` | Planner SFT，训练 1,812 条，验证 188 条 | `semantic-chunk-dependencies/datasets/teacher-sft/` |

**已知偏差（未修复）：** 训练集中 30 条官方颜色题的教师答案全部为单段、2 chunks。复用此配方前应验证简单提示词改写后是否仍生成正确主体，不能只检查 JSON 格式。见[实验诊断](../docs/experiments/20261126-vbench-full-4b-base-vs-sft-longlive-v1.md#高优先级未修复简单提示词的-sft-改写偏差)。

该版本包含 `train.jsonl`、`validation.jsonl` 和记录样本数与文件摘要的 `manifest.json`。其中 600 条提示词来自官方评测池，不应视为独立留出样本。
