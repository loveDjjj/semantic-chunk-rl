# 数据集目录

本地定版数据集按 `datasets/<用途>/<名称>-v<版本>/` 存放。内容或切分有变化就建新版本，不覆盖旧版。数据文件不入 Git；生成草稿、缓存和日志留在 `runs/`。

| 本地目录 | 用途 | 远端训练副本 |
|---|---|---|
| `sft/teacher-2000-v1/` | Planner SFT，训练 1,812 条，验证 188 条 | `semantic-chunk-dependencies/datasets/teacher-sft/` |

该版本包含 `train.jsonl`、`validation.jsonl` 和记录样本数与文件摘要的 `manifest.json`。其中 600 条提示词来自官方评测池，不应视为独立留出样本。
