# 在共享服务器运行

本页回答四件事：登录后去哪里、用哪个 Python、哪些命令只执行一次、哪些命令每个节点都要执行。SFT 选样、训练配方、评估参数和结果解读见[通用运行手册](running.md)。

独立 26 卡 VBench 的具体分片与启动步骤见[全量评测手册](vbench.md)。在获分配的卡上，约 3 GiB 常驻、计算利用率为 0 即可判作可用；不得占用未分配的卡。T5 条件缓存须通过 `SCRL_EVAL_CONDITIONS` 指向生成节点 `/var/tmp` 下的作业目录；视频、plan 和评分收据仍在共享盘。

## 登录后先定位

| 位置 | 放什么 |
|---|---|
| 控制机 `/data/260010081/project/semantic-chunk-rl` | 唯一开发仓库；私有登录清单 `configs/cluster.json` 只留在这里，不复制到共享盘 |
| 共享 `$W/semantic-chunk-rl`（下文 `$C`） | GitHub `main` 的工作树；实际拓扑 `configs/cluster.runtime.json` 和作业 `runs/<作业名>/` 不进 Git |
| 共享 `$W/semantic-chunk-dependencies`（下文 `$D`） | 模型、数据、上游源码、解释器；`configs/paths.json` 指向实物，`configs/pins.json` 记录独立审核的摘要 |
| 评测节点本地盘 | 周期 VBench 写入 `/var/tmp/semantic-chunk-vbench/<作业名>-<配置摘要前16位>/`，不是共享目录 |

`/hbt_share02/202611/ryt_workspace` 和 `/data_test/202611/ryt_workspace` 对应同一份文件，但不是每台机器都提供前一个路径。**配置和作业路径统一用 `/data_test`，不需要同步两遍。** 代码树里的 `models`、`runtime` 等是兼容链接，不是另一份资产。

每次打开新的远端 shell，先执行以下 Bash 命令：

```bash
W=/data_test/202611/ryt_workspace
C="$W/semantic-chunk-rl"
D="$W/semantic-chunk-dependencies"
cd "$C" || exit 1
export SCRL_PYTHON="$D/runtime/train-v1/bin/python"
export PATH="$D/runtime/train-v1/bin:$PATH"
```

### 解释器和输入

系统默认的 `python3` 在部分节点版本过旧。启动 RL 命令用上面的 `SCRL_PYTHON`；程序按实际路径配置为不同角色选择子进程解释器。不要将三个环境的包混装。

| 角色 | 解释器（相对 `$D/runtime/`） | Torch / Transformers | 关键输入（相对 `$D/`） |
|---|---|---|---|
| Planner / Actor | `planner-joint-v1/bin/python` | 2.13.0 / 5.18.0；PEFT 0.21.0、verl 0.5.0 | `models/Qwen3.5-4B` + `models/planner-sft-4b-2ep/adapter` |
| Executor / 8B Judge | `train-v1/bin/python` | 2.8.0+cu128 / 4.57.6 | `upstream/LongLive-1.3B`、LongLive/Wan 权重、`assets/executor_reference`；Judge 用 `models/Qwen3-VL-8B-Instruct` |
| VBench 画质评分 | `vbench-v1/bin/python` | 2.8.0+cu128 / 4.37.2 | `third_party/vbench`、`models/VBench`；独立全维评估用 `models/VBench-eval-v1` |

SFT 的定版数据在 `$D/datasets/teacher-sft/`，对应控制机 `datasets/sft/teacher-2000-v1/`；RL 用的是另一份 `$D/data/prompts5000.json`。换节点、环境或输入后重新验收，不要用新的基座替代已审批的 LoRA。

## 同步代码：一处执行

控制机提交前运行 `python3 tools/check_git_sync.py` 并检查 `git diff --cached`，只推已审核的 `main`。远端已是 Git 工作树，不需要再次初始化。确认训练和 watch 都已停止后，在维护节点（清单 2）更新一次：

```bash
git status --short --branch
# 应为 main 且无意外修改；否则先查明原因，不要继续拉取
git pull --ff-only origin main
git rev-parse HEAD
```

核对输出与控制机要部署的 commit 一致。共享盘其他节点立即看到同一版；不要逐台 pull、覆盖运行中的源码或执行 `git clean -fdx`。远端不能直连 GitHub 时，可从控制机交付已核对 GitHub SHA 的单分支 bundle，在共享工作树快进到同一提交，不传旧分支、标签或整个 `.git`。

## 24 卡 RL：每步在哪里执行

**1. 只在一台 prepare 节点创建作业。** 当前可选清单 3、4 或 6。原因是 prepare 会记录解析后的绝对路径；如果 `pwd -P` 返回 `/hbt_share02/...`，别的节点可能无法重开该配置。下面是单轮验收，不会启动 GPU 进程：

```bash
# RUN_ID 只生成一次；后续所有节点使用这条命令打印的同一 RUN 绝对路径
RUN_ID="whole24-$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$C/runs/$RUN_ID"
printf 'RUN=%s\n' "$RUN"
(
  set -eu
  test "$(pwd -P)" = "$C"
  test ! -e "$RUN"
  sh scripts/rl.sh prepare --recipe configs/rl-whole24.json \
    --paths "$D/configs/paths.json" --cluster configs/cluster.runtime.json \
    --approvals "$D/configs/pins.json" --run "$RUN" --rounds 1
)
```

prepare 成功后，`$RUN/config.json` 和 `$RUN/receipt.json` 应存在。当前配方对应 5,000 条提示词、14 张 Executor、8 张评分卡、2 张 Planner/Actor；另有 2 张独立评测卡。**这一步只验证输入，不代表已经开始训练。**

**2. 六台训练机各启动一次。** 先核对本机分配的物理卡、活动进程和共享内存；节点映射以此次 `config.json` 为准。下表是当前布局。注意最后一台：**清单 6 用 `--node 5`，清单 5 不运行 train。**

| 私有清单索引 | RL 运行时 `--node` | 角色 |
|---|---|---|
| 0、1 | 0、1 | 每台 4 张 Executor |
| 2 | 2 | 2 张 Executor、2 张 Planner/Actor |
| 3、4 | 3、4 | 每台 4 张评分卡 |
| 6 | 5 | 4 张 Executor |
| 5 | 不传，不启动训练 | 独立 benchmark |

每台先执行本页开头的环境设置，再设置相同的 `RUN` 和本机的 `N`。六台须通过私有环境注入同一个至少 32 字符的 `SCRL_RPC_TOKEN`；不要逐机生成，不要写到共享目录、命令参数或日志。

```bash
: "${RUN:?设置为 prepare 输出的完整作业路径}"
: "${N:?按上表设置本机运行时索引}"
: "${SCRL_RPC_TOKEN:?请先在私有环境中注入同一作业的 token}"
sh scripts/rl.sh train --run "$RUN" --node "$N"
```

看到 PID 只说明进程已提交。先看 `$RUN/launch/node-<N>.log` 和 `$RUN/launch/coordinator.log`；出现 `$RUN/FAIL.json` 时先处理其中记录的首错，不要重复启动。整轮的 `batches/*/report.json` 和双端 checkpoint 才能用于判断是否跑完，详见[日志与排障](running.md#日志与排障)。

**3. 单轮不启动 benchmark；长期训练按需启动。** 验收通过后在新 `RUN` 做 prepare，去掉 `--rounds 1`。若要续训同一训练身份，显式传 `--resume` 和已验证的联合 checkpoint；改配方不借用旧断点。六台训练机照步骤 2 启动。只在清单 5 的持久终端中，按需执行：

```bash
: "${RUN:?设置为长期作业的完整路径}"
sh scripts/rl.sh watch --run "$RUN"
```

当前每 20 步评估一次已发布的 checkpoint。更新代码或删除作业目录前先停止对应的 watch，避免它继续等待已经不存在的作业。
