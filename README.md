# Semantic Chunk RL

Qwen3.5-4B Planner 把提示词拆成有序的语义段，并决定每段占几个视频 chunk；LongLive-1.3B Executor 按这个计划生成视频。项目研究的是：同一条提示词有多种分法时，哪种分法能更好地完成任务并保持画质。

当前实验先做教师 SFT，再用[24 卡整计划配方](configs/rl-whole24.json)联合训练两端。它按整片语义和质量各 50% 计算回报；独立 VBench 不使用这个训练公式。[历史结果](docs/history/vbench-results-through-20261125.md)不能用于声称当前方案已有提升。

## 从这里开始

在仓库根目录运行以下命令，可检查脚本和示例配置如何拼成训练命令；不加载模型、不需要 GPU，也不启动作业：

```sh
sh scripts/sft.sh train --profile configs/launch.example.json --print-command
```

- [远端服务器手册](docs/server.md)：登录训练节点后，从这里查共享目录、解释器、Git 同步和 24 卡作业的启动步骤。
- [实验运行手册](docs/running.md)：从这里查 SFT、RL 和独立 VBench 各阶段的输入、配置、产物和排障方法。
- [方法](docs/methods.md)：当前整计划采样、训练奖励、比较组，以及历史四格对照。
- [数据目录约定](datasets/README.md)：SFT 切分与版本；数据文件不随 Git 分发。
- [VBench 运行手册](docs/vbench.md)与[实验结果总表](docs/results.md)：前者维护操作步骤，后者索引每次实验、覆盖与分数；[变更日志](docs/changelog.md)记录实现变更。
- [第三方说明](THIRD_PARTY_NOTICES.md)：本仓库的许可不替代模型、数据或第三方实现各自的许可。
