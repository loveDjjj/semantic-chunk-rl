"""研究流程的命令入口。选择子命令后才导入对应阶段。"""

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import sys


def _prepare(argv):
    from scrl.config import prepare

    parser = argparse.ArgumentParser(prog="scrl rl prepare", description="校验联合训练输入并写入新目录，不启动进程。")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--cluster", type=Path, required=True)
    parser.add_argument("--approvals", type=Path, required=True, help="独立审核的教师数据与初始策略摘要")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--resume", type=Path, help="仅允许同一训练身份的联合断点")
    prepare(parser.parse_args(argv))


def _train(argv):
    parser = argparse.ArgumentParser(prog="scrl rl train", description="只启动本节点在已校验训练配置中承担的角色。")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--node", type=int)
    parser.add_argument("--ssh-helper", type=Path, help="Controller SSH helper for periodic VBench")
    args = parser.parse_args(argv)
    from scrl.runtime.launch import launch

    from scrl.config import load_run
    path = args.run / "config.json"
    cfg = load_run(path)
    if cfg.get("periodic_vbench"):
        if args.node is not None:
            parser.error("Periodic training is launched once on the controller; omit --node")
        from scrl.rl.cycle import campaign
        helper = args.ssh_helper or os.environ.get("SCRL_SSH_HELPER")
        if not helper:
            parser.error("Supply --ssh-helper or SCRL_SSH_HELPER")
        return campaign(cfg, Path(helper))
    if args.node is None:
        parser.error("This non-periodic recipe requires --node")
    launch(path, args.node, wait=bool(cfg.get("cycle_window")))


COMMANDS = {
    "sft": {
        "teacher": ("scrl.sft.teacher", "从训练提示词生成待审核草稿"),
        "prepare": ("scrl.sft.data", "将审核后的计划导出为训练集和验证集"),
        "train": ("scrl.sft.train", "训练 4B Planner adapter"),
    },
    "rl": {
        "watch": ("scrl.rl.watch", "在专用评测节点顺序处理已发布断点"),
        "check": ("scrl.rl.preflight", "只读核对联合训练输入"),
        "diagnose": ("scrl.rl.diagnostics", "核对已完成批次的奖励和优势分布"),
        "prepare": (_prepare, "生成联合训练配置"),
        "train": (_train, "在指定节点启动联合训练"),
    },
    "eval": {
        "prepare": ("scrl.evaluation.generate:prepare_main", "固定评测提示词、模型和种子"),
        "generate": ("scrl.evaluation.generate", "按固定清单生成评估视频"),
        "vbench": ("scrl.evaluation.vbench", "独立评估总分、质量分和语义分"),
    },
}


def _launch_profile(stage, step, argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--profile", type=Path, default=os.environ.get("SCRL_PROFILE"))
    parser.add_argument("--print-command", action="store_true")
    flags, extra = parser.parse_known_args(argv)
    key = f"{stage}.{step}"
    cwd = Path.cwd()
    entry = {}
    if flags.profile:
        profile = Path(flags.profile).resolve()
        try:
            entry = json.loads(profile.read_bytes())["commands"][key]
        except (KeyError, TypeError, OSError, ValueError) as exc:
            parser.error(f"{profile}: cannot read commands.{key}: {exc}")
        cwd = profile.parent
    if not isinstance(entry, dict) or set(entry) - {"python", "args", "torchrun", "cuda_visible_devices"}:
        parser.error(f"{key}: unknown launch settings")
    python = entry.get("python", sys.executable)
    if not isinstance(python, str) or not python:
        parser.error(f"{key}.python must name an interpreter")
    options, distributed = entry.get("args", []), entry.get("torchrun")
    for values in (options, distributed if distributed is not None else []):
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            parser.error(f"{key}: args and torchrun must be arrays of strings")
    if any(v.split("=", 1)[0] in ("--profile", "--print-command") for v in options):
        parser.error(f"{key}: launch profiles cannot refer to another profile")
    cmd = [python, "-m", "scrl"]
    if distributed is not None:
        if key != "sft.train":
            parser.error("torchrun is only supported for sft.train; RL uses its cluster layout")
        cmd = [python, "-m", "torch.distributed.run", *distributed, "--module", "scrl"]
    cmd.extend([stage, step, *options, *extra])
    env = dict(os.environ)
    env.pop("SCRL_PROFILE", None)  # 子进程执行已展开的 argv，不再次展开同一配置。
    devices = entry.get("cuda_visible_devices")
    if devices is not None:
        if not isinstance(devices, str):
            parser.error(f"{key}.cuda_visible_devices must be a string")
        env["CUDA_VISIBLE_DEVICES"] = devices
    if flags.print_command:
        print(json.dumps({"cwd": str(cwd), "argv": cmd, "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES")},
                         ensure_ascii=False))
        return
    os.chdir(cwd)
    os.execvpe(cmd[0], cmd, env)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[1] in COMMANDS.get(argv[0], {}):
        if os.environ.get("SCRL_PROFILE") or any(
            arg.split("=", 1)[0] in ("--profile", "--print-command") for arg in argv[2:]
        ):
            return _launch_profile(argv[0], argv[1], argv[2:])
        handler, _ = COMMANDS[argv[0]][argv[1]]
        from scrl.eventlog import event
        command = ".".join(argv[:2])
        event(command, "started")
        try:
            if isinstance(handler, str):
                module, _, name = handler.partition(":")
                handler = getattr(import_module(module), name or "main")
            result = handler(argv[2:])
        except SystemExit as exc:
            event(command, "completed" if exc.code in (None, 0) else "failed")
            raise
        except BaseException as exc:
            event(command, "failed", error_type=type(exc).__name__)
            raise
        event(command, "completed")
        return result
    parser = argparse.ArgumentParser(prog="scrl", description="语义分段训练与视频评估")
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage, commands in COMMANDS.items():
        steps = stages.add_parser(stage).add_subparsers(dest="command", required=True)
        for name, (_, help_text) in commands.items():
            steps.add_parser(name, help=help_text)
    parser.parse_args(argv)
