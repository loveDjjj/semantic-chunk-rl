"""读取共享路径，为一次联合训练固定数据、策略、奖励和代码身份。"""

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

from scrl.plan import digest


PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE = PACKAGE_DIR if (PACKAGE_DIR / "scrl.py").is_file() else PACKAGE_DIR.parent
DEFAULT_CONFIGS = SOURCE.parent / "configs" if SOURCE == PACKAGE_DIR else Path(sys.prefix) / "share/scrl/configs"


def read(path):
    return json.loads(Path(path).read_bytes())


def runtime_cluster(cluster):
    """Return only the topology fields safe to publish to workers and run receipts."""
    env = cluster.get("communication_env", {})
    public_env = {"NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME", "NCCL_NET", "NCCL_NET_PLUGIN",
                  "NCCL_IB_HCA", "NCCL_DEBUG", "NCCL_P2P_DISABLE", "GLOO_SOCKET_IFNAME"}
    if not isinstance(env, dict) or any(
        key not in public_env or not isinstance(value, str)
        for key, value in env.items()
    ):
        raise ValueError("communication_env contains an unapproved communication setting")
    public = {
        "nodes": [{k: node[k] for k in ("index", "address", "gpus", "physical_gpus") if k in node}
                  for node in cluster["nodes"]],
        "service_cards": cluster["service_cards"], "actor_node": cluster["actor_node"],
        "base_port": cluster["base_port"], "communication_env": dict(env),
    }
    if "reserved_inventory_node" in cluster:
        reserved = cluster["reserved_inventory_node"]
        if type(reserved) is not int or not 0 <= reserved < len(public["nodes"]):
            raise ValueError("Invalid reserved inventory node")
        if reserved == public["actor_node"] or any(n == reserved for n, _ in public["service_cards"]):
            raise ValueError("Reserved inventory node cannot host training roles")
        public["reserved_inventory_node"] = reserved
    return public


def stage_cluster(inventory, topology):
    """Apply public role placement, never addresses or new devices, to a matching inventory."""
    cluster = runtime_cluster(inventory)
    if topology is None:
        return cluster
    def devices(nodes):
        return [(n["index"], n["gpus"], sorted(n.get("physical_gpus", range(n["gpus"])))) for n in nodes]
    if devices(cluster["nodes"]) != devices(topology["nodes"]):
        raise ValueError("Inventory devices differ from the selected release topology")
    for field in ("actor_node", "service_cards", "reserved_inventory_node"):
        cluster[field] = topology[field]
    return runtime_cluster(cluster)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_manifest():
    return {f"scrl/{p.relative_to(PACKAGE_DIR).as_posix()}": file_hash(p)
            for p in sorted(PACKAGE_DIR.rglob("*.py")) if p != PACKAGE_DIR / "scrl.py"}


def assets_from(paths):
    root = Path(paths["asset_root"])
    if not root.is_absolute():
        raise ValueError("asset_root must be absolute")

    def resolve(value):
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        return str(root / value) if isinstance(value, str) else value

    return resolve(paths["assets"])


def reserve_evaluator(cluster, inventory_index):
    nodes = cluster["nodes"]
    if [n["index"] for n in nodes] != list(range(len(nodes))):
        raise ValueError("Expected sequential inventory indices")
    reserved = next(n for n in nodes if n["index"] == inventory_index)
    workers = [n for n in nodes if n is not reserved]
    remap = {n["index"]: i for i, n in enumerate(workers)}
    if reserved["address"] in {n["address"] for n in workers}:
        raise ValueError("Evaluation host must be dedicated")
    return reserved, dict(cluster, nodes=[dict(n, index=i) for i,n in enumerate(workers)],
                          actor_node=remap[cluster["actor_node"]],
                          service_cards=[[remap[i], g] for i,g in cluster["service_cards"]])


def validate_scoring_recipe(cfg):
    attempts = cfg["training"].get("score_attempts", 1)
    if type(attempts) is not int or attempts not in (1, 2):
        raise ValueError("score_attempts must be 1 or 2")
    protocol = cfg["reward_protocol"]
    if "event_decisions" in protocol and (
            protocol["event_decisions"] != "bounded-fields-v1" or protocol["version"] != "event-reward-v2"):
        raise ValueError("Unsupported event decision protocol")
    if "event_cache_backend" in cfg and (cfg["event_cache_backend"] != "dispatcher-v1"
            or not cfg.get("run_local_contract_cache") or protocol["version"] != "event-reward-v2"):
        raise ValueError("Dispatcher event cache requires run-local event-reward-v2")


def _validate_resume(checkpoint, cfg):
    from scrl.rl.identity import training_fingerprint

    state = read(checkpoint / "state.json")
    if state.get("format") != "split-policies-v1" or set(state["files"]) != {"planner.pt", "executor.pt"}:
        raise ValueError("Expected a paired planner/executor checkpoint")
    for name, expected in state["files"].items():
        if file_hash(checkpoint / name) != expected:
            raise ValueError(f"Damaged checkpoint: {name}")
    parent = read(checkpoint.parent.parent / "config.json")
    expected_reward = cfg['stamp']['reward']
    if training_fingerprint(parent) != training_fingerprint(cfg):
        from scrl.rl.identity import quote_anchor_transition
        cfg['quote_transition'] = quote_anchor_transition(parent, cfg)
        expected_reward = cfg['quote_transition']['old_reward']
    if state['stamp']['reward'] != expected_reward or any(
        cfg[k] != parent[k] for k in ("prompt_indices", "fixed_sampling", "batch_prompts", "sampling")
    ) or any(cfg["stamp"][k] != state["stamp"][k] for k in (
        "sampler", "planner_reference", "executor_reference"
    )):
        raise ValueError("Checkpoint training identity changed")


def compose(args):
    from scrl.rl.objectives import CROSSED_CREDIT, validate_planner_credit
    from scrl.rl.preflight import audit
    from scrl.runtime.launch import explicit_layout

    cfg = read(args.recipe)
    cfg.pop("preflight")  # 训练循环不读审核规则，preflight 会从同一份配方读取。
    if cfg["training"]["mode"] != "joint":
        raise ValueError("rl prepare requires joint planner/executor updates")
    paths, cluster = read(args.paths), runtime_cluster(read(args.cluster))
    watch = cfg.get("vbench_watch")
    reserved = cluster.get("reserved_inventory_node", watch["inventory_node"] if watch else None)
    if reserved is not None:
        if watch and reserved != watch["inventory_node"]:
            raise ValueError("VBench inventory node differs from the reserved node")
        cfg["evaluation_node"], cluster = reserve_evaluator(cluster, reserved)
    if watch:
        period = watch["every_steps"]
        if (period <= 0 or period % cfg["checkpoint_interval"] or
                cfg["checkpoint_keep_every"] != period):
            raise ValueError("Watched checkpoints must be published and retained")
    if args.rounds is not None and args.rounds < 1:
        raise ValueError("--rounds must be positive")
    validate_planner_credit(cfg["training"]["planner_credit"])
    sampling = cfg["sampling"]
    if cfg["fixed_sampling"] != [sampling["planner_candidates"], sampling["executions_per_candidate"]]:
        raise ValueError("Sampling group sizes disagree")
    if cfg["training"]["planner_credit"] == CROSSED_CREDIT and (
            sampling["planner_candidates"] != 4 or sampling["executions_per_candidate"] < 2):
        raise ValueError("Crossed training requires four cells and at least two shared seeds")
    assets = assets_from(paths)
    event_reward = cfg["reward"].get("mode") == "event-v2"
    if event_reward != (cfg["reward_protocol"]["version"] == "event-reward-v2"):
        raise ValueError("Event reward mode and judge protocol must match")
    validate_scoring_recipe(cfg)
    if event_reward and not assets["rewards"].get("judge"):
        raise ValueError("Event rewards require a video judge")
    root = Path(paths["asset_root"])
    source = root / paths["dataset"]
    if file_hash(source) != cfg["dataset_sha256"]:
        raise ValueError("Dataset bytes differ from the selected recipe")
    rows = read(source)
    if not isinstance(rows, list) or not rows or not all(
        isinstance(r, dict) and isinstance(r.get("prompt"), str) and r["prompt"].strip() for r in rows
    ):
        raise ValueError("Expected a nonempty JSON array of prompts")
    order = list(range(len(rows)))
    random.Random(cfg["seed"]).shuffle(order)
    manifest = code_manifest()
    cfg["reward_protocol"]["implementation_sha256"] = digest({
        k: v for k, v in manifest.items() if k == "scrl/rl/reward.py" or k.startswith("scrl/models/scoring/")
    })
    protocol = cfg["reward_protocol"]
    stamp = dict(planner=0, executor=0, sampler=cfg["sampler"],
                 reward=protocol["version"] + ":" + digest([cfg["reward"], protocol, assets["rewards"]]))
    layout = explicit_layout(cluster["nodes"], cluster["service_cards"], actor_node=cluster["actor_node"],
                             base_port=cluster["base_port"], dedicated=cfg.get("dedicated_services", False),
                             actor_samples=cfg.get("dedicated_services", False))
    cfg.update(layout | {
        "nodes": cluster["nodes"], "communication_env": cluster.get("communication_env", {}),
        "assets": assets, "python": str(root / paths["python"]),
        "role_python": {k: str(root / v) for k, v in paths.get("role_python", {}).items()},
        "source_root": str(SOURCE), "source_sha256": digest(manifest),
        "stamp": stamp, "prompt_source": str(source), "source_prompt_count": len(rows),
        "prompt_indices": order, "prompts": [rows[i]["prompt"] for i in order],
        "run": str(Path(args.run).resolve()), "resume": None, "init_from": None,
        "created_unix": time.time(), "validation_round_limit": args.rounds,
    })
    cfg["ports"] = {k: cfg[group][0]["port"] for k, group in (("evaluation", "evaluations"), ("planner", "planners"))}
    evidence = audit(cfg, args.approvals, recipe_path=args.recipe)
    if evidence["input_blockers"]:
        raise ValueError("联合训练输入未通过核对: " + json.dumps(evidence["input_blockers"], ensure_ascii=False))
    # fresh_origin 标识整份候选配置，包含运行目录和时间，不属于可恢复的策略身份。
    pins = {k: v for k, v in evidence["checked_sha256"].items() if k != "fresh_origin"}
    cfg["verified_inputs"] = pins
    cfg["asset_manifest_sha256"] = digest([evidence["spec_sha256"], assets, pins])
    stamp.update(planner_reference=pins["teacher_sft_planner"], executor_reference="longlive:" + pins["trainable_executor"])
    if args.resume:
        checkpoint = Path(args.resume).resolve(strict=True)
        _validate_resume(checkpoint, cfg)
        cfg["resume"] = str(checkpoint)
    return cfg


def prepare(args):
    from scrl.rl.identity import training_fingerprint

    cfg = compose(args)
    run = Path(cfg["run"])
    run.mkdir(parents=True, exist_ok=False)
    # 多节点的映射 UID 可能不同；收据和配置中不能写集群 token。
    run.chmod(0o777)
    # Nodes can have different mapped UIDs. Keep event files owned by their writer.
    logs = run / "logs"
    logs.mkdir()
    logs.chmod(0o1777)
    path = run / "config.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    receipt = run / "receipt.json"
    receipt.write_text(json.dumps({"config_sha256": file_hash(path), "source_sha256": cfg["source_sha256"],
                                   "training_fingerprint": training_fingerprint(cfg)}, indent=2) + "\n")
    for p in (path, receipt):
        p.chmod(0o644)
    print(f"已准备 {run}，{len(cfg['prompts'])} 条提示词，{cfg['executor_count']} 个 Executor；未启动进程。")


def load_run(path):
    receipt = read(Path(path).with_name("receipt.json"))
    if file_hash(path) != receipt["config_sha256"] or digest(code_manifest()) != receipt["source_sha256"]:
        raise ValueError("Code or config changed since prepare")
    cfg = read(path)
    token = os.environ.get("SCRL_RPC_TOKEN", "")
    if len(token) < 32:
        raise ValueError("Supply the same private SCRL_RPC_TOKEN to each authorized node")
    cfg["token"] = token
    return cfg
