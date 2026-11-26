"""Exercise the real event-decision cache on a fresh shared directory, without a GPU.

Launch from identical source checkouts with the scorer Python and training mount/user:
  PYTHONPATH=src python tools/cache_contention_probe.py init /shared/NEW --ranks 2 \
      --backend dispatcher-v1 --address PRIVATE_BROKER_HOST --port UNUSED_PORT
  PYTHONPATH=src python tools/cache_contention_probe.py broker /shared/NEW
  PYTHONPATH=src python tools/cache_contention_probe.py worker /shared/NEW --rank 0
  PYTHONPATH=src python tools/cache_contention_probe.py worker /shared/NEW --rank 1
  PYTHONPATH=src python tools/cache_contention_probe.py check /shared/NEW

Use an isolated port on a trusted private network. Supply the SAME fresh, probe-only
SCRL_RPC_TOKEN privately to broker and workers. Do not reuse a training service/token.
Keep broker in the foreground on PRIVATE_BROKER_HOST; stop it with Ctrl-C after check.
Run the two workers concurrently on DISTINCT scoring nodes; run init and check once.
Use a restricted parent with appropriate shared write permissions/umask for all workers.
For local mechanics only, pass --min-hosts 1 to check. No existing directory is reused.
For a legacy file-lock comparison use --backend file-lock instead, omit address/port
and broker, and start with another new directory.
"""

import argparse
import json
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from scrl.models.scoring import events
from scrl.runtime.transport import EvaluationDispatcher, atomic_json, serve


def read(path):
    return json.loads(path.read_bytes())


def broker(root, cfg):
    if cfg["backend"] != "dispatcher-v1":
        raise ValueError("broker requires --backend dispatcher-v1 at init")
    dispatcher = EvaluationDispatcher({"run": root, "evaluations": [], "event_cache_backend": "dispatcher-v1"})

    def handle(op, request):
        if op == "probe_identity":
            return {"probe_id": cfg["probe_id"]}
        if op not in ("event_cache_acquire", "event_cache_abandon", "event_cache_publish"):
            raise ValueError("Not a cache probe operation")
        return dispatcher(op, request)

    token = os.environ["SCRL_RPC_TOKEN"]
    if len(token) < 16:
        raise ValueError("Set a fresh private probe token of at least 16 characters")
    serve(cfg["endpoint"]["port"], token, handle, concurrent=True)


def worker(root, rank, cfg, timeout):
    ranks, rounds = cfg["ranks"], cfg["rounds"]
    if not 0 <= rank < ranks:
        raise ValueError("rank outside initialized range")
    mine = root / f"rank-{rank}"
    mine.mkdir()
    judge = SimpleNamespace(cache=root / "prompt_contracts")
    if cfg["backend"] == "dispatcher-v1":
        from scrl.models.scoring.cache_client import cache_peer
        peer = cache_peer({"event_cache_endpoint": cfg["endpoint"]})
        deadline = time.monotonic() + timeout
        while True:
            try:
                identity = peer.call("probe_identity")
                break
            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.1)
        if identity != {"probe_id": cfg["probe_id"]}:
            raise ValueError("RPC endpoint is not this probe")
        judge.event_cache_peer = peer
    host = socket.gethostname()
    results = []
    for index in range(rounds):
        flag = f"ready-{index}"
        (mine / flag).touch(exist_ok=False)
        deadline = time.monotonic() + timeout
        while not all((root / f"rank-{r}" / flag).is_file() for r in range(ranks)):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"rank {rank}: barrier {index} timed out")
            time.sleep(.02)

        def generate(*args):
            token = uuid4().hex
            # Independent receipts expose duplicate generation even if last writer wins.
            (mine / f"computed-{index}-{token}").touch(exist_ok=False)
            time.sleep(.2)
            return {"nonce": token}

        payload = {"probe_round": index}
        with patch.object(events, "_generate", generate):
            value = events._cached(judge, "probe", payload, lambda x: x)
            cached = events._cached(judge, "probe", payload, lambda x: x)
        if cached != value:
            raise ValueError(f"rank {rank}: cache hit disagrees with initial read in round {index}")
        results.append(value["nonce"])
    atomic_json(mine / "result.json", {"host": host, "values": results})


def check(root, cfg, min_hosts):
    ranks, rounds = cfg["ranks"], cfg["rounds"]
    entries = [read(root / f"rank-{rank}" / "result.json") for rank in range(ranks)]
    hosts = sorted({entry["host"] for entry in entries})
    if len(hosts) < min_hosts:
        raise ValueError(f"expected {min_hosts} distinct hosts, found {hosts}")
    if any(len(e["values"]) != rounds for e in entries):
        raise ValueError("incomplete worker receipt")
    for index in range(rounds):
        generated = [p.name.rsplit("-", 1)[1] for p in root.glob(f"rank-*/computed-{index}-*")]
        if len(generated) != 1 or any(e["values"][index] != generated[0] for e in entries):
            raise ValueError(f"round {index}: {len(generated)} generation(s), or reader disagreement")
    if cfg["backend"] == "dispatcher-v1":
        cache = root / "prompt_contracts" / events.VERSION
        canonical = {r["value"]["nonce"] for r in map(read, (cache / "dispatcher-v1").glob("*.json"))}
        if list(cache.rglob("*.lock")) or canonical != set(entries[0]["values"]):
            raise ValueError("Expected lock-free dispatcher publications matching every generated decision")
    print(json.dumps({"state": "passed", "backend": cfg["backend"], "hosts": hosts, "ranks": ranks, "rounds": rounds}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    server = commands.add_parser("broker")
    run = commands.add_parser("worker")
    verify = commands.add_parser("check")
    for command in (init, server, run, verify):
        command.add_argument("directory", type=Path)
    init.add_argument("--ranks", type=int, required=True)
    init.add_argument("--rounds", type=int, default=24)
    init.add_argument("--backend", choices=["file-lock", "dispatcher-v1"], required=True)
    init.add_argument("--address")
    init.add_argument("--port", type=int)
    run.add_argument("--rank", type=int, required=True)
    run.add_argument("--barrier-timeout", type=float, default=120.)
    verify.add_argument("--min-hosts", type=int, default=2)
    args = parser.parse_args()
    root = args.directory
    if args.command == "init":
        if args.ranks < 2 or args.rounds < 1:
            parser.error("need at least two ranks and one round")
        cfg = {"ranks": args.ranks, "rounds": args.rounds, "backend": args.backend, "probe_id": uuid4().hex}
        if args.backend == "dispatcher-v1":
            if not args.address or not args.port or not 1 <= args.port <= 65535:
                parser.error("dispatcher-v1 requires --address and a dedicated --port")
            cfg["endpoint"] = {"address": args.address, "port": args.port}
        elif args.address is not None or args.port is not None:
            parser.error("file-lock does not use --address or --port")
        root.mkdir()
        atomic_json(root / "probe.json", cfg)
    elif args.command == "broker":
        broker(root, read(root / "probe.json"))
    elif args.command == "worker":
        worker(root, args.rank, read(root / "probe.json"), args.barrier_timeout)
    else:
        if args.min_hosts < 1:
            parser.error("--min-hosts must be positive")
        check(root, read(root / "probe.json"), args.min_hosts)


if __name__ == "__main__":
    main()
