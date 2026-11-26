"""Run on the reserved VBench host; consume atomically published checkpoints."""
import argparse
import fcntl
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from scrl.config import DEFAULT_CONFIGS, file_hash, read
from scrl.plan import digest
from scrl.rl.cycle import Fleet, evaluate
from scrl.runtime.transport import atomic_json


class LocalFleet(Fleet):
    def run(self, address, argv, log, *, env=None):
        if address != self.cfg["nodes"][0]["address"]:
            raise ValueError("Unexpected remote evaluation worker")
        settings = dict(os.environ, PYTHONPATH=self.cfg["source_root"],
                        HF_HUB_OFFLINE="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        settings.update(env or {})
        with log.open("a") as output:
            subprocess.run(list(map(str, argv)), check=True, env=settings, stdout=output, stderr=subprocess.STDOUT)


def published(directory, period):
    result = []
    for cp in directory.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", cp.name)
        if match and (step := int(match[1])) > 0 and step % period == 0:
            if all((cp/name).is_file() for name in ("state.json", "planner.pt", "executor.pt")):
                result.append((step, cp))
    return sorted(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--once", action="store_true", help="Drain current backlog and exit")
    parser.add_argument("--step-zero", action="store_true", help="Evaluate initial adapters before checkpoints")
    args = parser.parse_args(argv)
    path = args.run/"config.json"
    receipt = read(args.run/"receipt.json")
    from scrl.config import code_manifest
    if file_hash(path) != receipt["config_sha256"]:
        raise ValueError("Use the prepared run")
    cfg = read(path)
    live = code_manifest()
    if digest(live) != receipt["source_sha256"]:
        source = Path(cfg["source_root"])
        pinned = {"scrl/"+str(p.relative_to(source)):file_hash(p)
                  for p in source.rglob("*.py") if p != source/"scrl.py"}
        changed = {k for k in live.keys() | pinned.keys() if live.get(k) != pinned.get(k)}
        if (digest(pinned) != receipt["source_sha256"] or
                not changed <= {"scrl/rl/watch.py", "scrl/rl/cycle.py"}):
            raise ValueError("Only the evaluation scheduler may differ from the pinned release")
    node = cfg["evaluation_node"]
    with socket.socket() as sock:
        sock.bind((node["address"], 0))
    settings = dict(cfg["vbench_watch"], score_node=0)
    quality = cfg["assets"]["rewards"]["quality"]
    settings.update(recipe=str(DEFAULT_CONFIGS/settings["recipe"]),
                    prompts=str(DEFAULT_CONFIGS/settings["prompts"]), source=quality["source"],
                    metadata=str(Path(quality["source"])/settings["metadata_relative_to_vbench"]),
                    weights=str((Path(quality["weights"])/settings["weights_relative_to_training_weights"]).resolve()))
    directory = Path(cfg["run"])/"checkpoints"
    base = Path(settings["local_root"])
    out = base/(args.run.name+"-"+receipt["config_sha256"][:16])
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg, nodes=[node])
    fleet = LocalFleet(cfg, None)
    with (base/"watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = set()
        while True:
            queue = ([(0, None)] if args.step_zero else []) + published(directory, settings["every_steps"])
            for step, cp in queue:
                if step in done:
                    continue
                try:
                    atomic_json(out/"status.json", dict(state="evaluating", step=step))
                    evaluate(cfg, fleet, cp, step, settings, output_root=out)
                except Exception as exc:
                    atomic_json(out/"status.json", dict(state="retry", step=step, error=str(exc)))
                    if args.once:
                        raise
                    time.sleep(30)
                    break
                done.add(step)
            else:
                atomic_json(out/"status.json", dict(state="idle", completed=sorted(done)))
                if args.once:
                    return
                time.sleep(5)
