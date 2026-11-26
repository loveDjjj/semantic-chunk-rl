"""Per-process structured operational logs; payloads and environment are never serialized."""
from datetime import datetime, timezone
from functools import lru_cache
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import socket


@lru_cache(maxsize=32)
def _writer(directory, host, pid):
    directory.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(directory / f"{host}-{pid}.jsonl", maxBytes=8 << 20,
                                  backupCount=3, encoding="utf-8")
    logger = logging.Logger("scrl.events")
    logger.addHandler(handler)
    return logger


def event(component, state, *, run=None, node=None, rank=None, step=None, error_type=None):
    directory = os.environ.get("SCRL_LOG_DIR") or (Path(run) / "logs" if run else None)
    if not directory:
        return
    host, pid = socket.gethostname(), os.getpid()
    record = dict(time=datetime.now(timezone.utc).isoformat(), host=host, pid=pid,
                  component=component, state=state, node=node, rank=rank, step=step,
                  error_type=error_type)
    _writer(Path(directory).absolute(), host, pid).info(json.dumps(
        {k: v for k, v in record.items() if v is not None}, ensure_ascii=True))
