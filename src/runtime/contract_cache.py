"""Single-dispatcher authority for immutable event decisions in one training run."""

import hashlib
import json
from pathlib import Path
import re
import threading
from uuid import UUID

from scrl.runtime.transport import atomic_json


_HEX_KEY = re.compile(r"[0-9a-f]{64}")


def _publisher(owner):
    if not isinstance(owner, str):
        raise ValueError("Expected an owner UUID")
    uid = UUID(owner)
    if owner not in (uid.hex, str(uid)):
        raise ValueError("Expected a canonical owner UUID")
    return hashlib.sha256(uid.bytes).hexdigest()


class EventCacheAuthority:
    def __init__(self, run):
        self.directory = Path(run) / "prompt_contracts" / "event-reward-v2" / "dispatcher-v1"
        self.directory.mkdir(parents=True, exist_ok=True)
        # Stripes keep slow storage I/O for one key off the dispatcher's scheduling lock.
        self.stripes = [(threading.Lock(), {}) for _ in range(64)]

    def __call__(self, op, request):
        key = request["key"]
        if not isinstance(key, str) or _HEX_KEY.fullmatch(key) is None:
            raise ValueError("Expected a SHA256 cache key")
        publisher = _publisher(request["owner"])
        if op not in ("event_cache_acquire", "event_cache_publish", "event_cache_abandon"):
            raise ValueError(op)
        lock, pending = self.stripes[int(key[:2], 16) % len(self.stripes)]
        path = self.directory / (key + ".json")
        with lock:
            current = pending.get(key)
            if op == "event_cache_acquire" and current is not None:
                return {"state": "owner" if current == publisher else "wait"}
            if op == "event_cache_abandon":
                # A delayed retry after a lost reply must not release the next owner's claim.
                if current == publisher:
                    del pending[key]
                return {"state": "abandoned"}
            try:
                receipt = json.loads(path.read_bytes())
            except FileNotFoundError:
                receipt = None
            else:
                if (not isinstance(receipt, dict) or set(receipt) != {"publisher_sha256", "value"}
                        or not isinstance(receipt["value"], dict)
                        or not isinstance(receipt["publisher_sha256"], str)
                        or not _HEX_KEY.fullmatch(receipt["publisher_sha256"])):
                    raise ValueError("Invalid canonical cache receipt")
            if op == "event_cache_publish":
                value = request["value"]
                if not isinstance(value, dict):
                    raise ValueError("Expected a validated event decision object")
                if receipt is None:
                    if current != publisher:
                        raise ValueError("Stale or unauthorized cache publisher")
                    receipt = {"publisher_sha256": publisher, "value": value}
                    atomic_json(path, receipt)
                elif (receipt["publisher_sha256"] != publisher or
                      json.dumps(value, sort_keys=True, allow_nan=False) !=
                      json.dumps(receipt["value"], sort_keys=True, allow_nan=False)):
                    raise ValueError("Conflicting or unauthorized cache publication")
                pending.pop(key, None)
            if receipt is not None:
                return {"state": "hit", "value": receipt["value"]}
            pending[key] = publisher
            return {"state": "owner"}
