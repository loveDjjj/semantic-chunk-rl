"""Resolve text decisions through the run's sole cache publisher, without filesystem locks."""

import os
import time
from uuid import uuid4

from scrl.runtime.transport import Peer


def cache_peer(settings):
    endpoint = settings.get("event_cache_endpoint")
    if endpoint is None:
        return None
    return Peer(endpoint["address"], endpoint["port"], os.environ["SCRL_RPC_TOKEN"], timeout=20)


def single_flight(peer, key, generate, validate, wait_seconds=180):
    owner = uuid4().hex
    def call(op, **extra):
        # Repeat the same ownership token if the reply (not the operation) was lost.
        for attempt in range(2):
            try:
                return peer.call("event_cache_" + op, key=key, owner=owner, **extra)
            except (EOFError, ConnectionError, TimeoutError):
                if attempt:
                    raise

    deadline = time.monotonic() + wait_seconds
    delay = .1
    while True:
        reply = call("acquire")
        if reply["state"] == "hit":
            return validate(reply["value"])
        if reply["state"] == "owner":
            break
        if reply["state"] != "wait":
            raise RuntimeError("Unexpected event cache response")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for event cache owner: {key}")
        time.sleep(delay)
        delay = min(delay * 2, 1.)
    try:
        value = generate()
    except BaseException:
        call("abandon")
        raise
    published = call("publish", value=value)
    if published["state"] != "hit":
        raise RuntimeError("Event cache did not confirm publication")
    return validate(published["value"])
