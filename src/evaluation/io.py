"""Evaluation file primitives, independent of model and training environments."""
import json
import os
from pathlib import Path
import tempfile

from scrl.config import file_hash as file_sha256


def read(path):
    return json.loads(Path(path).read_bytes())


def settings():
    from scrl.config import DEFAULT_CONFIGS
    return read(DEFAULT_CONFIGS/'vbench.json')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
