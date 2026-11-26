"""Check the entire proposed Git snapshot, not working files or prior history."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys


RULES = Path(__file__).resolve().parents[1] / ".gitignore"
MAX_BYTES = 2 * 1024 * 1024  # Code and docs only; distribute artifacts separately.
KEY_HEADER = re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")
TOKEN = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{50,}|sk-[A-Za-z0-9_-]{24,})")
PRIVATE_FIELDS = {"password", "api_key", "access_token", "private_key", "rpc_token", "workspace_by_password"}


def credentials(value):
    if isinstance(value, dict):
        return any((k.lower() in PRIVATE_FIELDS and v) or credentials(v) for k, v in value.items())
    return isinstance(value, list) and any(credentials(v) for v in value)


def git(*args, data=None):
    return subprocess.check_output(["git", *args], input=data)


def main():
    os.chdir(os.fsdecode(git("rev-parse", "--show-toplevel").rstrip(b"\n")))
    # Use the project rules explicitly, not a developer's global Git excludes.
    forbidden = set(git("ls-files", "-ci", "-z", "--exclude-from=" + str(RULES)).split(b"\0"))
    entries = git("ls-files", "--stage", "-z").split(b"\0")[:-1]
    if not entries:
        print("Nothing in the Git index to check.", file=sys.stderr)
        return 1
    blobs = {}
    errors = []
    for entry in entries:
        header, name = entry.split(b"\t", 1)
        mode, oid, stage = header.split()
        if name in forbidden:
            errors.append((name, "local-only path"))
        elif stage != b"0" or mode not in (b"100644", b"100755"):
            errors.append((name, "unmerged or non-regular entry"))
        else:
            blobs.setdefault(oid, []).append(name)
    if blobs:
        sizes = git("cat-file", "--batch-check=%(objectname) %(objectsize)",
                    data=b"\n".join(blobs) + b"\n").splitlines()
        for row in sizes:
            oid, size = row.split()
            names = blobs[oid]
            if int(size) > MAX_BYTES:
                errors.extend((n, "blob exceeds 2 MiB") for n in names)
                continue
            data = git("cat-file", "blob", oid)
            if KEY_HEADER.search(data) or TOKEN.search(data):
                errors.extend((n, "credential signature in staged content") for n in names)
            for n in names:
                if n.lower().endswith(b".json"):
                    try:
                        has_credentials = credentials(json.loads(data))
                    except (ValueError, UnicodeError):
                        errors.append((n, "invalid JSON"))
                    else:
                        if has_credentials:
                            errors.append((n, "nonempty credential field in staged JSON"))
    for name, reason in sorted(errors):
        print(f"{os.fsdecode(name)!r}: {reason}", file=sys.stderr)
    if errors:
        return 1
    print(f"Checked {len(entries)} indexed files. Earlier commits were not checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
