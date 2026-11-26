#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${SCRL_PYTHON:-python3}" -m scrl sft "$@"
