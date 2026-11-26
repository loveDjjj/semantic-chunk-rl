#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
exec "${SCRL_VBENCH_PYTHON:?set to the VBench environment python}" -m scrl.evaluation.score_when_ready "$@"
