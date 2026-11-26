#!/usr/bin/env bash
# Install an offline, node-local Conda prefix from the verified shared release.
set -euo pipefail
role=${1:?expected planner-4b, judge-27b, or self-forcing}
deps=${SCRL_DEPS:?set SCRL_DEPS to the verified offline dependency directory}
case "$role" in
    planner-4b|judge-27b) base=qwen-runtime-v1 ;;
    self-forcing) base=self-forcing-v1 ;;
    *) exit 2 ;;
esac
umask 077
root=${SCRL_ENV_ROOT:-"$HOME/.cache/semantic-chunk-rl/envs"}
mkdir -p "$root"
dst="$root/$role-v1"
exec 9>"$root/.$role.lock"
flock 9
if [[ -f "$dst/.scrl-ready" ]]; then
    printf '%s\n' "$dst/bin/python"
    exit 0
fi
[[ ! -e "$dst" ]] || { echo "Incomplete prefix: $dst" >&2; exit 1; }
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
cd "$deps/envs"
sha256sum --check --status "$base.tar.sha256"
mkdir "$dst"
trap 'rm -rf -- "$dst"' ERR
tar --no-same-owner -xf "$base.tar" -C "$dst"
"$dst/bin/python" "$dst/bin/conda-unpack"
if [[ "$role" == planner-4b ]]; then
    sha256sum --check --status wheels.sha256
    "$dst/bin/python" -m pip install --no-index --no-deps wheels/*.whl
    "$dst/bin/python" -c 'import accelerate, peft; from transformers import Qwen3_5ForConditionalGeneration'
elif [[ "$role" == judge-27b ]]; then
    "$dst/bin/python" -c 'from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration'
else
    "$dst/bin/python" -c 'import diffusers, omegaconf, torch'
fi
touch "$dst/.scrl-ready"
printf '%s\n' "$dst/bin/python"
