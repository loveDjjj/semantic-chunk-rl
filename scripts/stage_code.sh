#!/bin/sh
# Export the working source tree, including uncommitted code, into a fresh release.
set -eu
if [ "$#" != 1 ]; then
    printf '%s\n' 'Usage: sh scripts/stage_code.sh NEW_RELEASE_DIRECTORY' >&2
    exit 2
fi
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
dest=$(realpath -m -- "$1")
case "$dest/" in "$root/"*) printf '%s\n' 'Choose a destination outside the checkout.' >&2; exit 2;; esac
umask 022
mkdir -- "$dest"  # Never merge into an old release or a live checkout.
cd "$root"
cp --parents -t "$dest" -- LICENSE README.md THIRD_PARTY_NOTICES.md pyproject.toml \
    docs/methods.md docs/running.md docs/results.md docs/changelog.md datasets/README.md tools/cache_contention_probe.py \
    configs/rl.json configs/rl-whole24.json configs/sft.json configs/sft-selection.json configs/vbench.json configs/cluster.example.json \
    configs/paths.example.json configs/launch.example.json configs/vbench-periodic-prompts.json \
    scripts/rl.sh scripts/sft.sh scripts/select_sft_prompts.sh scripts/vbench.sh scripts/serve_video_judge.sh
find src -type f -name '*.py' -exec cp --parents -t "$dest" -- {} +
inventory=${SCRL_CLUSTER_INVENTORY:-configs/cluster.json}
topology=${SCRL_CLUSTER_LAYOUT:-configs/cluster.example.json}
if [ -f "$inventory" ]; then
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/src" "${SCRL_PYTHON:-python3}" -c \
        'import json,sys; from scrl.config import read,stage_cluster; print(json.dumps(stage_cluster(read(sys.argv[1]), None if sys.argv[2] == "inventory" else read(sys.argv[2])), indent=2))' "$inventory" "$topology" \
        > "$dest/configs/cluster.runtime.json"
elif [ -n "${SCRL_CLUSTER_INVENTORY:-}" ]; then
    printf '%s\n' 'Specified cluster inventory is missing.' >&2; exit 2
fi
cd "$dest"
find src configs datasets docs scripts tools LICENSE README.md THIRD_PARTY_NOTICES.md pyproject.toml \
    -type f -exec sha256sum {} + > .manifest.tmp
LC_ALL=C sort -k2 -o .manifest.tmp .manifest.tmp
sha256sum --check --status .manifest.tmp
mv -- .manifest.tmp SHA256SUMS
printf 'Release: %s\n' "$dest"
