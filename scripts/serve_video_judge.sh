#!/usr/bin/env bash
# Start one loopback-only judge replica on exclusively allocated node-local GPUs.
set -euo pipefail
devices=${1:?usage: bash serve_video_judge.sh GPU_IDS [PORT]}
port=${2:-8041}
python=${JUDGE_PYTHON:-${JUDGE_ENV:+$JUDGE_ENV/bin/python}}
: "${python:?set JUDGE_PYTHON or JUDGE_ENV}" "${JUDGE_MODEL:?set JUDGE_MODEL}"
min_free=${JUDGE_MIN_FREE_MIB:-76000}
[[ $min_free =~ ^[1-9][0-9]*$ ]] || { echo "Invalid JUDGE_MIN_FREE_MIB" >&2; exit 2; }
[[ $devices =~ ^[0-9]+(,[0-9]+)?$ ]] || exit 2
IFS=, read -ra ids <<< "$devices"
[[ ${#ids[@]} == 1 || ${ids[0]} != "${ids[1]}" ]] || exit 2
for gpu in "${ids[@]}"; do
    processes=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
    free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    [[ $free =~ ^[0-9]+$ ]] || { echo "Unrecognized GPU memory report" >&2; exit 1; }
    if [[ -n $processes && $processes != "${JUDGE_BASELINE_PID:-}" ]] || ((free < min_free)); then
        printf 'GPU %s is not exclusively available (%s MiB free); not starting.\n' "$gpu" "$free" >&2
        exit 1
    fi
done
export CUDA_VISIBLE_DEVICES="$devices" HF_HUB_OFFLINE=1
envdir=$(dirname "$(dirname "$python")")
for toolkit in "${JUDGE_CUDA_HOME:-$envdir}"/bin/nvcc "$envdir"/lib/python*/site-packages/nvidia/cu13/bin/nvcc; do
    if [[ -x "$toolkit" ]]; then
        CUDA_HOME=$(dirname "$(dirname "$toolkit")")
        export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
        break
    fi
done
export VLLM_USE_FLASHINFER_SAMPLER=0
exec "$python" -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" --served-model-name rl-video-judge --host 127.0.0.1 --port "$port" \
    --dtype bfloat16 --tensor-parallel-size "${#ids[@]}" --max-model-len 6400 \
    --max-num-seqs "${JUDGE_SEQUENCES:-6}" --enforce-eager --gdn-prefill-backend triton \
    --kv-cache-memory-bytes 4294967296 --gpu-memory-utilization 0.75
