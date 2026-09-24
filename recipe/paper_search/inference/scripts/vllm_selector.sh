#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SELECTOR_MODEL_PATH="${PAPERSEARCH_SELECTOR_MODEL_PATH:?Set PAPERSEARCH_SELECTOR_MODEL_PATH to the selector checkpoint}"

vllm serve "$SELECTOR_MODEL_PATH" \
    --served-model-name "${PAPERSEARCH_SELECTOR_MODEL_NAME:-selector}" \
    --tensor-parallel-size "${PAPERSEARCH_SELECTOR_TENSOR_PARALLEL_SIZE:-1}" \
    --gpu-memory-utilization "${PAPERSEARCH_SELECTOR_GPU_MEMORY_UTILIZATION:-0.9}" \
    --max-model-len "${PAPERSEARCH_SELECTOR_MAX_MODEL_LEN:-2048}" \
    --host "${PAPERSEARCH_SELECTOR_HOST:-0.0.0.0}" \
    --port "${PAPERSEARCH_SELECTOR_PORT:-8993}" \
    --task classify
