#!/usr/bin/env bash
# Optional: serve the policy with ``vllm serve`` for manual HTTP experiments only.
# Batch inference uses in-process vLLM via ``run_paper_agent.py`` (see inference/.env.example).

set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MODEL_PATH="${PAPER_SEARCH_INFERENCE_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"

vllm serve "$MODEL_PATH" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-10240}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
  --port "${PAPER_SEARCH_INFERENCE_PORT:-8998}" \
  --served-model-name "${PAPER_SEARCH_INFERENCE_SERVED_MODEL_NAME:-paper-search-policy}"
