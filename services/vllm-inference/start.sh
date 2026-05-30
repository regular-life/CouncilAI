#!/bin/bash
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────
MODEL="${VLLM_MODEL_NAME:-microsoft/Phi-4-mini-instruct}"
DTYPE="${VLLM_DTYPE:-auto}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.85}"
QUANTIZATION="${VLLM_QUANTIZATION:-}"
TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"
SWAP_SPACE="${VLLM_SWAP_SPACE_GB:-0}"

# Laptop / low-resource scaling options
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-auto}"
CPU_OFFLOAD="${VLLM_CPU_OFFLOAD_GB:-0}"

# ── Build command ────────────────────────────────────────────────────
CMD=(
    python -m vllm.entrypoints.openai.api_server
    --model "$MODEL"
    --host 0.0.0.0
    --port 8001
    --dtype "$DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-num-seqs "$MAX_NUM_SEQS"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --trust-remote-code
)

# Optional: quantization
if [ -n "$QUANTIZATION" ] && [ "$QUANTIZATION" != "none" ]; then
    CMD+=(--quantization "$QUANTIZATION")
fi

# Optional: tensor parallelism
if [ "$TENSOR_PARALLEL" -gt 1 ] 2>/dev/null; then
    CMD+=(--tensor-parallel-size "$TENSOR_PARALLEL")
fi

# Optional: swap space
if [ "$SWAP_SPACE" -gt 0 ] 2>/dev/null; then
    CMD+=(--swap-space "$SWAP_SPACE")
fi

# Optional: CPU offloading
if [ "$CPU_OFFLOAD" -gt 0 ] 2>/dev/null; then
    CMD+=(--cpu-offload-gb "$CPU_OFFLOAD")
fi

echo "▶ Starting vLLM: ${CMD[*]}"
exec "${CMD[@]}"
