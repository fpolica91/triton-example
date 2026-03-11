#!/bin/bash
# GLM-5-FP8 vLLM server for AMD MI300X (gfx942)
# Ref: https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM5.html

# --- RCCL: DMA-BUF P2P IPC (kernel 6.8.12-no-p2p-no-p2p-v2: HSA_AMD_P2P=n, DMABUF_MOVE_NOTIFY=y) ---
export NCCL_DMABUF_ENABLE=1
export HSA_ENABLE_IPC_MODE_LEGACY=0

# --- AMD ROCm settings ---
export HSA_FORCE_FINE_GRAIN_PCIE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RCCL_MSCCL_ENABLE=0
export MSCCL_ENABLE=0
export NCCL_MSCCLPP_ENABLE=0

# --- AMD MI300X performance settings (firmware 137 < 177, so HSA_NO_SCRATCH_RECLAIM needed) ---
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_MLA=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

# --- Memory allocator ---
export SAFETENSORS_FAST_GPU=1

MODEL_PATH="/data0/huggingface/GLM-5-FP8"
LOG_FILE="/home/ubuntu/triton-example/vllm.log"
PORT=8002

echo "Starting vLLM server for GLM-5-FP8 on AMD MI300X..."
echo "Model: ${MODEL_PATH}"
echo "Port: ${PORT}"
echo "Log: ${LOG_FILE}"

nohup vllm serve "${MODEL_PATH}" \
  --served-model-name glm-5-fp8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 131072 \
  --max-num-seqs 32 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --disable-log-request \
  --block-size 1 \
  --enforce-eager \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "vLLM started in background. PID: ${PID}"
echo "${PID}" > /home/ubuntu/triton-example/vllm.pid
echo "Logs: tail -f ${LOG_FILE}"
