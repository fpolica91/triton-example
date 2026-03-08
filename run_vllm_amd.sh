#!/bin/bash
# MiniMax-M2.5 vLLM server for AMD MI300X
# Based on official vLLM deploy guide: https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/docs/vllm_deploy_guide.md
# With AMD ROCm workarounds for missing iommu=pt kernel parameter

# --- AMD ROCm workarounds (remove these if iommu=pt is added to kernel cmdline) ---
export NCCL_IPC_DISABLE=1
export NCCL_P2P_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RCCL_MSCCL_ENABLE=0
export MSCCL_ENABLE=0
export NCCL_MSCCLPP_ENABLE=0

# --- Performance tuning ---
export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

MODEL_PATH="/data0/huggingface/MiniMax-M2.5"
LOG_FILE="/home/ubuntu/triton-example/vllm.log"
PORT=8002

echo "Starting vLLM server for MiniMax-M2.5 on AMD MI300X..."
echo "Model: ${MODEL_PATH}"
echo "Port: ${PORT}"
echo "Log: ${LOG_FILE}"

nohup vllm serve "${MODEL_PATH}" \
  --served-model-name minimax-m2.5 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 131072 \
  --max-num-seqs 32 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2 \
  --disable-custom-all-reduce \
  --enforce-eager \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "vLLM started in background. PID: ${PID}"
echo "${PID}" > /home/ubuntu/triton-example/vllm.pid
echo "Logs: tail -f ${LOG_FILE}"
