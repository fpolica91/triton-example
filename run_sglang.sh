#!/bin/bash
# GLM-5-FP8 on AMD MI300X via SGLang + TileLang
# Requirements:
#   - Docker with ROCm support
#   - 8x AMD Instinct MI300X (gfx942)
#   - Model weights at /data0/huggingface/GLM-5-FP8
#
# Notes:
#   - The Docker image has ROCm aiter source at /sgl-workspace/aiter but it needs
#     pip install -e . to activate (handled in entrypoint)
#   - transformers>=5.3.0 required for glm_moe_dsa architecture (handled in entrypoint)
#   - No --block-size 1 or --enforce-eager needed (unlike vLLM)
#   - TileLang handles GLM-5 DSA sparse attention natively on MI300X
#
# Performance: ~12-15 tok/s single request on 8x MI300X

set -e

MODEL_PATH="${MODEL_PATH:-/data0/huggingface/GLM-5-FP8}"
PORT="${PORT:-8002}"
LOG_FILE="$(dirname "$0")/sglang.log"

echo "Starting SGLang GLM-5-FP8 server..."
echo "Model: ${MODEL_PATH}"
echo "Port:  ${PORT}"
echo "Log:   ${LOG_FILE}"

docker rm -f sglang-glm5 2>/dev/null || true

docker run -d \
  --name sglang-glm5 \
  --restart unless-stopped \
  --network host \
  --ipc host \
  --shm-size 64g \
  --cap-add SYS_PTRACE \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  -v /data0:/data0 \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e PORT="${PORT}" \
  rocm/sgl-dev:v0.5.8.post1-rocm720-mi30x-20260219 \
  bash -c "
    cd /sgl-workspace/aiter && pip install -e . -q &&
    pip install 'transformers>=5.3.0' -q &&
    python -m sglang.launch_server \
      --model \$MODEL_PATH \
      --tp 8 \
      --trust-remote-code \
      --nsa-prefill-backend tilelang \
      --nsa-decode-backend tilelang \
      --chunked-prefill-size 131072 \
      --mem-fraction-static 0.80 \
      --watchdog-timeout 1200 \
      --served-model-name glm-5-fp8 \
      --host 0.0.0.0 \
      --port \$PORT
  "

echo "Container started. Following logs (Ctrl+C to detach)..."
docker logs -f sglang-glm5 2>&1 | tee "${LOG_FILE}"
