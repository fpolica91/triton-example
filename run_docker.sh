#!/bin/bash
docker run --gpus all --net=host --rm -d \
  --shm-size=1G \
  --name triton-minimax \
  -e SAFETENSORS_FAST_GPU=1 \
  -v /data0:/data0 \
  -v /home/ubuntu/triton-serve-inference/model_repository:/models \
  triton-minimax-nightly:latest \
  python3 /opt/tritonserver/python/openai/openai_frontend/main.py \
    --model-repository /models \
    --tokenizer /data0/MiniMax-M2.5-FP8-INT4-AWQ \
    --tool-call-parser minimax_m2 \
    --openai-port 8002
