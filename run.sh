#!/bin/bash
docker run --gpus all --net=host --rm \
  --shm-size=16G \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
  -v /data0:/data0 \
  -v /home/ubuntu/triton-serve-inference/model_repository:/models \
  nvcr.io/nvidia/tritonserver:26.02-vllm-python-py3 \
  tritonserver \
    --model-repository /models \
    --http-port 8001 \
    --grpc-port 8002 \
    --metrics-port 8003
