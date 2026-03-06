#!/bin/bash
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export PYTHONPATH=/home/ubuntu/.local/lib/python3.12/site-packages

/usr/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model /data0/MiniMax-M2.5-FP8-INT4-AWQ \
  --served-model-name MiniMax-M2.5 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --host 0.0.0.0 \
  --port 8002 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2
