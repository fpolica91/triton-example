# Triton + vLLM + MiniMax-M2.5 — Status & Notes

## Current Setup
- **Machine**: 8x A100-40GB, 944GB RAM, AMD EPYC 7542
- **Model**: `mratsim/MiniMax-M2.5-FP8-INT4-AWQ` at `/data0/MiniMax-M2.5-FP8-INT4-AWQ` (123GB)
- **Working standalone vLLM**: `~/vllm_run.sh` — vLLM 0.16.0 on port 8002, tool calling works, Cloudflare tunnel works
- **Tunnel**: `cloudflared tunnel --url http://localhost:8002` → `https://aviation-pcs-river-ellis.trycloudflare.com`
- **GPU clocks**: Set to max with `sudo nvidia-smi -ac 1215,1410` (resets on reboot)

## What Works
1. **Standalone vLLM 0.16.0** (`~/vllm_run.sh`) — fully working with:
   - Tool calling (`--enable-auto-tool-choice --tool-call-parser minimax_m2`)
   - Continuous batching (`--max-num-seqs 32`)
   - 0.63s TTFB through Cloudflare tunnel
   - Tested and confirmed via curl

2. **llama.cpp** (the original setup) — works but single-slot, no batching

## Current Attempt: Triton Container
**Dockerfile** (`triton-serve-inference/Dockerfile`):
```
FROM nvcr.io/nvidia/tritonserver:26.02-vllm-python-py3
RUN pip install --no-cache-dir --no-deps compressed-tensors --upgrade
COPY patch_model.py /tmp/patch_model.py
RUN python3 /tmp/patch_model.py && rm /tmp/patch_model.py
```

**Strategy**: Keep Triton 26.02's vLLM 0.15.1 + PyTorch + flash-attn intact. Only upgrade `compressed-tensors` (pure Python) to add `block` FP8 strategy support that the model needs.

**Status**: FAILED. `compressed-tensors` upgrade alone is not enough. The `block` FP8 strategy requires refactored `create_weights()` logic that only exists in vLLM 0.16. Patching 0.15.1 is not viable — it's a full rewrite of the quantization scheme.

## Next Steps (for tomorrow)

### Option A: Full vLLM upgrade + flash-attn rebuild (single container, ~45 min build)
```dockerfile
FROM nvcr.io/nvidia/tritonserver:26.02-vllm-python-py3
RUN pip install vllm==0.16.0
RUN pip install flash-attn --no-build-isolation --force-reinstall
COPY patch_model.py /tmp/patch_model.py
RUN python3 /tmp/patch_model.py && rm /tmp/patch_model.py
```
This replaces NVIDIA's PyTorch but rebuilds flash-attn from source to match. Long build but should work.

### Option B: Triton OpenAI frontend as separate process
Use the built-in `/opt/tritonserver/python/openai/openai_frontend/main.py` which is a Python script (not the tritonserver binary). Could potentially run directly with the vLLM 0.16 image since it's pure Python. Needs investigation.

### Option C: Wait for Triton 26.03 (~3 weeks)
Will likely ship with vLLM 0.16.x natively.

## What Failed & Why

### 1. GGUF via vLLM (won't work)
- `transformers` library doesn't support `minimax-m2` GGUF architecture
- Error: `GGUF model with architecture minimax-m2 is not supported yet`

### 2. Full `pip install vllm==0.16.0` inside Triton 26.02
- Replaces NVIDIA's custom PyTorch 2.11 with pip's PyTorch 2.9
- Breaks flash-attn ABI: `undefined symbol: _ZN3c104cuda...`

### 3. `pip install vllm==0.16.0 --no-deps` inside Triton 26.02
- vLLM 0.16's `_C.abi3.so` compiled against different PyTorch
- Error: `libcudart.so.12: cannot open shared object file`

### 4. Multi-stage: Copy tritonserver from 25.01 into vLLM image
- 25.01 is Ubuntu 24.04 (glibc 2.39), vLLM image is Ubuntu 22.04 (glibc 2.35)
- Error: `GLIBC_2.36 not found`

### 5. Multi-stage: Copy tritonserver from 24.09 into vLLM image
- 24.09 is Ubuntu 22.04 (glibc matches!) but Python 3.10
- Python backend stub linked to `libpython3.10.so`, vLLM has 3.12
- Installing libpython3.10 → stub can't find numpy (wrong site-packages)

### 6. Mixed: 24.09 binary + 26.02 Python backend stub
- 26.02 stub needs glibc 2.38, vLLM image has 2.35 — same mismatch

## Key Discovery: Triton OpenAI Frontend
Triton has a built-in OpenAI-compatible frontend at `/opt/tritonserver/python/openai/`:
- Provides `/v1/chat/completions`, `/v1/completions`, etc.
- Supports `--tool-call-parser` (llama3, mistral built-in)
- Sits on top of the vLLM backend
- Docs: https://github.com/triton-inference-server/server/blob/main/python/openai/README.md

### TODO: Test this frontend
```bash
# Inside the Triton container:
cd /opt/tritonserver/python/openai
python3 openai_frontend/main.py \
  --model-repository /models \
  --tokenizer /data0/MiniMax-M2.5-FP8-INT4-AWQ
```
- Need to check if `minimax_m2` tool parser exists in 0.15.1 or if a custom one is needed
- May need to add custom parser or use model's native tool format

## Files
- `~/vllm_run.sh` — standalone vLLM (working, use as fallback)
- `~/triton-serve-inference/run.sh` — Triton docker run command
- `~/triton-serve-inference/Dockerfile` — current build
- `~/triton-serve-inference/patch_model.py` — strips tool-call args from model.json before AsyncEngineArgs
- `~/triton-serve-inference/model_repository/minimax-m2.5/1/model.json` — vLLM engine config
- `~/triton-serve-inference/model_repository/minimax-m2.5/config.pbtxt` — Triton model config (KIND_MODEL for multi-GPU)
- `~/triton.log` — latest Triton launch log

## NGC Auth
Logged into `nvcr.io` — pulls are fast now. Anonymous pulls were throttled.
