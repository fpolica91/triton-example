#!/usr/bin/env python3
"""
Patch Triton's vLLM backend:
1. model.py - strip tool-call args from model.json before AsyncEngineArgs
2. compressed_tensors_w8a16_fp8.py - add BLOCK strategy support
"""

# Patch 1: model.py tool-call args
BACKEND_PATH = "/opt/tritonserver/backends/vllm/model.py"
with open(BACKEND_PATH, "r") as f:
    content = f.read()

PATTERNS = [
    ('        aync_engine_args = AsyncEngineArgs(**self.vllm_engine_config)',
     '''        _server_only_keys = [
            "enable_auto_tool_choice", "tool_call_parser", "reasoning_parser",
            "served_model_name", "chat_template",
        ]
        self._server_args = {
            k: self.vllm_engine_config.pop(k)
            for k in _server_only_keys
            if k in self.vllm_engine_config
        }
        aync_engine_args = AsyncEngineArgs(**self.vllm_engine_config)'''),
    ('        self._aync_engine_args = AsyncEngineArgs(**self.vllm_engine_config)',
     '''        _server_only_keys = [
            "enable_auto_tool_choice", "tool_call_parser", "reasoning_parser",
            "served_model_name", "chat_template",
        ]
        self._server_args = {
            k: self.vllm_engine_config.pop(k)
            for k in _server_only_keys
            if k in self.vllm_engine_config
        }
        self._aync_engine_args = AsyncEngineArgs(**self.vllm_engine_config)'''),
]

patched = False
for old, new in PATTERNS:
    if old in content:
        content = content.replace(old, new)
        patched = True
        break

if not patched:
    print("WARNING: Could not patch model.py (may already be patched or different format)")
else:
    with open(BACKEND_PATH, "w") as f:
        f.write(content)
    print("Patched model.py successfully")

# Patch 2: Add BLOCK strategy to compressed_tensors_w8a16_fp8.py
FP8_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py"
with open(FP8_PATH, "r") as f:
    content = f.read()

OLD_STRATEGIES = "SUPPORTED_STRATEGIES = [QuantizationStrategy.CHANNEL, QuantizationStrategy.TENSOR]"
NEW_STRATEGIES = "SUPPORTED_STRATEGIES = [QuantizationStrategy.CHANNEL, QuantizationStrategy.TENSOR, QuantizationStrategy.BLOCK]"

if OLD_STRATEGIES in content:
    content = content.replace(OLD_STRATEGIES, NEW_STRATEGIES)
    with open(FP8_PATH, "w") as f:
        f.write(content)
    print("Patched compressed_tensors_w8a16_fp8.py successfully")
else:
    print("WARNING: Could not patch fp8 strategies (may already include BLOCK)")
