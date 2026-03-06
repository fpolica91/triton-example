#!/usr/bin/env python3
"""
Patch Triton's vLLM backend model.py to support tool-calling args in model.json.

These args are server-level in vLLM, not engine-level, so they need to be
popped from the config before passing to AsyncEngineArgs.
"""

BACKEND_PATH = "/opt/tritonserver/backends/vllm/model.py"

# The original line that creates AsyncEngineArgs
OLD = '        self._aync_engine_args = AsyncEngineArgs(**self.vllm_engine_config)'

# Pop server-level args before creating AsyncEngineArgs, store them for later use
NEW = '''        # [PATCH] Extract server-level args not supported by AsyncEngineArgs
        _server_only_keys = [
            "enable_auto_tool_choice", "tool_call_parser", "reasoning_parser",
            "served_model_name", "chat_template",
        ]
        self._server_args = {
            k: self.vllm_engine_config.pop(k)
            for k in _server_only_keys
            if k in self.vllm_engine_config
        }
        self._aync_engine_args = AsyncEngineArgs(**self.vllm_engine_config)'''

with open(BACKEND_PATH, "r") as f:
    content = f.read()

if OLD not in content:
    print("ERROR: Could not find target line to patch")
    exit(1)

content = content.replace(OLD, NEW)

with open(BACKEND_PATH, "w") as f:
    f.write(content)

print("Patched successfully")
