FROM nvcr.io/nvidia/tritonserver:26.02-vllm-python-py3

# Upgrade vLLM to nightly with minimax_m2 parser support
RUN pip install -U --pre 'triton-kernels @ git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels' vllm --extra-index-url https://wheels.vllm.ai/nightly

# Add minimax_m2 tool call parser to the OpenAI frontend
COPY minimax_m2_tool_call_parser.py /opt/tritonserver/python/openai/openai_frontend/engine/utils/tool_call_parsers/minimax_m2_tool_call_parser.py

# Register it
RUN sed -i 's/from .mistral_tool_call_parser import MistralToolParser/from .mistral_tool_call_parser import MistralToolParser\nfrom .minimax_m2_tool_call_parser import MinimaxM2ToolParser/' /opt/tritonserver/python/openai/openai_frontend/engine/utils/tool_call_parsers/__init__.py && \
    sed -i 's/"MistralToolParser",/"MistralToolParser",\n    "MinimaxM2ToolParser",/' /opt/tritonserver/python/openai/openai_frontend/engine/utils/tool_call_parsers/__init__.py

# Install frontend dependencies
RUN pip install packaging fastapi==0.121.2 httpx==0.27.2 openai==1.107.3 partial-json-parser scipy==1.16.3 "starlette>=0.49.1"
