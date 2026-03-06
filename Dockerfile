FROM nvcr.io/nvidia/tritonserver:26.02-vllm-python-py3
RUN pip install --upgrade vllm==0.16.0
COPY patch_model.py /tmp/patch_model.py
RUN python3 /tmp/patch_model.py && rm /tmp/patch_model.py
