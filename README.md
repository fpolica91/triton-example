# triton-example

Triton Inference Server + vLLM backend serving MiniMax-M2.5 (FP8+INT4-AWQ).

## Structure

```
├── run.sh                       # Docker run command for Triton
├── vllm_run.sh                  # Standalone vLLM server (no Triton)
└── model_repository/
    └── minimax-m2.5/
        ├── config.pbtxt         # Triton model config
        └── 1/
            └── model.json       # vLLM engine config
```

## Usage

### Triton + vLLM backend
```bash
bash run.sh
```

### Standalone vLLM
```bash
bash vllm_run.sh
```

Both serve on port 8002 (HTTP).
