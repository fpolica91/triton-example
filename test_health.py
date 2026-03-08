#!/usr/bin/env python3
"""Quick health check for the MiniMax-M2.5 vLLM server."""

import sys
import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8002"
MODEL_NAME = "minimax-m2.5"


def check(name, url, validate=None):
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            if validate:
                validate(data)
            print(f"  ✓ {name}")
            return True
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return False


def main():
    print("MiniMax-M2.5 vLLM Health Check")
    print("=" * 40)
    results = []

    # 1. Server health
    # Server health returns empty 200, not JSON
    try:
        req = urllib.request.Request(f"{BASE_URL}/health")
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = resp.status == 200
            print(f"  ✓ Server reachable (HTTP {resp.status})")
            results.append(True)
    except Exception as e:
        print(f"  ✗ Server reachable: {e}")
        results.append(False)

    # 2. Model loaded
    def validate_model(data):
        models = [m["id"] for m in data["data"]]
        assert MODEL_NAME in models, f"Expected '{MODEL_NAME}', got {models}"
    results.append(check("Model loaded", f"{BASE_URL}/v1/models", validate_model))

    # 3. Simple completion
    print("  … Testing chat completion (may take a few seconds)")
    try:
        payload = json.dumps({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Reply with exactly: HEALTH_OK"}],
            "max_tokens": 20,
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            tokens = data["usage"]["completion_tokens"]
            print(f"  ✓ Chat completion (got {tokens} tokens)")
            results.append(True)
    except Exception as e:
        print(f"  ✗ Chat completion: {e}")
        results.append(False)

    # 4. Metrics endpoint (for Prometheus)
    try:
        req = urllib.request.Request(f"{BASE_URL}/metrics")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            has_metrics = "vllm:" in body or "vllm_" in body
            if has_metrics:
                print("  ✓ Prometheus metrics endpoint")
            else:
                print("  ✗ Metrics endpoint returned no vllm metrics")
            results.append(has_metrics)
    except Exception as e:
        print(f"  ✗ Metrics endpoint: {e}")
        results.append(False)

    print("=" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Result: {passed}/{total} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
