#!/usr/bin/env python3
"""Test tool calling capabilities of the MiniMax-M2.5 vLLM server."""

import sys
import json
import urllib.request

BASE_URL = "http://localhost:8002"
MODEL_NAME = "minimax-m2.5"


def api_call(messages, tools=None, tool_choice="auto"):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def test_single_tool_call():
    """Test that the model can call a single tool."""
    print("1. Single tool call")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name, e.g. 'San Francisco, CA'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location", "unit"]
            }
        }
    }]
    resp = api_call(
        messages=[{"role": "user", "content": "What's the weather in Tokyo? Use celsius."}],
        tools=tools,
    )
    msg = resp["choices"][0]["message"]
    if not msg.get("tool_calls"):
        print(f"   ✗ No tool calls in response. Content: {msg.get('content', '')[:100]}")
        return False
    tc = msg["tool_calls"][0]["function"]
    args = json.loads(tc["arguments"])
    ok = tc["name"] == "get_weather" and "location" in args
    print(f"   {'✓' if ok else '✗'} Called: {tc['name']}({json.dumps(args)})")
    return ok


def test_tool_response_roundtrip():
    """Test a full tool call -> tool response -> final answer roundtrip."""
    print("2. Tool response roundtrip")
    tools = [{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                },
                "required": ["expression"]
            }
        }
    }]
    # Step 1: get tool call
    resp1 = api_call(
        messages=[{"role": "user", "content": "What is 137 * 251?"}],
        tools=tools,
    )
    msg1 = resp1["choices"][0]["message"]
    if not msg1.get("tool_calls"):
        print(f"   ✗ No tool call generated. Content: {msg1.get('content', '')[:100]}")
        return False
    tc = msg1["tool_calls"][0]
    print(f"   ✓ Step 1: Model called {tc['function']['name']}({tc['function']['arguments']})")

    # Step 2: send tool result back
    resp2 = api_call(
        messages=[
            {"role": "user", "content": "What is 137 * 251?"},
            {"role": "assistant", "content": None, "tool_calls": [tc]},
            {"role": "tool", "tool_call_id": tc["id"], "content": "34387"},
        ],
        tools=tools,
    )
    msg2 = resp2["choices"][0]["message"]
    content = msg2.get("content", "")
    has_answer = "34387" in content or "34,387" in content
    print(f"   {'✓' if has_answer else '✗'} Step 2: Final answer contains 34387: {content[:120]}")
    return has_answer


def test_no_tool_when_unnecessary():
    """Test that the model does NOT call tools when the question doesn't need one."""
    print("3. No tool call when unnecessary")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }]
    resp = api_call(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        tools=tools,
    )
    msg = resp["choices"][0]["message"]
    no_tool = not msg.get("tool_calls")
    content = msg.get("content", "")
    has_paris = "paris" in content.lower()
    ok = no_tool and has_paris
    print(f"   {'✓' if ok else '✗'} No tool called: {no_tool}, Answer mentions Paris: {has_paris}")
    return ok


def main():
    print("MiniMax-M2.5 Tool Calling Tests")
    print("=" * 40)
    results = []

    for test_fn in [test_single_tool_call, test_tool_response_roundtrip, test_no_tool_when_unnecessary]:
        try:
            results.append(test_fn())
        except Exception as e:
            print(f"   ✗ Exception: {e}")
            results.append(False)

    print("=" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Result: {passed}/{total} tests passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
