#!/usr/bin/env python3
"""Gemma 4 + Ollama の tool-calling サポートを 3 ケースで確認する最小プローブ。

issue #19 (agent 拡充) で LangGraph の `create_react_agent` 路線が成立するかは
モデルが Ollama の `tools=[...]` を理解して `message.tool_calls` を返せるかに
依存する。本スクリプトは agent コンテナや langchain を経由せず、Ollama の
/api/chat を直接叩いて以下を判定する:

  1. 引数なし tool を発火できるか      (`get_current_date`)
  2. 引数を抽出して tool を発火できるか (`add(a, b)`)
  3. tool 不要時に発火しないか         ("こんにちは")

実行 (ホスト側):
    python3 agent/experiments/tool_calling_probe.py
環境変数で上書き可:
    OLLAMA_URL=http://localhost:7050  (デフォルト)
    OLLAMA_MODEL=gemma4:e4b           (デフォルト)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:7050").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "現在の日付を YYYY-MM-DD 形式で返す。日付や曜日を聞かれたときに呼ぶ。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "2 つの整数を足し算して結果を返す。算数の計算を聞かれたときに呼ぶ。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "1 つ目の整数"},
                    "b": {"type": "integer", "description": "2 つ目の整数"},
                },
                "required": ["a", "b"],
            },
        },
    },
]

PROBES = [
    ("引数なし tool 発火",   "今日の日付を教えて。",          ["get_current_date"]),
    ("引数あり tool 発火",   "3 と 5 を足したらいくつ？",     ["add"]),
    ("tool 不要 (false-positive 検査)", "こんにちは、元気？", []),
]


def chat(user_text: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": user_text}],
        "tools": TOOLS,
        "stream": False,
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def summarize(resp: dict) -> tuple[list[str], str]:
    """Return (tool_call_names, assistant_text)."""
    msg = resp.get("message") or {}
    tool_calls = msg.get("tool_calls") or []
    names = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name", "?")
        args = fn.get("arguments")
        names.append(f"{name}({args})" if args else name)
    return names, (msg.get("content") or "").strip()


def main() -> int:
    print(f"Probing {OLLAMA_URL} model={MODEL}\n")
    all_ok = True
    for label, prompt, expected_tools in PROBES:
        print(f"--- {label} ---")
        print(f"user: {prompt}")
        try:
            resp = chat(prompt)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            print(f"  HTTP {e.code}: {body[:400]}")
            all_ok = False
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {type(e).__name__}: {e}")
            all_ok = False
            continue

        names, text = summarize(resp)
        print(f"  tool_calls: {names if names else '(none)'}")
        if text:
            print(f"  content:    {text[:200]}")

        if expected_tools:
            ok = any(any(n.startswith(t) for n in names) for t in expected_tools)
            verdict = "OK" if ok else "FAIL (no tool_call)"
        else:
            ok = not names
            verdict = "OK" if ok else "FAIL (unexpected tool_call)"
        print(f"  => {verdict}\n")
        if not ok:
            all_ok = False

    print("ALL OK" if all_ok else "SOME PROBES FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
