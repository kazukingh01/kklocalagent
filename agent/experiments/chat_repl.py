#!/usr/bin/env python3
"""agent /api/chat を叩く最小 CLI REPL (issue #19 step 2 のテストハーネス).

本番の voice pipeline (orchestrator → tts-streamer → audio-io) を通さずに
agent 単体を CLI で連続会話できるようにするためのスクリプト。tool 呼び出しの
挙動を目視確認したり、system prompt の効きを試したりするのに使う。

agent 側の wire format (ollama-compat ndjson stream) と接続経路は orchestrator
が叩くものと同一なので、ここで動けば voice 経由でも動く (はず)。

依存は stdlib のみ — `python3 chat_repl.py` で動く。

使い方:
    python3 agent/experiments/chat_repl.py
    python3 agent/experiments/chat_repl.py --url http://localhost:7080

セッションの conversation memory は agent 側 SqliteSaver が持っている。
連続会話の中で「さっきの話」を引き継げるか確認するときはそのまま続けて
入力する。Ctrl-D / Ctrl-C で終了。

`--session` (= GET /session) を確認することで現在の thread_id と idle 時間が
取れる。tools 有効/無効 (= AGENT_TOOLS_ENABLED) もここに出るので、間違った
agent に向けていないかの一次切り分けに使える。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# 1 ターンの応答に最大これだけ待つ。tool 呼び出しが連鎖すると数秒かかるので
# 余裕を持って 120s。voice agent 本番は recursion_limit + 各 tool timeout で
# 物理的に上限が決まる。
RESPONSE_TIMEOUT_S = 120


def session_info(url: str) -> dict:
    """GET /session — 起動直後の sanity check に使う。"""
    with urllib.request.urlopen(f"{url}/session", timeout=5) as resp:
        return json.loads(resp.read())


def chat_once(url: str, message: str) -> None:
    """1 ターン分の往復。stdout に token を逐次 flush しながら書き出す。

    レスポンスは ollama-compat ndjson (`{"message":{"content":...},"done":false}\n` ×
    N + `{"done":true}\n`)。urllib のレスポンスオブジェクトは行イテレータと
    して使えるので別途 buffering ロジックは要らない。
    """
    body = json.dumps(
        {
            "model": "agent",  # agent 側は body.model を見ない
            "messages": [{"role": "user", "content": message}],
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=RESPONSE_TIMEOUT_S) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # 想定外の行はそのまま見せる (デバッグ用)。
                    sys.stdout.write(f"\n[non-json] {line!r}\n")
                    continue
                if msg.get("done"):
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return
                content = (msg.get("message") or {}).get("content", "")
                if content:
                    sys.stdout.write(content)
                    sys.stdout.flush()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        sys.stdout.write(f"\n[HTTP {e.code}] {body[:400]}\n")
    except Exception as e:  # noqa: BLE001
        sys.stdout.write(f"\n[client error] {type(e).__name__}: {e}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url",
        default="http://localhost:7080",
        help="agent base URL (default: http://localhost:7080)",
    )
    args = parser.parse_args()

    # 起動直後のヘルス確認 — 接続先が間違ってると入力受けてから初めて気付くのを
    # 防ぐ。tools 有効/無効も表示してテストの一次切り分けに使う。
    try:
        info = session_info(args.url)
    except Exception as e:  # noqa: BLE001
        print(f"failed to reach {args.url}: {e}", file=sys.stderr)
        return 1
    print(f"agent: {args.url}")
    print(
        f"session={info.get('session_id', '?')[:8]}  "
        f"tools_enabled={info.get('tools_enabled', '?')}  "
        f"idle={info.get('idle_sec', '?')}s"
    )
    print("Ctrl-D or Ctrl-C to exit.")
    print()

    while True:
        try:
            text = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        sys.stdout.write("agent> ")
        sys.stdout.flush()
        chat_once(args.url, text)


if __name__ == "__main__":
    sys.exit(main())
