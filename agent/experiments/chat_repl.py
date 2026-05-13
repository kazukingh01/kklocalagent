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

# `input()` を readline 経由にして、↑↓ で履歴・←→ でカーソル移動・Ctrl-A/E
# などの行編集が効くようにする。import するだけで Python の `input()` が
# 自動的に readline 経由になる (副作用ベースの API)。本 REPL は agent
# (linux container) 内で `docker compose exec` 経由でしか動かさない前提
# なので readline は確実に存在する。
import readline  # noqa: F401

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
    # `got_content` / `saw_done`: agent から content も done も来ずに stream
    # が閉じた場合に「agent からの応答が空でした」と明示するためのフラグ。
    # これが無いと、REPL は次の `user> ` を `agent> ` と同じ行に貼り付けて
    # しまい、何が起きたか分かりにくくなる (実観測: tool エラー後に LLM が
    # 一文字も生成せず close するケース)。
    got_content = False
    saw_done = False
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
                    saw_done = True
                    break
                content = (msg.get("message") or {}).get("content", "")
                if content:
                    got_content = True
                    sys.stdout.write(content)
                    sys.stdout.flush()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        sys.stdout.write(f"\n[HTTP {e.code}] {body[:400]}")
    except Exception as e:  # noqa: BLE001
        sys.stdout.write(f"\n[client error] {type(e).__name__}: {e}")

    # `agent> ` プロンプトを必ず改行で閉じる。content が無くても (= LLM が
    # 何も生成せずに終わった or 接続が途中で切れた) 次のユーザプロンプトが
    # 同じ行に出てしまう問題を防ぐ。
    if not got_content:
        sys.stdout.write("(no response)" if saw_done else "(stream closed)")
    sys.stdout.write("\n")
    sys.stdout.flush()


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
