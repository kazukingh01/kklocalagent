#!/usr/bin/env python3
"""agent /api/chat を叩く最小 CLI REPL (issue #19 step 2 のテストハーネス).

wire format と接続経路は orchestrator が叩くものと同一なので、
ここで動けば voice 経由でも動く (はず)。依存は stdlib のみ。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# import するだけで `input()` が readline 経由になり行編集が効く (副作用 API)。
import readline  # noqa: F401

RESPONSE_TIMEOUT_S = 120


def session_info(url: str) -> dict:
    with urllib.request.urlopen(f"{url}/session", timeout=5) as resp:
        return json.loads(resp.read())


def chat_once(url: str, message: str) -> None:
    """1 ターン分の往復。stdout に token を逐次 flush しながら書き出す。"""
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
    # 実観測: tool エラー後に LLM が一文字も生成せず close するケースがあり、
    # 明示しないと次の `user> ` が `agent> ` と同じ行に貼り付く。
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
