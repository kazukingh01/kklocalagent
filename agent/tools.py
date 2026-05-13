"""LangChain tool 定義 + ack phrase マッピング (issue #19).

各 tool は `@tool` デコレータで LangChain Tool に変換され、agent_node に
`create_react_agent` 経由でバインドされる。LLM (Gemma 4) は description を
読んで「いつどの tool を呼ぶか」を確率的に判断する — コード側はルーティング
するだけで判断には関与しない (詳細は issue #19 設計コメント参照)。

ack phrase は tool 実行中の無音をカバーする目的の合成 chunk。`TOOL_ACK_PHRASES`
で tool 名 → 文字列にマップし、None なら ack 挿入をスキップする (即応 tool 向け)。

セキュリティ:
* `run_shell` は env `AGENT_SHELL_ALLOWLIST` (CSV) で許可コマンド名を完全一致
  照合。`asyncio.create_subprocess_exec` を使い `shell=True` は禁止 — パイプ
  やリダイレクトは構造的に不可能。
* `read_file` は env `AGENT_FILE_ROOT` 配下の **相対パス** のみ受け付ける。
  `Path.resolve()` で symlink 経由の jailbreak も弾く。
* 違反は `sandbox.SandboxError` (ValueError 派生) で raise → LangChain が
  ToolMessage として LLM に渡し、retry / 諦めを LLM 自身に判断させる
  (recursion_limit で打ち切り)。
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone, timedelta

from langchain_core.tools import tool

from sandbox import ensure_command_allowed, ensure_path_in_root

# 日本標準時。voice agent は日本語前提で動くので JST 固定が自然。海外向けに
# 切り出すときは TZ env を見るように変える。
JST = timezone(timedelta(hours=9))

# --- 環境変数 (起動時に 1 度読む) -------------------------------------------

# `AGENT_SHELL_ALLOWLIST="date,ip,uname,df,uptime,who"` 形式。空白は許容。
# set 化して O(1) で照合する。空文字列なら空 set = 一切のコマンドを拒否。
_SHELL_ALLOWLIST: set[str] = {
    s.strip()
    for s in os.environ.get("AGENT_SHELL_ALLOWLIST", "").split(",")
    if s.strip()
}
# read_file の root。compose で volume mount したパスをここに指す。空なら
# read_file 自体が「未設定」エラーで raise (= tool が事実上 disabled)。
_FILE_ROOT: str = os.environ.get("AGENT_FILE_ROOT", "")

# 出力サイズ上限 (issue #19 設計の「個別 tool ごと truncate」決定に基づく)。
_SHELL_STDOUT_MAX = 3000
_FILE_READ_MAX = 51200  # 50 KB

# subprocess の壁時計タイムアウト。voice agent としては即応が前提なので
# 短めに切る。長く回したい操作は run_shell のスコープ外。
_SHELL_TIMEOUT_S = 5.0


# --- tool 実装 ----------------------------------------------------------

@tool
def get_current_datetime() -> str:
    """現在の日時を JST で ISO 8601 形式 (例: 2026-05-13T15:30:00+09:00) で返す。

    日付・時刻・曜日・今が何月か・現在年などを聞かれたときに呼ぶ。
    """
    return datetime.now(JST).isoformat(timespec="seconds")


@tool
async def run_shell(command: str) -> str:
    """Linux のシェルコマンドを 1 つ実行して標準出力を返す。

    許可されたコマンド名のみ実行できる (例: date, uptime, df, uname, ip, who)。
    パイプ (`|`) やリダイレクト (`>`)、複数コマンドの連結 (`;`, `&&`) は使えない。
    日付・ネットワーク状況・ディスク使用量・カーネル情報などを調べるときに呼ぶ。

    引数 command にはコマンド本体と引数をスペース区切りで渡す
    (例: `"uname -a"`, `"df -h"`, `"ip addr"`)。
    """
    argv = ensure_command_allowed(command, _SHELL_ALLOWLIST)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=_SHELL_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        # SIGKILL してから wait() で zombie 化を防ぐ。barge-in による
        # CancelledError もここを通って同じ後始末を踏む。
        proc.kill()
        await proc.wait()
        raise TimeoutError(
            f"command timed out after {_SHELL_TIMEOUT_S:.1f}s: {argv[0]}"
        )
    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    # 結果文字列の組み立て: 終了コード != 0 なら明示、stderr があれば追記。
    # LLM が「失敗した、別コマンドで retry」と判断できるよう情報を残す。
    parts: list[str] = []
    if proc.returncode != 0:
        parts.append(f"[exit {proc.returncode}]")
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    result = "\n".join(parts) if parts else "(no output)"
    if len(result) > _SHELL_STDOUT_MAX:
        result = result[:_SHELL_STDOUT_MAX] + "\n[...truncated]"
    return result


@tool
async def read_file(path: str) -> str:
    """指定ファイルを読んで内容をそのまま返す。

    path は `AGENT_FILE_ROOT` (= 共有ディレクトリ) からの相対パスを渡す
    (例: `"memo.txt"`, `"docs/recipe.md"`)。絶対パスや `..` を含むパスは
    拒否される。

    voice agent のユースケース: ユーザに「メモを読んで」「あのファイルを
    読んで」と言われたとき、ここで内容を取得して読み上げに繋げる。
    内容は 50 KB を超える場合は末尾が省略される。
    """
    if not _FILE_ROOT:
        raise RuntimeError(
            "AGENT_FILE_ROOT is not configured — read_file is disabled"
        )
    resolved = ensure_path_in_root(path, _FILE_ROOT)
    if not resolved.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    # ファイル I/O は asyncio.to_thread でオフロードして event loop の
    # ブロックを避ける。50 KB なら一瞬だが、ネットワーク FS にマウント
    # された場合のレイテンシを吸収する保険。
    data = await asyncio.to_thread(resolved.read_bytes)
    truncated = len(data) > _FILE_READ_MAX
    text = data[:_FILE_READ_MAX].decode("utf-8", errors="replace")
    if truncated:
        text += "\n[...truncated]"
    return text


# --- tool 登録 ----------------------------------------------------------

def all_tools() -> list:
    """登録 tool の一覧。`create_react_agent` に渡す。

    issue #19 の段階的な追加に伴い後段の PR で `web_search` / `list_drive_files`
    / `read_drive_file` が積まれる予定。
    """
    return [get_current_datetime, run_shell, read_file]


# tool name → ack phrase。None なら ack 挿入なし。
#
# 即応 tool (date / shell / file) は None でよく、遅い tool (web 検索や Drive
# アクセス) だけ ack を入れて voice agent の無音を埋める。文言は環境変数で
# 上書き可能にしておくと運用しやすい — 各 tool 単位の ack 環境変数は対応する
# tool が wire された PR で追加する。
TOOL_ACK_PHRASES: dict[str, str | None] = {
    "get_current_datetime": None,
    "run_shell": None,
    "read_file": None,
}
