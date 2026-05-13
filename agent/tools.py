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

エラーハンドリング:
* tool 内部の例外は全て catch し `[denied] ...` / `[error] ...` 文字列として
  return する。raise すると LangChain が必ずしも捕捉しきれず、AIMessage の
  tool_calls が state に残ったまま対応 ToolMessage が積まれず以降のターンが
  INVALID_CHAT_HISTORY で死ぬ (検証で確認済)。string return にすれば LLM が
  「失敗した、別の方法を試す or 諦める」を ToolMessage を読んで判断できる。
* allowlist 違反のように LLM が retry しても無意味な失敗は `[denied]` プレ
  フィックスで区別し、LLM が「無理だから諦めて答える」方に倒しやすくする。
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta

from langchain_core.tools import tool

from sandbox import SandboxError, ensure_command_allowed, ensure_path_in_root

log = logging.getLogger("agent.tools")

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
_WEB_SEARCH_MAX_RESULTS = 5      # top-N 件
_WEB_SEARCH_SNIPPET_MAX = 120    # 各件 snippet の文字数上限
_WEB_SEARCH_TOTAL_MAX = 1500     # 全体上限 (≈ context 圧迫を避ける)

# subprocess の壁時計タイムアウト。voice agent としては即応が前提なので
# 短めに切る。長く回したい操作は run_shell のスコープ外。
_SHELL_TIMEOUT_S = 5.0
# DDG 検索のタイムアウト。voice agent は即応性が大事なので短め。rate limit
# でレスポンスが詰まったら諦めて LLM に「失敗」を渡す方が UX が良い。
_WEB_SEARCH_TIMEOUT_S = 5.0


# --- tool 実装 ----------------------------------------------------------

@tool
def get_current_datetime() -> str:
    """現在の日時を JST で ISO 8601 形式 (例: 2026-05-13T15:30:00+09:00) で返す。

    日付・時刻・曜日・今が何月か・現在年などを聞かれたときに呼ぶ。
    """
    return datetime.now(JST).isoformat(timespec="seconds")


async def _run_shell_impl(command: str) -> str:
    """run_shell の本体。例外は raise する — `run_shell` 側で catch して
    文字列化する。テスト容易性のため `@tool` ラッパとは分離している。"""
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


def _build_run_shell_description() -> str:
    """Tool description を `_SHELL_ALLOWLIST` から動的に組み立てる。

    `@tool` デコレータ呼び出し時に評価して LangChain の tool schema に
    入れる。これで LLM は **実際にこの環境で許可されているコマンド名** を
    把握でき、`.env` で allowlist を上書きしても LLM の手元の説明が古い
    まま (`ls` を呼ぶと「unknown command」と思い込む) という mismatch が
    起きない。

    env を変えたら agent コンテナを restart する必要があるのは現状と同じ
    (`_SHELL_ALLOWLIST` 自体が module load 時 1 回読みなので)。
    """
    if _SHELL_ALLOWLIST:
        listed = ", ".join(sorted(_SHELL_ALLOWLIST))
        avail = f"許可されているコマンド名は **{listed}** のみ"
    else:
        avail = "現在は何も許可されていない (AGENT_SHELL_ALLOWLIST が空)"
    return (
        "Linux のシェルコマンドを 1 つ実行して標準出力を返す。\n\n"
        f"{avail}。これ以外を渡すと [denied] エラーが返る。\n"
        "パイプ (`|`)、リダイレクト (`>`)、複数コマンドの連結 (`;`, `&&`) は"
        "構造的に使えない (`shell=True` 不使用)。\n\n"
        "command にはコマンド本体と引数をスペース区切りで渡す "
        "(例: `\"date\"`, `\"ls /workspace\"`)。実行は 5 秒で打ち切られ、"
        "stdout は 3KB を超えると末尾省略される。"
    )


@tool(description=_build_run_shell_description())
async def run_shell(command: str) -> str:
    # docstring ではなく @tool(description=...) で description を渡している。
    # 詳細は `_build_run_shell_description` の docstring 参照。
    return await _safe_invoke("run_shell", _run_shell_impl(command))


async def _read_file_impl(path: str) -> str:
    """read_file の本体。例外は raise する — `read_file` 側で catch する。"""
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


async def _web_search_impl(query: str) -> str:
    """web_search の本体。DDG の sync API を asyncio.to_thread で
    オフロードしつつ全体を wait_for でラップして 5s で打ち切る。"""
    query = (query or "").strip()
    if not query:
        raise ValueError("query is empty")

    # 遅延 import: ddgs が requirements に無い古い環境でも tools モジュール
    # 自体は import できるようにしておく (CI のすり抜け検知用)。
    # 旧パッケージ名 `duckduckgo_search` は 2025 年に `ddgs` にリネーム
    # されており、旧名で import すると検索が空 list を返す挙動になる。
    from ddgs import DDGS

    def _do_search() -> list[dict]:
        # DDGS はコンテキストマネージャ。`text()` は generator/list を返す。
        # max_results=N を渡せば最初の N 件で打ち切る。
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=_WEB_SEARCH_MAX_RESULTS))

    results = await asyncio.wait_for(
        asyncio.to_thread(_do_search),
        timeout=_WEB_SEARCH_TIMEOUT_S,
    )
    if not results:
        return "(no search results)"

    # 件ごとに title + snippet + href を整形。LLM はこれを context として
    # 受け取り「要点を口頭でまとめて」読み上げる想定。href は短縮しない —
    # LLM が引用 URL として言及できる方が voice agent として誠実。
    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip()
        snippet = (r.get("body") or "").strip()
        href = (r.get("href") or "").strip()
        if len(snippet) > _WEB_SEARCH_SNIPPET_MAX:
            snippet = snippet[:_WEB_SEARCH_SNIPPET_MAX] + "…"
        parts.append(f"{i}. {title}\n   {snippet}\n   {href}")
    text = "\n".join(parts)
    if len(text) > _WEB_SEARCH_TOTAL_MAX:
        text = text[:_WEB_SEARCH_TOTAL_MAX] + "\n[...truncated]"
    return text


@tool
async def web_search(query: str) -> str:
    """インターネットを検索して上位 5 件の概要を返す。

    最新ニュース・商品情報・調べごとなど、自分の知識だけでは答えられないことを
    聞かれたときに呼ぶ。query には自然文の検索クエリを渡す (例:
    `"今日の東京の天気"`, `"Rust async runtime 比較"`)。

    結果は各件 `N. タイトル / snippet / URL` 形式で最大 5 件、全体 1500 字を
    超えると末尾省略。リアルタイムページ取得や本文全文は別 tool 範囲外なので、
    snippet を見て要点を自然な言葉で要約して答える。
    """
    return await _safe_invoke("web_search", _web_search_impl(query))


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
    return await _safe_invoke("read_file", _read_file_impl(path))


async def _safe_invoke(name: str, coro) -> str:
    """共通エラーハンドラ。tool 本体 (coroutine) を await し、例外は文字列化
    して返す。

    raise すると LangChain v0.3 系の create_react_agent が AIMessage の
    tool_calls を state に積んだまま ToolMessage を積まないことがあり、以降の
    ターンで INVALID_CHAT_HISTORY (Found AIMessages with tool_calls that do
    not have a corresponding ToolMessage) で全 chat が死ぬ。本関数で string に
    変換しておけば、ReAct ループが必ず ToolMessage を state に積むので state
    が壊れない。

    `SandboxError` は LLM が retry しても通らない設定/権限系なので
    `[denied]` プレフィックスで明示し、LLM が「諦めて答える」方向に倒れやすく
    する (system prompt の「無理なら正直に伝えて」と組合せる)。それ以外は
    `[error]` で retry-worth な扱いに。
    """
    try:
        return await coro
    except SandboxError as e:
        log.info("tool %s denied: %s", name, e)
        return f"[denied] {e}"
    except Exception as e:  # noqa: BLE001
        log.warning("tool %s failed: %s: %s", name, type(e).__name__, e)
        return f"[error] {type(e).__name__}: {e}"


# --- tool 登録 ----------------------------------------------------------

def all_tools() -> list:
    """登録 tool の一覧。`create_react_agent` に渡す。

    issue #19 の段階的な追加に伴い後段の PR で `list_drive_files` /
    `read_drive_file` が積まれる予定。
    """
    return [get_current_datetime, run_shell, read_file, web_search]


# tool name → ack phrase。None なら ack 挿入なし。
#
# 即応 tool (date / shell / file) は None でよく、遅い tool (web 検索や Drive
# アクセス) だけ ack を入れて voice agent の無音を埋める。文言は環境変数で
# 上書き可能にしておくと運用しやすい — Drive 系は対応 tool 追加の PR で同様に
# AGENT_TOOL_ACK_DRIVE などを追加する想定。
TOOL_ACK_PHRASES: dict[str, str | None] = {
    "get_current_datetime": None,
    "run_shell": None,
    "read_file": None,
    "web_search": os.environ.get(
        "AGENT_TOOL_ACK_WEB_SEARCH", "ちょっと検索してみるね。"
    ),
}
