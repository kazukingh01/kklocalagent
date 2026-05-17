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
import audioop  # Python 3.13 で deprecated だが、stdlib のままで動く間は依存ゼロで便利。
import difflib
import json
import logging
import os
import sys
import time
import wave
from pathlib import Path

import aiohttp
from langchain_core.tools import tool

from sandbox import SandboxError, ensure_command_allowed, ensure_path_in_root

log = logging.getLogger("agent.tools")

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

# play_audio_file: audio-io の /spk WebSocket URL (track 番号も含む)。
# 例: ws://host.docker.internal:7010/spk?track=1
# 空なら tool の description が「現在使用不可」と LLM に伝え、呼ばれても
# 即 [error] で返す。
_AUDIO_IO_SPK_URL: str = os.environ.get("AGENT_AUDIO_IO_SPK_URL", "")
# audio-io の wire format。WAV のサンプルレート / チャンネル数がこれと
# 一致しない場合は audioop で吸収して送り出す (44.1kHz stereo の WAV でも
# 16kHz mono に変換して再生できる)。defaults はプロジェクト標準。
_AUDIO_IO_WIRE_RATE: int = int(os.environ.get("AGENT_AUDIO_IO_WIRE_RATE", "16000"))
_AUDIO_IO_WIRE_CHANNELS: int = int(os.environ.get("AGENT_AUDIO_IO_WIRE_CHANNELS", "1"))
# サンプルレート / チャンネル不一致は audioop で吸収する (stereo→mono は
# `tomono`、レート変換は `ratecv`)。LLM が「48kHz の WAV ですが？」と
# 言い訳せず再生できるようにするための保険。
_AUDIO_SAMPLE_WIDTH = 2  # s16le 固定 (= 16-bit PCM)
# 単一再生の壁時計上限 (秒)。1 時間の WAV をうっかり渡されると tool が
# その間 block するので DoS 対策。
_AUDIO_PLAY_MAX_S: float = float(os.environ.get("AGENT_AUDIO_PLAY_MAX_S", "300"))

# 出力サイズ上限 (issue #19 設計の「個別 tool ごと truncate」決定に基づく)。
_SHELL_STDOUT_MAX = 3000
_FILE_READ_MAX = 51200  # 50 KB
_WEB_SEARCH_MAX_RESULTS = 5      # top-N 件
_WEB_SEARCH_SNIPPET_MAX = 120    # 各件 snippet の文字数上限
_WEB_SEARCH_TOTAL_MAX = 1500     # 全体上限 (≈ context 圧迫を避ける)
# audio-io へ送る 1 batch (= 1 WS message) の長さ。20ms は audio-io の
# 既定の frame_ms と合わせている。これより細かいと WS のオーバーヘッドが
# 増え、これより粗いと barge-in (途中停止) のレスポンスが鈍る。
_AUDIO_PLAY_FRAME_MS = 20
# {"type":"eos"} を投げた後 {"type":"drained"} を待つ最大時間。長尾の
# 最終文 (~30s) でも drain 完了するよう余裕をもたせる。
_AUDIO_DRAIN_TIMEOUT_S = 30.0

# subprocess の壁時計タイムアウト。voice agent としては即応が前提なので
# 短めに切る。長く回したい操作は run_shell のスコープ外。
_SHELL_TIMEOUT_S = 5.0
# DDG 検索のタイムアウト。voice agent は即応性が大事なので短め。rate limit
# でレスポンスが詰まったら諦めて LLM に「失敗」を渡す方が UX が良い。
_WEB_SEARCH_TIMEOUT_S = 5.0


# --- tool 実装 ----------------------------------------------------------
# 「現在時刻」系の独立 tool は持たない — `date` を run_shell 経由で呼べば
# 十分で、agent コンテナの TZ を Asia/Tokyo に固定してあるので JST が返る
# (compose.yaml の agent.environment.TZ 参照)。

async def _run_shell_impl(command: str) -> str:
    """run_shell の本体。例外は raise する — `run_shell` 側で catch して
    文字列化する。テスト容易性のため `@tool` ラッパとは分離している。

    cwd は `_FILE_ROOT` (set されていれば) に揃える。`read_file` が
    `_FILE_ROOT` 相対のパスを取るのと意味論を合わせ、LLM が `ls` / `cat`
    を引数なし or 相対パスで呼んだときに同じ場所を見るようにする
    (未設定なら inherited = `/app`、agent 本体のコードが見える点は現状維持)。
    """
    argv = ensure_command_allowed(command, _SHELL_ALLOWLIST)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=_FILE_ROOT or None,
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
        avail = f"Only these command names are allowed: **{listed}**"
    else:
        avail = (
            "No commands are currently allowed "
            "(AGENT_SHELL_ALLOWLIST is empty)"
        )
    cwd_hint = (
        f"Current working directory is `{_FILE_ROOT}` "
        "(same as the read_file root). Relative paths target files "
        "inside that directory (e.g. `ls`, `cat memo.txt`). "
        "Pass an absolute path to look elsewhere (e.g. `ls /etc`)."
        if _FILE_ROOT
        else "Current working directory is the agent container's "
        "`/app`. Pass absolute paths to reach files outside it "
        "(e.g. `ls /etc`)."
    )
    return (
        "Run one Linux shell command and return its stdout.\n\n"
        f"{avail}. Anything else returns a [denied] error.\n"
        "Pipes (`|`), redirects (`>`), and command chaining "
        "(`;`, `&&`) are structurally impossible "
        "(`shell=True` is not used).\n\n"
        f"{cwd_hint}\n\n"
        "`command` takes the executable name and its arguments "
        "separated by spaces (e.g. `\"date\"`, `\"ls\"`). Each call "
        "is capped at 5 seconds and stdout is truncated past 3KB."
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
    """Search the web and return up to 5 result summaries.

    Call this for current news, product info, weather, or anything
    you can't answer from your own knowledge. `query` is a natural-
    language search string (e.g. `"weather in Tokyo today"`,
    `"Rust async runtime comparison"`).

    Results come back as `N. title / snippet / URL` per entry, up to
    5 entries, with the whole payload capped at ~1500 characters.
    Fetching full page bodies or following links is out of scope —
    distil the snippets into a short natural-language answer.
    """
    return await _safe_invoke("web_search", _web_search_impl(query))


async def _play_audio_file_impl(path: str) -> str:
    """play_audio_file 本体。audio-io の /spk (?track=N) WS に PCM を realtime
    ペーシングで流し、`{"type":"eos"}` で drain handshake して終わる。

    全例外は `_safe_invoke` で string 化される。barge-in は orchestrator が
    /api/chat の HTTP を切ることで `asyncio.CancelledError` を伝播させて
    アボートする — その時点で `async with` の context manager が WS を閉じ、
    audio-io 側もリングが drain → close を見て後始末する。
    """
    if not _AUDIO_IO_SPK_URL:
        raise RuntimeError(
            "AGENT_AUDIO_IO_SPK_URL is not set — audio playback is disabled"
        )
    # 相対パスを許容する: read_file と意味論を合わせて _FILE_ROOT 配下として
    # 解釈し、root 外脱出は弾く。LLM が `ls` で見た相対パスをそのまま渡して
    # きても再生できるようにする (実運用での誤り頻度が高かったため)。
    p = Path(path)
    if not p.is_absolute():
        if not _FILE_ROOT:
            raise ValueError(
                f"path must be absolute, got: {path!r} "
                "(AGENT_FILE_ROOT not set so relative paths cannot be resolved)"
            )
        p = ensure_path_in_root(path, _FILE_ROOT)
    if not p.is_file():
        # 似た名前のファイルを同じ親ディレクトリから 5 件まで返す。
        # LLM が「該当ファイルが無い → このリストの中から選び直す or
        # 諦めて user に伝える」と判断できるようにするための補助情報。
        # 親ディレクトリ自体が無いケースも想定して try/except で包む。
        hint = ""
        try:
            parent = p.parent
            if parent.is_dir():
                names = [c.name for c in parent.iterdir() if c.is_file()]
                close = difflib.get_close_matches(p.name, names, n=5, cutoff=0.4)
                if close:
                    hint = f" (did you mean: {', '.join(close)})"
                elif names:
                    hint = f" (files in {parent}: {', '.join(sorted(names)[:5])})"
        except OSError:
            pass
        raise FileNotFoundError(f"not a file: {p}{hint}")

    # WAV パース + リサンプル / リミックスは sync I/O + CPU 仕事なので
    # まとめて thread にオフロード。圧縮 / sample width は raise する
    # (audioop は s16le 固定で扱う)。sample rate / channels の不一致は
    # ここで吸収して wire format に揃える。
    def _read_wav() -> tuple[bytes, float, int, int]:
        with wave.open(str(p), "rb") as wav:
            comp = wav.getcomptype()
            if comp != "NONE":
                raise ValueError(
                    f"compressed WAV ({comp}) is not supported; "
                    "give uncompressed PCM (s16le)"
                )
            sw = wav.getsampwidth()
            if sw != _AUDIO_SAMPLE_WIDTH:
                raise ValueError(
                    f"unsupported sample width {sw * 8} bits; "
                    "only 16-bit PCM (s16le) is supported"
                )
            fr = wav.getframerate()
            ch = wav.getnchannels()
            n = wav.getnframes()
            duration_s = n / fr
            if duration_s > _AUDIO_PLAY_MAX_S:
                raise ValueError(
                    f"duration {duration_s:.1f}s exceeds limit "
                    f"{_AUDIO_PLAY_MAX_S:.0f}s"
                )
            pcm = wav.readframes(n)

        # チャンネル変換 (audioop は 1↔2 のみネイティブ対応)。
        # rate 変換より先にやると後段の処理データ量が減るので少しだけ速い。
        if ch != _AUDIO_IO_WIRE_CHANNELS:
            if ch == 2 and _AUDIO_IO_WIRE_CHANNELS == 1:
                pcm = audioop.tomono(pcm, _AUDIO_SAMPLE_WIDTH, 0.5, 0.5)
            elif ch == 1 and _AUDIO_IO_WIRE_CHANNELS == 2:
                pcm = audioop.tostereo(pcm, _AUDIO_SAMPLE_WIDTH, 1.0, 1.0)
            else:
                raise ValueError(
                    f"cannot convert {ch}-channel WAV to "
                    f"{_AUDIO_IO_WIRE_CHANNELS}-channel wire format "
                    "(only 1↔2 channel conversions are supported)"
                )

        # サンプルレート変換。`state=None` は新規変換 (連続呼び出し時の
        # 補間状態を引き継がない) を意味する。今回は単発なので None で OK。
        if fr != _AUDIO_IO_WIRE_RATE:
            pcm, _ = audioop.ratecv(
                pcm,
                _AUDIO_SAMPLE_WIDTH,
                _AUDIO_IO_WIRE_CHANNELS,
                fr,
                _AUDIO_IO_WIRE_RATE,
                None,
            )

        return pcm, duration_s, fr, ch

    pcm, duration_s, src_rate, src_ch = await asyncio.to_thread(_read_wav)
    bytes_per_sample = _AUDIO_SAMPLE_WIDTH * _AUDIO_IO_WIRE_CHANNELS
    samples_per_frame = _AUDIO_IO_WIRE_RATE * _AUDIO_PLAY_FRAME_MS // 1000
    bytes_per_frame = samples_per_frame * bytes_per_sample

    if (src_rate, src_ch) != (_AUDIO_IO_WIRE_RATE, _AUDIO_IO_WIRE_CHANNELS):
        log.info(
            "play_audio_file: %s (%.1fs) converted %dHz/%dch → %dHz/%dch → %s",
            p.name, duration_s, src_rate, src_ch,
            _AUDIO_IO_WIRE_RATE, _AUDIO_IO_WIRE_CHANNELS,
            _AUDIO_IO_SPK_URL,
        )
    else:
        log.info(
            "play_audio_file: %s (%.1fs, %dHz, %dch) → %s",
            p.name, duration_s, _AUDIO_IO_WIRE_RATE,
            _AUDIO_IO_WIRE_CHANNELS, _AUDIO_IO_SPK_URL,
        )

    # ws_connect の sock_connect で接続フェーズを短く (audio-io が居なければ
    # ここで即落ちて [error] にする)。全体 timeout は意図的に None — 長尺
    # 再生中ずっと send/recv するので。
    client_timeout = aiohttp.ClientTimeout(total=None, sock_connect=10.0)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.ws_connect(_AUDIO_IO_SPK_URL) as ws:
            start = time.monotonic()
            frame_idx = 0
            offset = 0
            while offset < len(pcm):
                chunk = pcm[offset : offset + bytes_per_frame]
                offset += len(chunk)
                # audio-io は奇数バイトの WS frame を拒否する (s16le なので
                # 通常は偶数だが、末尾の半端な chunk が来た場合だけパディング)。
                if len(chunk) % 2 != 0:
                    chunk = chunk + b"\x00"
                await ws.send_bytes(chunk)
                frame_idx += 1
                # 実時間ペーシング。各 frame は frame_idx * FRAME_MS のタイミングで
                # 送出する。audio-io 側の ring (deafult 10s) を溢れさせない目的。
                target = start + frame_idx * _AUDIO_PLAY_FRAME_MS / 1000.0
                sleep_for = target - time.monotonic()
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
            # Drain handshake: eos を送ってから "drained" が戻るまで待つ。
            # これで audio-io の cpal リングが本当に空になったことが保証される。
            await ws.send_str(json.dumps({"type": "eos"}))
            try:
                msg = await asyncio.wait_for(
                    ws.receive(), timeout=_AUDIO_DRAIN_TIMEOUT_S
                )
                if msg.type != aiohttp.WSMsgType.TEXT:
                    log.warning(
                        "play_audio_file: drain reply was %s, not TEXT",
                        msg.type,
                    )
            except asyncio.TimeoutError:
                log.warning(
                    "play_audio_file: drain handshake timed out after %.0fs",
                    _AUDIO_DRAIN_TIMEOUT_S,
                )
    return f"played {duration_s:.1f}s from {p.name}"


def _build_play_audio_description() -> str:
    """Tool description を起動時の env (`_AUDIO_IO_SPK_URL` の有無) に応じて
    動的に生成する。URL 未設定なら「無効」を明示し、LLM が呼ばずに状況を
    user に伝えるよう誘導する。"""
    if not _AUDIO_IO_SPK_URL:
        return (
            "Play a WAV audio file via the speaker.\n\n"
            "**CURRENTLY UNAVAILABLE** — the AGENT_AUDIO_IO_SPK_URL env "
            "is not set, so this feature is disabled in this environment. "
            "Do NOT call this tool; instead, tell the user in one sentence "
            "that audio playback is not configured here so they know the "
            "limit is on the setup, not on their request."
        )
    path_hint = (
        f"`path` is a WAV file path. Either absolute (e.g. `/workspace/share/foo.wav`) "
        f"or relative to the share root `{_FILE_ROOT}` (e.g. `share/foo.wav`)."
        if _FILE_ROOT
        else "`path` is an **absolute** filesystem path to a WAV file."
    )
    return (
        "Play a WAV audio file via the speaker.\n\n"
        f"{path_hint} The file must be uncompressed 16-bit PCM (s16le). "
        "Sample rate and channel count are auto-converted to "
        f"{_AUDIO_IO_WIRE_RATE}Hz / {_AUDIO_IO_WIRE_CHANNELS} channel(s) "
        "(audio-io's wire format) if they differ, so 44.1kHz stereo and "
        "48kHz mono and similar are all accepted. Compressed WAVs and "
        ">2-channel surround formats are rejected. If the file isn't "
        "found, the error message lists similar filenames in the same "
        "directory — pick one and retry rather than asking the user.\n\n"
        "Call this when the user asks to play a specific audio file — "
        "for example a pre-rendered news summary or notification produced "
        "by another agent. The tool blocks until playback completes; if "
        "the user interrupts via wake-word, playback aborts cleanly. "
        f"Duration is capped at {_AUDIO_PLAY_MAX_S:.0f} seconds."
    )


@tool(description=_build_play_audio_description())
async def play_audio_file(path: str) -> str:
    # description は @tool(description=...) で動的注入。詳細は
    # `_build_play_audio_description` の docstring を参照。
    return await _safe_invoke("play_audio_file", _play_audio_file_impl(path))


@tool
async def read_file(path: str) -> str:
    """Read a file and return its contents verbatim.

    `path` is interpreted relative to `AGENT_FILE_ROOT` (the shared
    directory), e.g. `"memo.txt"`, `"docs/recipe.md"`. Absolute paths
    and paths containing `..` are rejected.

    Use this when the user asks to read a memo or a specific file —
    fetch the contents here, then summarise or read it back to them.
    Files larger than 50 KB are truncated at the tail.
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

    issue #19 の段階的な追加に伴い後段の PR で別 tool が積まれる可能性あり。
    """
    return [run_shell, read_file, web_search, play_audio_file]


# tool name → ack phrase。None なら ack 挿入なし。
#
# 即応 tool (shell / file) は None でよく、遅い tool (web 検索 / 音声再生)
# だけ ack を入れて voice agent の無音を埋める。文言は環境変数で上書き可能。
#
# play_audio_file は AGENT_AUDIO_IO_SPK_URL が未設定なら呼び出し即 [error]
# になるので、その場合は ack も抑止しておく (「再生するね」と言った直後に
# 失敗を返すと UX が混乱するため)。
TOOL_ACK_PHRASES: dict[str, str | None] = {
    "run_shell": None,
    "read_file": None,
    "web_search": os.environ.get(
        "AGENT_TOOL_ACK_WEB_SEARCH", "ちょっと検索してみるね。"
    ),
    "play_audio_file": (
        os.environ.get("AGENT_TOOL_ACK_PLAY_AUDIO", "音声を再生するね。")
        if _AUDIO_IO_SPK_URL
        else None
    ),
}


# --- CLI (`python tools.py <tool> <args>`) -------------------------------
#
# LLM 抜きで個別 tool を手叩きするための薄い entry point。
# LangGraph ToolNode は `.ainvoke({"path": "..."})` を呼ぶだけなので、
# ここでも同じ呼び出しを再現する。これで「LLM が呼ぶときと同じコードパス」
# (= `_safe_invoke` で例外を string 化、ack は CLI には出ない) で挙動確認できる。
#
# Usage:
#   python tools.py                                 # list tools
#   python tools.py <name>                          # show args & description
#   python tools.py <name> <value>                  # single-arg tools
#   python tools.py <name> key=value [key=value]    # multi-arg / named
#
# 例:
#   python tools.py play_audio_file /workspace/share/foo.wav
#   python tools.py run_shell command="ls -la"
def _cli_list(tools: list) -> None:
    print("available tools:")
    for t in tools:
        desc = (t.description or "").splitlines()[0] if t.description else ""
        print(f"  {t.name:20s} {desc}")


def _cli_show(t) -> None:
    print(f"{t.name}\n")
    print(t.description or "(no description)")
    print()
    print("args:")
    for k, v in (t.args or {}).items():
        print(f"  {k}: {v}")


def _cli_parse(values: list[str], schema_keys: list[str]) -> dict:
    """argv 後半を {param: value} に直す。

    - 全要素に `=` を含むなら全部 key=value 形式とみなす
    - そうでなければ positional 扱いで schema 順に zip
    - 引数 1 個 + schema 1 個 の最頻ケースは positional でそのまま渡る
    """
    if not values:
        return {}
    if all("=" in v for v in values):
        out: dict = {}
        for v in values:
            k, _, val = v.partition("=")
            out[k] = val
        return out
    if len(values) != len(schema_keys):
        raise SystemExit(
            f"expected {len(schema_keys)} positional arg(s) "
            f"({', '.join(schema_keys) or '-'}), got {len(values)}; "
            "use key=value form for multi-arg tools"
        )
    return dict(zip(schema_keys, values))


async def _cli_main(argv: list[str]) -> int:
    tools = all_tools()
    by_name = {t.name: t for t in tools}

    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python tools.py <tool_name> [<args...>]\n")
        _cli_list(tools)
        return 0

    name = argv[0]
    if name not in by_name:
        print(f"unknown tool: {name}", file=sys.stderr)
        print(f"available: {', '.join(by_name)}", file=sys.stderr)
        return 1

    tool_obj = by_name[name]
    if len(argv) == 1:
        _cli_show(tool_obj)
        return 0

    args = _cli_parse(argv[1:], list((tool_obj.args or {}).keys()))
    result = await tool_obj.ainvoke(args)
    print(result)
    return 0


if __name__ == "__main__":
    # CLI 専用に logging を stderr へ INFO で出す。production の app.py は
    # 自前で logging 設定するので、こちらの basicConfig は __main__ 経路だけ。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    sys.exit(asyncio.run(_cli_main(sys.argv[1:])))
