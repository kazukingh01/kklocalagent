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
import audioop  # 3.11 で deprecated、3.13 で stdlib から削除。3.11 pin の間は依存ゼロで使えるが、base image を上げる際は要置換。
import difflib
import json
import logging
import math
import os
import struct
import sys
import time
import wave
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

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
# 合計再生時間の上限は撤廃。ファイルは 1 本ずつ逐次デコードしてストリーミングする
# (`_stream_files_to_spk`) のでメモリは常に ~1 本ぶんに収まり、長すぎる場合は
# stop_audio / wake-word で止められる (旧 DoS 上限の代替)。


def _derive_stop_url(spk_url: str) -> str:
    """play 用の /spk WS URL から、停止用の /spk/stop HTTP URL を導出する。

    `ws://host:7010/spk?track=1` → `http://host:7010/spk/stop?track=1`。
    scheme は ws→http / wss→https、path 末尾の `/spk` に `/stop` を足し、
    `?track=N` クエリはそのまま引き継ぐ (= play と同じ track を狙い撃ちする)。
    設定を 1 本化するため専用 env は持たず、ここで変換する。

    `?track=N` は必須。未指定だと派生する /spk/stop も track なし = audio-io
    側で全 track flush (TTS の track 0 まで巻き込む) になってしまうので、起動
    時にここで明示的に弾く。N は非負整数 (audio-io の track id は 0 始まり)。
    """
    if not spk_url:
        return ""
    parts = urlsplit(spk_url)
    track_vals = parse_qs(parts.query).get("track")
    if not track_vals:
        raise ValueError(
            f"AGENT_AUDIO_IO_SPK_URL must include a ?track=N query "
            f"(N >= 0); got {spk_url!r}"
        )
    try:
        track = int(track_vals[0])
    except ValueError as e:
        raise ValueError(
            f"?track= must be an integer, got {track_vals[0]!r} in {spk_url!r}"
        ) from e
    if track < 0:
        raise ValueError(f"?track= must be >= 0, got {track} in {spk_url!r}")
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    path = parts.path.rstrip("/")
    # 通常 path は `/spk`。それ以外でも `/stop` を足して壊さないようにする。
    path = (path[: -len("/spk")] + "/spk/stop") if path.endswith("/spk") else path + "/stop"
    return urlunsplit((scheme, parts.netloc, path, parts.query, parts.fragment))


# stop_audio: play_audio_file が使う track を flush する HTTP エンドポイント。
# 空なら tool は「使用不可」を LLM に伝え、呼ばれても即 [error]。
_AUDIO_IO_STOP_URL: str = _derive_stop_url(_AUDIO_IO_SPK_URL)
# /spk/stop は即応 (flush signal を投げるだけ) なので短め。
_AUDIO_STOP_TIMEOUT_S = 5.0

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
        avail = f"Allowed commands: {listed} — nothing else."
    else:
        avail = "No commands are currently allowed (the allowlist is empty)."
    cwd = f"`{_FILE_ROOT}` (the shared directory)" if _FILE_ROOT else "`/app`"
    return (
        "Run one Linux shell command and return its output. "
        "Use it for the current date/time (`date`) or to list files (`ls`).\n"
        f"{avail} No pipes, redirects, or `;`/`&&` chaining.\n"
        f"`command` = executable + arguments (e.g. \"date\", \"ls news\"). "
        f"Working directory: {cwd}."
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
    """Search the web and return up to 5 results (title / snippet / URL).

    Call this for news, weather, prices — anything current you can't
    know yourself. `query` is a natural-language search string (e.g.
    "東京 今日の天気"). You can't fetch full pages; answer from the
    snippets in your own words.
    """
    return await _safe_invoke("web_search", _web_search_impl(query))


def _read_and_convert_wav(p: Path) -> tuple[bytes, float, int, int]:
    """単一 WAV を読んで wire format (rate/channels) に揃えた PCM を返す。

    返り値は `(pcm, duration_s, src_rate, src_ch)`。圧縮 / 非 s16le は raise。
    sync I/O + CPU 仕事なので呼び出し側で `asyncio.to_thread` に乗せる
    (`_stream_files_to_spk` が 1 本ずつ呼んで逐次ストリーミングする)。
    """
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
    # 補間状態を引き継がない) を意味する。各ファイル独立に変換するので None。
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


def _resolve_audio_path(path: str) -> Path:
    """play_audio_file の path を解決する。

    相対パス・絶対パスとも read_file と意味論を合わせて `_FILE_ROOT` 配下に
    confine する (LLM が `ls` で見た相対パスをそのまま渡しても、絶対パスを
    渡しても、root 外脱出は弾く)。`ensure_path_in_root` は絶対パスでも join +
    resolve 後に `relative_to` で判定するので、root 内の絶対パスは通り、外は
    `PathOutsideRoot` で弾かれる。見つからなければ似た名前を最大 5 件 hint に
    付けて raise。
    """
    if not _FILE_ROOT:
        raise ValueError(
            f"cannot resolve audio path {path!r}: AGENT_FILE_ROOT is not set"
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
    return p


def _is_audio_glob(path: str) -> bool:
    """True if `path` looks like a glob pattern (vs a single file path)."""
    return any(c in path for c in ("*", "?", "["))


def _expand_audio_glob(pattern: str) -> list[Path]:
    """Expand a glob (e.g. `*.wav`, `news/2026*.wav`, `**/*.wav`) to matching
    WAV files under `_FILE_ROOT`, sorted by name.

    Confined to the root: an absolute pattern must be inside it, `..` is
    rejected, and every match is re-checked with `relative_to(root)` (defence
    in depth). Non-`.wav` matches are skipped — the tool only plays WAV — so a
    bare `*` plays just the audio files. Raises if nothing matches so the LLM
    gets a clear error instead of silent no-op."""
    if not _FILE_ROOT:
        raise ValueError(
            f"cannot expand audio glob {pattern!r}: AGENT_FILE_ROOT is not set"
        )
    root = Path(_FILE_ROOT).resolve()
    pat = Path(pattern)
    if pat.is_absolute():
        try:
            rel = pat.relative_to(root)
        except ValueError as e:
            raise ValueError(
                f"glob {pattern!r} is outside the share root {root}"
            ) from e
    else:
        rel = pat
    if ".." in rel.parts:
        raise ValueError(f"glob must not contain '..': {pattern!r}")
    matches: list[Path] = []
    for p in sorted(root.glob(str(rel))):
        if not p.is_file() or p.suffix.lower() != ".wav":
            continue
        rp = p.resolve()
        try:
            rp.relative_to(root)
        except ValueError:
            continue
        matches.append(rp)
    if not matches:
        raise FileNotFoundError(
            f"no .wav files match glob {pattern!r} under the share root"
        )
    return matches


# Detached playback tasks (fire-and-forget). Hold strong refs so the event
# loop doesn't GC a running task mid-stream; the done-callback drops them.
_PLAYBACK_TASKS: set = set()


async def _stream_files_to_spk(
    session: aiohttp.ClientSession,
    ws: aiohttp.ClientWebSocketResponse,
    paths: list[Path],
    first_pcm: bytes,
    bytes_per_frame: int,
    label: str,
) -> None:
    """Background half of play_audio_file: realtime-stream each file's PCM over
    an ALREADY-CONNECTED `ws`, gaplessly, then EOS/drain, and always close
    ws + session.

    Files are decoded ONE AT A TIME, prefetching file i+1 while file i streams
    (file 0 was decoded in the foreground). So playback starts immediately, only
    ~one file of PCM is ever in memory, and there is NO total-duration cap — a
    big "play all" just streams for as long as it takes (stop it with stop_audio
    or wake-word). A file that fails to decode mid-run is logged and skipped.

    Connect failures were already surfaced to the LLM in the foreground; this
    detached task only does the realtime work, so play_audio_file stays
    non-blocking. Stops early when `/spk/stop?track=N` closes the ws."""
    next_task = None
    try:
        start = time.monotonic()
        frame_idx = 0
        pcm = first_pcm
        for i in range(len(paths)):
            # Decode the NEXT file while this one streams, so its PCM is ready
            # the instant the current file ends (gapless). Decode << realtime, so
            # it finishes during the per-frame sleeps below.
            next_task = (
                asyncio.create_task(
                    asyncio.to_thread(_read_and_convert_wav, paths[i + 1])
                )
                if i + 1 < len(paths)
                else None
            )
            offset = 0
            while offset < len(pcm):
                chunk = pcm[offset : offset + bytes_per_frame]
                offset += len(chunk)
                # 末尾の半端な chunk だけパディング (audio-io は奇数バイト拒否)。
                if len(chunk) % 2 != 0:
                    chunk = chunk + b"\x00"
                await ws.send_bytes(chunk)
                frame_idx += 1
                # 実時間ペーシング (連結全体で連続。ファイル境界も跨いで一定)。
                target = start + frame_idx * _AUDIO_PLAY_FRAME_MS / 1000.0
                sleep_for = target - time.monotonic()
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
            if next_task is not None:
                try:
                    pcm, _dur, _rate, _ch = await next_task
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "play_audio_file: skipping %s (%s: %s)",
                        paths[i + 1].name, type(e).__name__, e,
                    )
                    pcm = b""  # skipped file contributes no audio
        # Drain handshake: eos を送ってから "drained" が戻るまで待つ。
        await ws.send_str(json.dumps({"type": "eos"}))
        try:
            msg = await asyncio.wait_for(
                ws.receive(), timeout=_AUDIO_DRAIN_TIMEOUT_S
            )
            if msg.type != aiohttp.WSMsgType.TEXT:
                log.warning(
                    "play_audio_file: drain reply was %s, not TEXT", msg.type
                )
        except asyncio.TimeoutError:
            log.warning(
                "play_audio_file: drain handshake timed out after %.0fs",
                _AUDIO_DRAIN_TIMEOUT_S,
            )
        log.info("play_audio_file: finished playback (%s)", label)
    except asyncio.CancelledError:
        log.info("play_audio_file: playback cancelled (%s)", label)
        raise
    except Exception as e:  # noqa: BLE001
        # WS closed by /spk/stop mid-stream, connection dropped, etc. Once
        # streaming has started these are normal "playback ended" conditions
        # (setup failures were already caught at connect time), so just log.
        log.info(
            "play_audio_file: playback ended early (%s): %s", type(e).__name__, e
        )
    finally:
        # Disown any in-flight prefetch decode (e.g. stopped mid-sequence: the
        # ws was closed by /spk/stop, so the remaining files won't play) and
        # always release the foreground-opened ws + session.
        if next_task is not None and not next_task.done():
            next_task.cancel()
        try:
            await ws.close()
        finally:
            await session.close()


async def _play_audio_file_impl(paths: list[str] | str) -> str:
    """play_audio_file 本体。全 WAV を検証・デコードし、audio-io /spk (?track=N)
    WS を **前景で接続** してから、realtime 送信＋drain を **背景タスク** に投げて
    即 return する (非ブロック・ギャップレス連続再生)。

    接続失敗 (audio-io 未到達 / track 不正) はこの前景で raise され `_safe_invoke`
    が string 化して LLM に返す (= 再生できなかったと気付ける)。再生開始後の停止は
    `stop_audio` ツール / 任意プロセスの `POST /spk/stop?track=N` / wake-word
    barge-in のいずれでも効く (audio-io が ring flush ＋ その track の WS close →
    背景タスクの送信が例外で終了)。背景タスクは終了時に ws+session を必ず閉じる。
    """
    if not _AUDIO_IO_SPK_URL:
        raise RuntimeError(
            "AGENT_AUDIO_IO_SPK_URL is not set — audio playback is disabled"
        )
    # LLM が単一文字列を渡してくる可能性があるので list に正規化する
    # (tool スキーマは array だが boundary なので念のため吸収)。
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        raise ValueError("no audio file given; pass at least one WAV path")

    # 各 entry をファイル or glob として解決する。glob (`*.wav` 等) は root 配下の
    # 一致 .wav を名前順で展開。1 件でも見つからない / 不正なら、再生を一切始める
    # 前に弾く (連続再生は単一 drain なので、途中中断より全件検証が安全)。
    resolved: list[Path] = []
    for path in paths:
        if _is_audio_glob(path):
            resolved.extend(_expand_audio_glob(path))
        else:
            resolved.append(_resolve_audio_path(path))

    bytes_per_sample = _AUDIO_SAMPLE_WIDTH * _AUDIO_IO_WIRE_CHANNELS
    samples_per_frame = _AUDIO_IO_WIRE_RATE * _AUDIO_PLAY_FRAME_MS // 1000
    bytes_per_frame = samples_per_frame * bytes_per_sample
    names = ", ".join(p.name for p in resolved)

    # Decode only the FIRST file up front (in a thread): gives an immediate chunk
    # to stream and surfaces a gross decode error (corrupt / non-PCM WAV) to the
    # LLM before we claim playback started. Files 2..N are decoded just-in-time
    # inside the streamer (prefetched one ahead), so playback starts fast, memory
    # stays at ~one file, and there is no total-duration cap.
    first_pcm, _first_dur, _first_rate, _first_ch = await asyncio.to_thread(
        _read_and_convert_wav, resolved[0]
    )
    label = f"{len(resolved)} file(s) [{names}]"
    log.info(
        "play_audio_file: %d file(s) [%s] → %s (streamed one at a time)",
        len(resolved), names, _AUDIO_IO_SPK_URL,
    )

    # Connect to audio-io in the FOREGROUND so a connection failure (audio-io
    # down, wrong host, bad track) raises here and is surfaced to the LLM via
    # _safe_invoke — instead of being silently swallowed by the detached task.
    # Only the realtime streaming is backgrounded, so playback stays non-blocking.
    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=10.0)
    )
    try:
        ws = await session.ws_connect(_AUDIO_IO_SPK_URL)
    except Exception as e:  # noqa: BLE001
        await session.close()
        raise RuntimeError(
            f"could not connect to audio output at {_AUDIO_IO_SPK_URL} "
            f"({type(e).__name__}: {e}) — audio was NOT played"
        ) from e
    # Hand the live connection + the resolved file list to the detached streamer;
    # it decodes the rest one-at-a-time and closes ws+session when done.
    task = asyncio.create_task(
        _stream_files_to_spk(session, ws, resolved, first_pcm, bytes_per_frame, label)
    )
    _PLAYBACK_TASKS.add(task)
    task.add_done_callback(_PLAYBACK_TASKS.discard)
    return (
        f"started playing {len(resolved)} file(s) in the background; you're free "
        f"now. They play gapless in order and stop at the end, or immediately "
        f"when stop_audio is called."
    )


def _build_play_audio_description() -> str:
    """Tool description を起動時の env (`_AUDIO_IO_SPK_URL` の有無) に応じて
    動的に生成する。URL 未設定なら「無効」を明示し、LLM が呼ばずに状況を
    user に伝えるよう誘導する。"""
    if not _AUDIO_IO_SPK_URL:
        return (
            "Play WAV audio file(s) on the speaker.\n"
            "**UNAVAILABLE** — audio output is not configured here. Do NOT "
            "call this; tell the user audio playback isn't set up."
        )
    path_hint = (
        f"`paths`: WAV paths under `{_FILE_ROOT}` — relative to that root or "
        f"absolute inside it. An entry may be a glob (`*.wav`, "
        f"`news/2026*.wav`) matching every .wav under the root."
        if _FILE_ROOT
        else "`paths`: WAV file paths (AGENT_FILE_ROOT is unset, so playback "
        "is effectively disabled)."
    )
    return (
        "Play WAV audio file(s) on the speaker. Call when the user asks to "
        "play an audio file.\n"
        f"{path_hint} Multiple paths play back-to-back in order. Sample rate "
        "and channels are auto-converted; only uncompressed 16-bit PCM WAV.\n"
        "Plays in the background — this returns immediately, and a missing "
        "file means nothing plays (the error suggests similar filenames; "
        "retry with one). Stop with stop_audio."
    )


@tool(description=_build_play_audio_description())
async def play_audio_file(paths: list[str]) -> str:
    # description は @tool(description=...) で動的注入。詳細は
    # `_build_play_audio_description` の docstring を参照。
    return await _safe_invoke("play_audio_file", _play_audio_file_impl(paths))


async def _stop_audio_impl() -> str:
    """stop_audio 本体。audio-io の /spk/stop?track=N を POST して、
    play_audio_file が使う track のリングを flush する。

    flush は「今鳴っている / バッファに残っている PCM を捨てる」操作なので、
    何も鳴っていなくても 200 が返る no-op。LLM の声 (TTS) は別 track なので
    影響しない。即応操作なので drain handshake は不要。
    """
    if not _AUDIO_IO_STOP_URL:
        raise RuntimeError(
            "AGENT_AUDIO_IO_SPK_URL is not set — audio stop is disabled"
        )
    timeout = aiohttp.ClientTimeout(total=_AUDIO_STOP_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(_AUDIO_IO_STOP_URL) as resp:
            body = await resp.text()
            if resp.status != 200:
                # 404 = track 範囲外 (audio-io が当該 track を持っていない)
                # など。retry しても通らないが [error] で LLM に状況を渡す。
                raise RuntimeError(
                    f"audio-io /spk/stop returned {resp.status}: {body[:200]}"
                )
    log.info("stop_audio: flushed via %s", _AUDIO_IO_STOP_URL)
    return "stopped audio playback"


def _build_stop_audio_description() -> str:
    """stop_audio の description を起動時 env に応じて動的生成する。"""
    if not _AUDIO_IO_STOP_URL:
        return (
            "Stop audio file playback.\n"
            "**UNAVAILABLE** — audio output is not configured here. Do NOT "
            "call this; tell the user audio isn't set up."
        )
    return (
        "Stop the audio file that is playing (started by play_audio_file). "
        "Call when the user says e.g. 「音声止めて」「再生やめて」.\n"
        "No arguments. Your own spoken voice is unaffected; harmless no-op "
        "if nothing is playing."
    )


@tool(description=_build_stop_audio_description())
async def stop_audio() -> str:
    # description は @tool(description=...) で動的注入。詳細は
    # `_build_stop_audio_description` の docstring を参照。
    return await _safe_invoke("stop_audio", _stop_audio_impl())


# --- system_health (自己診断「システムチェックして」, Phase 1) ---------------
#
# 各サービスの liveness を 1 ショットで並列集約し、短い日本語サマリを返す
# self-diagnostic tool。対象 URL は compose ネットワーク内のサービス名 +
# コンテナ内部ポート (host 公開ポートではない — agent 自身がネットワーク内に
# いるため)。audio-io だけは Windows ネイティブ (compose 外) なので、再生 tool
# と同じ SPK URL から host:port を借りて /health を導出する。
#
# terminal tool にはしない: 結果が動的 (どれが落ちているか) なので固定 ack
# では伝わらない。非 terminal にして LLM に ToolMessage (このサマリ) を読ませ、
# そのまま読み上げさせる。probe 中の無音は "確認するね。" の ack で埋める。

# ollama (llm) の base URL。agent が chat に使うのと同じ env を流用する。
_OLLAMA_URL: str = os.environ.get("AGENT_OLLAMA_URL", "http://llm:11434").rstrip("/")
# 各 probe の総タイムアウト (秒)。全 probe は gather で並列に走るので自己診断
# 全体もおおむねこの時間内に返る。落ちている対象を素早く「不調」と判定するため
# 短め。
_HEALTH_TIMEOUT_S: float = float(os.environ.get("AGENT_HEALTH_TIMEOUT_S", "2.5"))


def _derive_audio_io_health_url(spk_url: str) -> str:
    """/spk WS URL から audio-io の /health HTTP URL を導出する。

    `ws://host:7010/spk?track=1` → `http://host:7010/health`。audio-io は
    Windows ネイティブ (compose 外) で、agent が知っている唯一の到達情報が
    SPK URL なので、そこから scheme (ws→http) と host:port だけ借り、path は
    /health に差し替え、track クエリは捨てる。SPK URL 未設定なら空文字を返し、
    audio-io は probe 対象から外れる。
    """
    if not spk_url:
        return ""
    parts = urlsplit(spk_url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, "/health", "", ""))


_AUDIO_IO_HEALTH_URL: str = _derive_audio_io_health_url(_AUDIO_IO_SPK_URL)


def _default_health_targets() -> list[dict]:
    """probe 対象のデフォルト一覧。name は音声で読み上げる前提の短い日本語。

    kind:
      "http"    … HTTP 2xx のみ healthy (liveness + 軽い readiness)。
      "connect" … HTTP 応答が返れば (ステータス不問) healthy。POST 専用
                  エンドポイント (whisper.cpp /inference) 用 — GET は 4xx に
                  なるが「サーバが応答している」= 生きている証拠。

    除外: voice-activity-detection は health エンドポイントを持たない。

    NB: wake-word-detection の /health は「モデル loaded かつ mic WS 接続済」の
    ときだけ 200 を返す (それ以外 503)。つまり audio-io.exe が落ちて /mic に
    繋がらないと WWD も 503 になるので、この probe は audio-io 断も間接的に拾う。
    """
    targets: list[dict] = []
    # audio-io (Windows ネイティブ)。SPK URL 未設定ならスキップ。
    if _AUDIO_IO_HEALTH_URL:
        targets.append({"name": "音声入出力", "url": _AUDIO_IO_HEALTH_URL, "kind": "http"})
    # 以降は compose ネットワーク内のサービス。GET /api/tags 200 = ollama 生存
    # (Phase 1 は liveness のみ — モデルの load 状態確認は Phase 2 に回す)。
    targets += [
        {"name": "言語モデル", "url": f"{_OLLAMA_URL}/api/tags", "kind": "http"},
        {"name": "司令塔", "url": "http://orchestrator:7000/health", "kind": "http"},
        {
            "name": "音声認識",
            "url": "http://automatic-speech-recognition:8080/inference",
            "kind": "connect",
        },
        {"name": "音声合成", "url": "http://text-to-speech:50021/version", "kind": "http"},
        {"name": "音声合成の送信", "url": "http://tts-streamer:7070/health", "kind": "http"},
        {"name": "ウェイクワード", "url": "http://wake-word-detection:7030/health", "kind": "http"},
    ]
    return targets


def _health_targets() -> list[dict]:
    """probe 対象。env `AGENT_HEALTH_TARGETS` (JSON 配列) があればそれで上書き。

    形式: `[{"name": "...", "url": "http://...", "kind": "http"|"connect"}, ...]`。
    構成を変えた (サービス追加 / ポート変更) 際にコード変更なしで追従するための
    逃げ道。パース失敗時は warning を出してデフォルトに倒す。
    """
    raw = os.environ.get("AGENT_HEALTH_TARGETS", "").strip()
    if not raw:
        return _default_health_targets()
    try:
        parsed = json.loads(raw)
        targets = [
            {
                "name": str(t["name"]),
                "url": str(t["url"]),
                "kind": str(t.get("kind", "http")),
            }
            for t in parsed
        ]
        if not targets:
            raise ValueError("empty target list")
        return targets
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        log.warning("AGENT_HEALTH_TARGETS invalid (%s); using defaults", e)
        return _default_health_targets()


async def _probe_url(session: aiohttp.ClientSession, target: dict) -> dict:
    """1 対象を 1 回 GET して `{name, ok, detail}` を返す。

    例外は全て握りつぶして ok=False に変換する — 1 つ落ちている対象が
    `asyncio.gather` 全体を倒さないため。detail は log / デバッグ用で、音声
    サマリには name しか出さない。
    """
    name = target["name"]
    url = target["url"]
    kind = target.get("kind", "http")
    try:
        async with session.get(url) as resp:
            status = resp.status
            # "connect" は応答が返った時点で生存確認 (POST 専用 endpoint 対策)。
            # "http" は 2xx のみ healthy (例: WWD は未接続だと 503 を返す)。
            ok = True if kind == "connect" else (200 <= status < 300)
            return {"name": name, "ok": ok, "detail": f"HTTP {status}"}
    except asyncio.TimeoutError:
        # aiohttp の ServerTimeoutError もここで捕捉される (asyncio.TimeoutError
        # のサブクラス)。タイムアウト = 応答なし = 不調扱い。
        return {"name": name, "ok": False, "detail": "timeout"}
    except aiohttp.ClientError as e:
        # 接続拒否 / 名前解決失敗 (コンテナ未起動) 等。
        return {"name": name, "ok": False, "detail": type(e).__name__}
    except Exception as e:  # noqa: BLE001
        return {"name": name, "ok": False, "detail": f"{type(e).__name__}: {e}"}


async def _system_health_impl() -> str:
    """system_health 本体。全 probe を並列に走らせ短い日本語サマリを返す。"""
    targets = _health_targets()
    timeout = aiohttp.ClientTimeout(total=_HEALTH_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*(_probe_url(session, t) for t in targets))
    for r in results:
        log.info("system_health: %s ok=%s (%s)", r["name"], r["ok"], r["detail"])
    unhealthy = [r for r in results if not r["ok"]]
    total = len(results)
    healthy_n = total - len(unhealthy)
    log.info(
        "system_health summary: %d/%d ok, down=%s",
        healthy_n,
        total,
        [r["name"] for r in unhealthy],
    )
    if not unhealthy:
        return f"システムは正常だよ。{total}個のサービス、全部動いてる。"
    if healthy_n == 0:
        # 全滅。個別に名前を並べても情報量がない (むしろ network/DNS 障害の
        # 可能性が高い) ので、まとめて伝える。
        return f"全部のサービスが応答してないよ。{total}個とも不調。"
    names = "、".join(r["name"] for r in unhealthy)
    return f"{names}が不調だよ。ほかの{healthy_n}個は正常。"


def _build_system_health_description() -> str:
    """system_health の description を起動時 env (probe 対象) に応じて生成する。"""
    names = "、".join(t["name"] for t in _health_targets())
    return (
        "Self-diagnose the assistant's own backend services "
        f"({names}) and return a short Japanese summary. "
        "Call when the user asks for a system check (e.g. 「システムチェック"
        "して」). No arguments, read-only. Speak the summary back as-is; "
        "don't invent services it didn't mention."
    )


@tool(description=_build_system_health_description())
async def system_health() -> str:
    # description は @tool(description=...) で動的注入。詳細は
    # `_build_system_health_description` の docstring を参照。非 terminal tool
    # なので、返り値 (サマリ) は LLM が読み上げる。
    return await _safe_invoke("system_health", _system_health_impl())


# --- timer (タイマー起動 + 起動中タイマーの確認 + キャンセル) ----------------
#
# タイマーは agent プロセス内の asyncio バックグラウンドタスク。発火 (= sleep
# 満了) すると audio-io の別 track (既定 track=2) にアラーム音を直送する。
# タイマー発火は「ユーザーのターン」が無い非同期イベントなので、TTS (track 0)
# や play_audio_file (track 1) と違い orchestrator 経由で喋れない —— track 直送
# が agent の唯一の自律出力チャネル。track を分けてあるので、発火時に TTS や
# ファイル再生が鳴っていても WASAPI 共有ミックスで重なって聞こえる。
#
# 制約: タイマーは in-memory。agent を再起動すると消える (永続化は将来課題)。
# アラームは track=2 → audio-io 側で runtime.playback_tracks >= 3 (track 0/1/2)
# が必要。track が無いと audio-io が WS を即 close し、アラームは鳴らず warn
# ログだけ残る。

# アラーム送出先 track。play 用 SPK URL の ?track= を差し替えて track=2 を狙う。
# AGENT_TIMER_SPK_URL で URL ごと、AGENT_TIMER_TRACK で track 番号だけ上書き可。
_TIMER_TRACK: int = int(os.environ.get("AGENT_TIMER_TRACK", "2"))


def _derive_track_url(spk_url: str, track: int) -> str:
    """spk WS URL の `?track=` を `track` に差し替えた URL を返す。

    他のクエリは温存して track だけ差し替える。空 URL なら空文字 (= 機能無効)。
    """
    if not spk_url:
        return ""
    parts = urlsplit(spk_url)
    q = parse_qs(parts.query)
    q["track"] = [str(track)]
    new_query = urlencode({k: v[-1] for k, v in q.items()})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


_TIMER_SPK_URL: str = (
    os.environ.get("AGENT_TIMER_SPK_URL", "").strip()
    or _derive_track_url(_AUDIO_IO_SPK_URL, _TIMER_TRACK)
)
# タイマー長の上限 (秒)。誤認識で「3分」が「300分」等にならない保険＋暴走防止。
_TIMER_MAX_S: float = float(os.environ.get("AGENT_TIMER_MAX_S", "86400"))  # 24h

# 起動中タイマーの registry: id -> {label, fire_at(monotonic), duration_s, task}。
# check/cancel が参照する。発火 or cancel で entry を pop する。
_TIMERS: dict[int, dict] = {}
# タイマー id の採番 (単調増加、再利用しない)。agent 再起動でリセット。
_timer_seq: int = 0
# タスクの強参照保持。registry は発火時に entry を pop するので、それだけだと
# アラーム再生中にタスクが GC されうる。play_audio_file の _PLAYBACK_TASKS と同型。
_TIMER_TASKS: set = set()
# 生成アラーム PCM の memo (決定的なので 1 度作れば使い回す)。
_ALARM_PCM_CACHE: bytes | None = None


def _fmt_duration(sec: float) -> str:
    """秒を「X時間Y分Z秒」の日本語表記に。ゼロの単位は省略。0 以下は「0秒」。"""
    s = int(round(sec))
    if s <= 0:
        return "0秒"
    h, rem = divmod(s, 3600)
    m, s2 = divmod(rem, 60)
    out = ""
    if h:
        out += f"{h}時間"
    if m:
        out += f"{m}分"
    if s2:
        out += f"{s2}秒"
    return out


def _gen_tone(freq: float, dur_s: float, rate: int, amp: float = 0.5) -> bytes:
    """単一周波数のビープ (s16le mono) を生成。両端 5ms フェードでクリック除去。"""
    n = int(rate * dur_s)
    fade = max(1, int(rate * 0.005))
    peak = amp * 32767.0
    samples = []
    for i in range(n):
        if i < fade:
            env = i / fade
        elif i >= n - fade:
            env = max(0.0, (n - i) / fade)
        else:
            env = 1.0
        samples.append(int(peak * env * math.sin(2.0 * math.pi * freq * i / rate)))
    return struct.pack("<%dh" % n, *samples)


def _alarm_pcm() -> bytes:
    """キッチンタイマー風「ピピピピ」を 3 回繰り返した s16le PCM を生成する。"""
    rate = _AUDIO_IO_WIRE_RATE
    beep = _gen_tone(880.0, 0.12, rate)
    short_gap = b"\x00\x00" * int(rate * 0.08)
    long_gap = b"\x00\x00" * int(rate * 0.40)
    burst = beep + short_gap + beep + short_gap + beep + short_gap + beep
    pcm = (burst + long_gap) * 3
    # wire がステレオなら mono→stereo に複製 (既定は mono なので通常は素通り)。
    if _AUDIO_IO_WIRE_CHANNELS == 2:
        pcm = audioop.tostereo(pcm, _AUDIO_SAMPLE_WIDTH, 1.0, 1.0)
    return pcm


def _alarm_pcm_cached() -> bytes:
    """アラーム PCM を返す。AGENT_TIMER_ALARM_WAV があればそれを wire format に
    変換して使い、無ければ生成ビープ。結果は 1 度だけ作って memo する。"""
    global _ALARM_PCM_CACHE
    if _ALARM_PCM_CACHE is not None:
        return _ALARM_PCM_CACHE
    pcm: bytes | None = None
    custom = os.environ.get("AGENT_TIMER_ALARM_WAV", "").strip()
    if custom:
        try:
            p = _resolve_audio_path(custom)
            pcm, _d, _r, _c = _read_and_convert_wav(p)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "timer: custom alarm WAV %r unusable (%s: %s); using generated beep",
                custom, type(e).__name__, e,
            )
            pcm = None
    if pcm is None:
        pcm = _alarm_pcm()
    _ALARM_PCM_CACHE = pcm
    return pcm


async def _play_pcm_on_track(spk_url: str, pcm: bytes, label: str) -> None:
    """生 PCM を指定 track の /spk へ realtime 送信し、drain して閉じる。

    play_audio_file の送信ループと同型だが、対象は in-memory の単一バッファ
    (ファイル解決 / prefetch 無し)。発火は非同期でターンが無いため、失敗は
    raise せず warn ログのみ (呼び出し元に拾い手がいない)。track 未設定だと
    audio-io が WS を即 close するので、その場合もここで warn に落ちる。"""
    if not spk_url:
        log.warning("timer alarm: no spk url configured; cannot play (%s)", label)
        return
    bytes_per_sample = _AUDIO_SAMPLE_WIDTH * _AUDIO_IO_WIRE_CHANNELS
    samples_per_frame = _AUDIO_IO_WIRE_RATE * _AUDIO_PLAY_FRAME_MS // 1000
    bytes_per_frame = samples_per_frame * bytes_per_sample
    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=10.0)
    )
    try:
        ws = await session.ws_connect(spk_url)
    except Exception as e:  # noqa: BLE001
        await session.close()
        log.warning(
            "timer alarm: could not connect to %s (%s: %s) — alarm NOT played (%s)",
            spk_url, type(e).__name__, e, label,
        )
        return
    try:
        start = time.monotonic()
        frame_idx = 0
        offset = 0
        while offset < len(pcm):
            chunk = pcm[offset : offset + bytes_per_frame]
            offset += len(chunk)
            if len(chunk) % 2 != 0:
                chunk = chunk + b"\x00"
            await ws.send_bytes(chunk)
            frame_idx += 1
            target = start + frame_idx * _AUDIO_PLAY_FRAME_MS / 1000.0
            sleep_for = target - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        await ws.send_str(json.dumps({"type": "eos"}))
        try:
            await asyncio.wait_for(ws.receive(), timeout=_AUDIO_DRAIN_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.warning("timer alarm: drain timed out (%s)", label)
        log.info("timer alarm: played (%s)", label)
    except Exception as e:  # noqa: BLE001
        log.warning(
            "timer alarm: ended early (%s: %s) — is audio-io track configured? (%s)",
            type(e).__name__, e, label,
        )
    finally:
        try:
            await ws.close()
        finally:
            await session.close()


async def _timer_fire(timer_id: int, label: str) -> None:
    """発火処理: registry から外し、track にアラームを流す。"""
    _TIMERS.pop(timer_id, None)
    tag = f" ({label})" if label else ""
    log.info("timer #%d fired%s; playing alarm on %s", timer_id, tag, _TIMER_SPK_URL)
    await _play_pcm_on_track(_TIMER_SPK_URL, _alarm_pcm_cached(), f"timer #{timer_id}{tag}")


async def _timer_task(timer_id: int, seconds: float, label: str) -> None:
    """1 本のタイマー。sleep 満了で発火、cancel されたら静かに終了。"""
    try:
        await asyncio.sleep(seconds)
    except asyncio.CancelledError:
        log.info("timer #%d cancelled before firing", timer_id)
        return
    await _timer_fire(timer_id, label)


async def _start_timer_impl(seconds, label: str = "") -> str:
    """start_timer 本体。タスクを起こして registry に登録、即 return する。"""
    global _timer_seq
    if not _TIMER_SPK_URL:
        raise RuntimeError(
            "AGENT_AUDIO_IO_SPK_URL is not set — a timer has no way to sound "
            "its alarm, so timers are disabled"
        )
    try:
        seconds = float(seconds)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"seconds must be a number of seconds, got {seconds!r}"
        ) from e
    if seconds <= 0:
        raise ValueError("timer duration must be positive (seconds > 0)")
    if seconds > _TIMER_MAX_S:
        raise ValueError(
            f"timer too long ({_fmt_duration(seconds)}); max is "
            f"{_fmt_duration(_TIMER_MAX_S)}"
        )
    label = (label or "").strip()
    _timer_seq += 1
    tid = _timer_seq
    task = asyncio.create_task(_timer_task(tid, seconds, label))
    _TIMER_TASKS.add(task)
    _TIMERS[tid] = {
        "label": label,
        "fire_at": time.monotonic() + seconds,
        "duration_s": seconds,
        "task": task,
    }

    def _done(t: asyncio.Task, tid: int = tid) -> None:
        # 強参照を解放しつつ、異常終了でも registry に残骸を残さない
        # (正常発火 / cancel は既に pop 済みなので no-op)。
        _TIMER_TASKS.discard(t)
        _TIMERS.pop(tid, None)

    task.add_done_callback(_done)
    log.info(
        "timer #%d set for %s%s",
        tid, _fmt_duration(seconds), f" ({label})" if label else "",
    )
    lead = f"「{label}」の" if label else ""
    return f"{lead}{_fmt_duration(seconds)}タイマーをかけたよ。（番号{tid}）"


async def _check_timers_impl() -> str:
    """check_timers 本体。起動中タイマーと残り時間を日本語サマリで返す。"""
    if not _TIMERS:
        return "今は動いてるタイマーは無いよ。"
    now = time.monotonic()
    items = sorted(_TIMERS.items(), key=lambda kv: kv[1]["fire_at"])
    parts = []
    for tid, e in items:
        remaining = max(0.0, e["fire_at"] - now)
        who = f"「{e['label']}」" if e["label"] else f"番号{tid}"
        parts.append(f"{who}があと{_fmt_duration(remaining)}")
    if len(parts) == 1:
        return parts[0] + "だよ。"
    return f"タイマーが{len(parts)}件。" + "、".join(parts) + "。"


async def _cancel_timer_impl(which: str = "") -> str:
    """cancel_timer 本体。番号 / ラベル部分一致 / 「全部」/ 唯一 で対象を選び cancel。"""
    if not _TIMERS:
        return "今は動いてるタイマーは無いよ。"
    which = (which or "").strip()
    if which in ("全部", "すべて", "ぜんぶ", "みんな", "all", "ALL"):
        target_ids = list(_TIMERS.keys())
    elif not which:
        if len(_TIMERS) == 1:
            target_ids = [next(iter(_TIMERS))]
        else:
            opts = "、".join((e["label"] or f"番号{t}") for t, e in _TIMERS.items())
            return f"タイマーが{len(_TIMERS)}件あるよ。どれ止める？（{opts}）"
    elif which.isdigit() and int(which) in _TIMERS:
        target_ids = [int(which)]
    else:
        target_ids = [t for t, e in _TIMERS.items() if which in (e["label"] or "")]
        if not target_ids:
            return f"「{which}」に合うタイマーが見つからないよ。"
    cancelled = []
    for tid in target_ids:
        e = _TIMERS.pop(tid, None)
        if e:
            e["task"].cancel()
            cancelled.append(e["label"] or f"番号{tid}")
    return "、".join(cancelled) + "のタイマーを止めたよ。"


def _build_start_timer_description() -> str:
    """start_timer の description を起動時 env (_TIMER_SPK_URL の有無) で生成。"""
    if not _TIMER_SPK_URL:
        return (
            "Start a countdown timer.\n"
            "**UNAVAILABLE** — no audio output for the alarm in this "
            "environment. Do NOT call this; tell the user timers aren't "
            "set up."
        )
    return (
        "Start a countdown timer; an alarm sounds on the speaker when it "
        "ends. Call when the user asks for a timer/alarm (e.g. 「3分タイマー"
        "かけて」「10分後に教えて」).\n"
        "`seconds`: duration in seconds — convert the spoken duration "
        "yourself (「3分」→180, 「1時間」→3600). `label`: optional short name "
        "(e.g. 「パスタ」). Returns immediately; confirm the duration back "
        "to the user."
    )


@tool(description=_build_start_timer_description())
async def start_timer(seconds: int, label: str = "") -> str:
    # description は @tool(description=...) で動的注入。非 terminal: 返り値
    # (セットした長さの確認) を LLM が読み上げてユーザーに復唱する。
    return await _safe_invoke("start_timer", _start_timer_impl(seconds, label))


@tool
async def check_timers() -> str:
    """List the running timers and the time left on each.

    Call when the user asks about timers (e.g. 「タイマーあと何分?」「今タイ
    マー動いてる?」). No arguments. Speak the result back to the user.
    """
    return await _safe_invoke("check_timers", _check_timers_impl())


@tool
async def cancel_timer(which: str = "") -> str:
    """Cancel a running timer so its alarm won't sound.

    Call when the user asks to stop/cancel a timer (e.g. 「タイマー止めて」
    「タイマー全部キャンセル」). `which`: a label substring, a timer number,
    or 「全部」 for all; omit it when only one timer is running.
    """
    return await _safe_invoke("cancel_timer", _cancel_timer_impl(which))


@tool
async def read_file(path: str) -> str:
    """Read a text file and return its contents.

    Call when the user asks to read a memo or file — then summarise or
    read it aloud. `path` is relative to the shared directory (e.g.
    "memo.txt", "docs/recipe.md"); paths outside it are rejected.
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


# --- conversation-memory reset -----------------------------------------
# Handle for `reset_memory` to clear the LangGraph conversation history.
# Tools are standalone functions with no reference to app.py's
# SessionManager, so app.py registers a callback here at startup
# (set_reset_memory_hook(sessions.reset)). The callback rotates the session
# id → the NEXT turn runs on a fresh thread_id = empty memory. The current
# turn (the "reset" request itself) stays on the old, now-abandoned thread.
_RESET_MEMORY_HOOK = None


def set_reset_memory_hook(fn) -> None:
    """Register the async () -> str callback that clears conversation memory.
    app.py wires this to SessionManager.reset."""
    global _RESET_MEMORY_HOOK
    _RESET_MEMORY_HOOK = fn


async def _reset_memory_impl() -> str:
    if _RESET_MEMORY_HOOK is None:
        # Tools enabled but app.py never wired the hook — surface, don't lie.
        raise RuntimeError("reset hook not registered")
    await _RESET_MEMORY_HOOK()
    return "会話の記憶をリセットした"


@tool
async def reset_memory() -> str:
    """Clear the conversation memory and start fresh.

    Call when the user asks to forget the conversation or start over
    (e.g. 「記憶リセットして」「履歴消して」「最初から」). No arguments.
    """
    return await _safe_invoke("reset_memory", _reset_memory_impl())


# --- tool 登録 ----------------------------------------------------------

def all_tools() -> list:
    """登録 tool の一覧。`create_react_agent` に渡す。

    issue #19 の段階的な追加に伴い後段の PR で別 tool が積まれる可能性あり。
    """
    return [
        run_shell,
        read_file,
        web_search,
        play_audio_file,
        stop_audio,
        system_health,
        start_timer,
        check_timers,
        cancel_timer,
        reset_memory,
    ]


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
    # stop は即応かつ決定的なので、2回目の LLM 生成を省く (agent 側で stop_audio
    # を terminal tool 扱いにして即 END)。確認文はこの固定 ack で返す。
    "stop_audio": os.environ.get("AGENT_TOOL_ACK_STOP_AUDIO", "止めたよ。"),
    # system_health は全サービスを並列 probe する間 (~2.5s) の無音を埋める ack。
    # 結果は非 terminal なので、この ack の後に LLM が診断サマリを読み上げる。
    "system_health": os.environ.get("AGENT_TOOL_ACK_SYSTEM_HEALTH", "確認するね。"),
    # timer 系は即応 (タスク登録 / registry 参照 / cancel)。filler 不要なので
    # None。非 terminal なので直後に LLM が結果 (確認文・残り時間) を読み上げる。
    "start_timer": None,
    "check_timers": None,
    "cancel_timer": None,
    # reset_memory is terminal (no 2nd LLM call) — this fixed line IS the
    # spoken confirmation. Overridable like the others.
    "reset_memory": os.environ.get("AGENT_TOOL_ACK_RESET_MEMORY", "記憶をリセットしたよ。"),
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
#   python tools.py play_audio_file share/a.wav share/b.wav   # 連続再生
#   python tools.py stop_audio                                 # 再生停止
#   python tools.py system_health                              # 自己診断
#   python tools.py start_timer seconds=10 label=test          # タイマー起動
#   python tools.py check_timers                               # 起動中の確認
#   python tools.py cancel_timer which=test                    # 取消
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
        # 引数なしの tool (e.g. stop_audio) は名前だけで実行する。
        # スキーマに引数があるものは従来通り description を表示。
        if not (tool_obj.args or {}):
            print(await tool_obj.ainvoke({}))
            return 0
        _cli_show(tool_obj)
        return 0

    if name == "play_audio_file":
        # `paths` は list なので汎用の positional zip では扱えない。
        # 残り argv を全部まとめて 1 つの list として渡す。
        args = {"paths": argv[1:]}
    else:
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
