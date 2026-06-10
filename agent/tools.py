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

_SHELL_ALLOWLIST: set[str] = {
    s.strip()
    for s in os.environ.get("AGENT_SHELL_ALLOWLIST", "").split(",")
    if s.strip()
}
_FILE_ROOT: str = os.environ.get("AGENT_FILE_ROOT", "")

_AUDIO_IO_SPK_URL: str = os.environ.get("AGENT_AUDIO_IO_SPK_URL", "")
_AUDIO_IO_WIRE_RATE: int = int(os.environ.get("AGENT_AUDIO_IO_WIRE_RATE", "16000"))
_AUDIO_IO_WIRE_CHANNELS: int = int(os.environ.get("AGENT_AUDIO_IO_WIRE_CHANNELS", "1"))
_AUDIO_SAMPLE_WIDTH = 2


def _derive_stop_url(spk_url: str) -> str:
    """`?track=N` は必須。未指定だと派生する /spk/stop も track なし = audio-io
    側で全 track flush (TTS の track 0 まで巻き込む) になるため、起動時に弾く。
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
    path = (path[: -len("/spk")] + "/spk/stop") if path.endswith("/spk") else path + "/stop"
    return urlunsplit((scheme, parts.netloc, path, parts.query, parts.fragment))


_AUDIO_IO_STOP_URL: str = _derive_stop_url(_AUDIO_IO_SPK_URL)
_AUDIO_STOP_TIMEOUT_S = 5.0

_SHELL_STDOUT_MAX = 3000
_FILE_READ_MAX = 51200
_WEB_SEARCH_MAX_RESULTS = 5
_WEB_SEARCH_SNIPPET_MAX = 120
_WEB_SEARCH_TOTAL_MAX = 1500
# 20ms は audio-io の既定 frame_ms に合わせる。
_AUDIO_PLAY_FRAME_MS = 20
_AUDIO_DRAIN_TIMEOUT_S = 30.0

_SHELL_TIMEOUT_S = 5.0
_WEB_SEARCH_TIMEOUT_S = 5.0


async def _run_shell_impl(command: str) -> str:
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
        proc.kill()
        await proc.wait()
        raise TimeoutError(
            f"command timed out after {_SHELL_TIMEOUT_S:.1f}s: {argv[0]}"
        )
    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
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
    return await _safe_invoke("run_shell", _run_shell_impl(command))


async def _read_file_impl(path: str) -> str:
    if not _FILE_ROOT:
        raise RuntimeError(
            "AGENT_FILE_ROOT is not configured — read_file is disabled"
        )
    resolved = ensure_path_in_root(path, _FILE_ROOT)
    if not resolved.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    data = await asyncio.to_thread(resolved.read_bytes)
    truncated = len(data) > _FILE_READ_MAX
    text = data[:_FILE_READ_MAX].decode("utf-8", errors="replace")
    if truncated:
        text += "\n[...truncated]"
    return text


async def _web_search_impl(query: str) -> str:
    query = (query or "").strip()
    if not query:
        raise ValueError("query is empty")

    # 旧パッケージ名 `duckduckgo_search` は 2025 年に `ddgs` にリネーム
    # されており、旧名で import すると検索が空 list を返す挙動になる。
    from ddgs import DDGS

    def _do_search() -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=_WEB_SEARCH_MAX_RESULTS))

    results = await asyncio.wait_for(
        asyncio.to_thread(_do_search),
        timeout=_WEB_SEARCH_TIMEOUT_S,
    )
    if not results:
        return "(no search results)"

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
    """`ensure_path_in_root` は絶対パスでも join + resolve 後に `relative_to`
    で判定するので、root 内の絶対パスは通り、外は弾かれる。"""
    if not _FILE_ROOT:
        raise ValueError(
            f"cannot resolve audio path {path!r}: AGENT_FILE_ROOT is not set"
        )
    p = ensure_path_in_root(path, _FILE_ROOT)
    if not p.is_file():
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
    return any(c in path for c in ("*", "?", "["))


def _expand_audio_glob(pattern: str) -> list[Path]:
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


# Strong refs: the event loop only holds weak refs to tasks, so a running
# detached playback task could be GC'd mid-stream without these.
_PLAYBACK_TASKS: set = set()


async def _stream_files_to_spk(
    session: aiohttp.ClientSession,
    ws: aiohttp.ClientWebSocketResponse,
    paths: list[Path],
    first_pcm: bytes,
    bytes_per_frame: int,
    label: str,
) -> None:
    next_task = None
    try:
        start = time.monotonic()
        frame_idx = 0
        pcm = first_pcm
        for i in range(len(paths)):
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
                # audio-io は奇数バイトを拒否するので末尾だけパディング。
                if len(chunk) % 2 != 0:
                    chunk = chunk + b"\x00"
                await ws.send_bytes(chunk)
                frame_idx += 1
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
                    pcm = b""
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
        log.info(
            "play_audio_file: playback ended early (%s): %s", type(e).__name__, e
        )
    finally:
        if next_task is not None and not next_task.done():
            next_task.cancel()
        try:
            await ws.close()
        finally:
            await session.close()


async def _play_audio_file_impl(paths: list[str] | str) -> str:
    if not _AUDIO_IO_SPK_URL:
        raise RuntimeError(
            "AGENT_AUDIO_IO_SPK_URL is not set — audio playback is disabled"
        )
    # tool スキーマは array だが、LLM が単一文字列を渡してくることがある。
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        raise ValueError("no audio file given; pass at least one WAV path")

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

    first_pcm, _first_dur, _first_rate, _first_ch = await asyncio.to_thread(
        _read_and_convert_wav, resolved[0]
    )
    label = f"{len(resolved)} file(s) [{names}]"
    log.info(
        "play_audio_file: %d file(s) [%s] → %s (streamed one at a time)",
        len(resolved), names, _AUDIO_IO_SPK_URL,
    )

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
    return await _safe_invoke("play_audio_file", _play_audio_file_impl(paths))


async def _stop_audio_impl() -> str:
    if not _AUDIO_IO_STOP_URL:
        raise RuntimeError(
            "AGENT_AUDIO_IO_SPK_URL is not set — audio stop is disabled"
        )
    timeout = aiohttp.ClientTimeout(total=_AUDIO_STOP_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(_AUDIO_IO_STOP_URL) as resp:
            body = await resp.text()
            if resp.status != 200:
                raise RuntimeError(
                    f"audio-io /spk/stop returned {resp.status}: {body[:200]}"
                )
    log.info("stop_audio: flushed via %s", _AUDIO_IO_STOP_URL)
    return "stopped audio playback"


def _build_stop_audio_description() -> str:
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
    return await _safe_invoke("stop_audio", _stop_audio_impl())


_OLLAMA_URL: str = os.environ.get("AGENT_OLLAMA_URL", "http://llm:11434").rstrip("/")
_HEALTH_TIMEOUT_S: float = float(os.environ.get("AGENT_HEALTH_TIMEOUT_S", "2.5"))


# audio-io だけは Windows ネイティブ (compose 外) なので SPK URL から /health
# を導出する。
def _derive_audio_io_health_url(spk_url: str) -> str:
    if not spk_url:
        return ""
    parts = urlsplit(spk_url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, "/health", "", ""))


_AUDIO_IO_HEALTH_URL: str = _derive_audio_io_health_url(_AUDIO_IO_SPK_URL)


def _default_health_targets() -> list[dict]:
    """kind "connect" は HTTP 応答が返れば (ステータス不問) healthy — POST 専用
    エンドポイント (whisper.cpp /inference) は GET だと 4xx になるため。"""
    targets: list[dict] = []
    if _AUDIO_IO_HEALTH_URL:
        targets.append({"name": "音声入出力", "url": _AUDIO_IO_HEALTH_URL, "kind": "http"})
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
    name = target["name"]
    url = target["url"]
    kind = target.get("kind", "http")
    try:
        async with session.get(url) as resp:
            status = resp.status
            ok = True if kind == "connect" else (200 <= status < 300)
            return {"name": name, "ok": ok, "detail": f"HTTP {status}"}
    except asyncio.TimeoutError:
        # aiohttp の ServerTimeoutError も asyncio.TimeoutError のサブクラス。
        return {"name": name, "ok": False, "detail": "timeout"}
    except aiohttp.ClientError as e:
        return {"name": name, "ok": False, "detail": type(e).__name__}
    except Exception as e:  # noqa: BLE001
        return {"name": name, "ok": False, "detail": f"{type(e).__name__}: {e}"}


async def _system_health_impl() -> str:
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
        return f"全部のサービスが応答してないよ。{total}個とも不調。"
    names = "、".join(r["name"] for r in unhealthy)
    return f"{names}が不調だよ。ほかの{healthy_n}個は正常。"


def _build_system_health_description() -> str:
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
    return await _safe_invoke("system_health", _system_health_impl())


# タイマー発火は「ユーザーのターン」が無い非同期イベントなので orchestrator
# 経由で喋れない — audio-io の別 track への直送が agent の唯一の自律出力
# チャネル。track=2 は audio-io 側で runtime.playback_tracks >= 3 が必要 —
# 無いと WS が即 close され、アラームは鳴らず warn ログだけ残る。
_TIMER_TRACK: int = int(os.environ.get("AGENT_TIMER_TRACK", "2"))


def _derive_track_url(spk_url: str, track: int) -> str:
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
_TIMER_MAX_S: float = float(os.environ.get("AGENT_TIMER_MAX_S", "86400"))

_TIMERS: dict[int, dict] = {}
_timer_seq: int = 0
# 強参照保持: registry は発火時に entry を pop するので、それだけだと
# アラーム再生中にタスクが GC されうる (_PLAYBACK_TASKS と同型)。
_TIMER_TASKS: set = set()
_ALARM_PCM_CACHE: bytes | None = None


def _fmt_duration(sec: float) -> str:
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
    rate = _AUDIO_IO_WIRE_RATE
    beep = _gen_tone(880.0, 0.12, rate)
    short_gap = b"\x00\x00" * int(rate * 0.08)
    long_gap = b"\x00\x00" * int(rate * 0.40)
    burst = beep + short_gap + beep + short_gap + beep + short_gap + beep
    pcm = (burst + long_gap) * 3
    if _AUDIO_IO_WIRE_CHANNELS == 2:
        pcm = audioop.tostereo(pcm, _AUDIO_SAMPLE_WIDTH, 1.0, 1.0)
    return pcm


def _alarm_pcm_cached() -> bytes:
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
    _TIMERS.pop(timer_id, None)
    tag = f" ({label})" if label else ""
    log.info("timer #%d fired%s; playing alarm on %s", timer_id, tag, _TIMER_SPK_URL)
    await _play_pcm_on_track(_TIMER_SPK_URL, _alarm_pcm_cached(), f"timer #{timer_id}{tag}")


async def _timer_task(timer_id: int, seconds: float, label: str) -> None:
    try:
        await asyncio.sleep(seconds)
    except asyncio.CancelledError:
        log.info("timer #%d cancelled before firing", timer_id)
        return
    await _timer_fire(timer_id, label)


async def _start_timer_impl(seconds, label: str = "") -> str:
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
    """raise すると LangChain v0.3 系の create_react_agent が AIMessage の
    tool_calls を state に積んだまま ToolMessage を積まないことがあり、以降の
    ターンが INVALID_CHAT_HISTORY で全 chat 死する (検証で確認済)。

    `SandboxError` は retry しても通らない設定/権限系なので `[denied]` で
    区別し、LLM が「諦めて答える」方向に倒れやすくする。他は `[error]`。
    """
    try:
        return await coro
    except SandboxError as e:
        log.info("tool %s denied: %s", name, e)
        return f"[denied] {e}"
    except Exception as e:  # noqa: BLE001
        log.warning("tool %s failed: %s: %s", name, type(e).__name__, e)
        return f"[error] {type(e).__name__}: {e}"


_RESET_MEMORY_HOOK = None


def set_reset_memory_hook(fn) -> None:
    global _RESET_MEMORY_HOOK
    _RESET_MEMORY_HOOK = fn


async def _reset_memory_impl() -> str:
    if _RESET_MEMORY_HOOK is None:
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


def all_tools() -> list:
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
    "stop_audio": os.environ.get("AGENT_TOOL_ACK_STOP_AUDIO", "止めたよ。"),
    "system_health": os.environ.get("AGENT_TOOL_ACK_SYSTEM_HEALTH", "確認するね。"),
    "start_timer": None,
    "check_timers": None,
    "cancel_timer": None,
    "reset_memory": os.environ.get("AGENT_TOOL_ACK_RESET_MEMORY", "記憶をリセットしたよ。"),
}


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
        if not (tool_obj.args or {}):
            print(await tool_obj.ainvoke({}))
            return 0
        _cli_show(tool_obj)
        return 0

    if name == "play_audio_file":
        args = {"paths": argv[1:]}
    else:
        args = _cli_parse(argv[1:], list((tool_obj.args or {}).keys()))
    result = await tool_obj.ainvoke(args)
    print(result)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    sys.exit(asyncio.run(_cli_main(sys.argv[1:])))
