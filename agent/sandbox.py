"""Sandbox helpers for tool 入力検証 (issue #19). Pure functions — no I/O,
no env reads. shell allowlist はコマンド名のみで照合 (issue #19 オープン
項目 #2 で確定 — 引数 regex の粒度は導入しない)。
"""

from __future__ import annotations

import shlex
from pathlib import Path


class SandboxError(ValueError):
    """Base class. ValueError 派生だと LangChain の tool error ハンドラが
    素直に拾って LLM に message を渡す。"""


class CommandNotAllowed(SandboxError):
    """run_shell の allowlist 違反。"""


class PathOutsideRoot(SandboxError):
    """read_file の root 配下脱出 (symlink 経由を含む)。"""


def parse_shell_command(command: str) -> list[str]:
    """shlex.split で argv に分解 (`shell=True` 不使用 = パイプ・リダイレクト
    が構造的に不可能)。空コマンドは `create_subprocess_exec` の OSError より
    手前でメッセージ性のある例外にする。"""
    parts = shlex.split(command)
    if not parts:
        raise CommandNotAllowed("empty command")
    return parts


def ensure_command_allowed(command: str, allowlist: set[str]) -> list[str]:
    """argv に分解し、`argv[0]` が allowlist にあるか完全一致で確認する
    (prefix マッチだと「rm」で「rmdir」も通る等の罠が生じる)。"""
    argv = parse_shell_command(command)
    cmd_name = argv[0]
    if cmd_name not in allowlist:
        raise CommandNotAllowed(
            f"command not allowed: {cmd_name!r} "
            f"(allowed: {sorted(allowlist)})"
        )
    return argv


def ensure_path_in_root(path: str, root: str) -> Path:
    """`path` を `root` 相対として解釈し、resolve 後 (= symlink 追跡後) も
    `root` 配下に収まることを確認した上で絶対 Path を返す。"""
    if not path:
        raise PathOutsideRoot("empty path")
    root_resolved = Path(root).resolve()
    candidate = (root_resolved / path).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as e:
        raise PathOutsideRoot(
            f"path outside file root: {path!r} (root={str(root_resolved)!r})"
        ) from e
    return candidate
