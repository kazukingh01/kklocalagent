from __future__ import annotations

import shlex
from pathlib import Path


class SandboxError(ValueError):
    """ValueError 派生だと LangChain の tool error ハンドラが素直に拾って
    LLM に message を渡す。"""


class CommandNotAllowed(SandboxError):
    pass


class PathOutsideRoot(SandboxError):
    pass


def parse_shell_command(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise CommandNotAllowed("empty command")
    return parts


def ensure_command_allowed(command: str, allowlist: set[str]) -> list[str]:
    argv = parse_shell_command(command)
    cmd_name = argv[0]
    if cmd_name not in allowlist:
        raise CommandNotAllowed(
            f"command not allowed: {cmd_name!r} "
            f"(allowed: {sorted(allowlist)})"
        )
    return argv


def ensure_path_in_root(path: str, root: str) -> Path:
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
