"""Sandbox helpers for tool 入力検証 (issue #19).

Pure functions — no I/O, no env reads, no async. tools.py が env から
allowlist / root を読み込んでこれらに渡すという責務分離。テストが書きやすく、
ヘルパ単体の不変条件を pin できる。

設計判断:
* allowlist 違反 / root 脱出は `ValueError` サブクラスで raise する。
  `create_react_agent` のデフォルト error handling (handle_tool_errors=True)
  は例外メッセージを ToolMessage として LLM に返すので、LLM が「別のコマンドで
  retry する」「諦めて謝る」を recursion_limit の範囲で判断できる。
* shell allowlist は **コマンド名のみ** で照合 (issue #19 オープン項目 #2
  で確定)。引数 regex まで縛る粒度は導入しない — 危険 API (rm/curl/sh など)
  をそもそも入れない運用で十分。
* path は `Path.resolve(strict=False)` で symlink を辿った後の絶対パスで
  root と比較。`relative_to` の例外を `PathOutsideRoot` に再 raise する。
  strict=False は「存在しないパス」も resolve できるよう (存在判定は
  別段で `.is_file()` する)。
"""

from __future__ import annotations

import shlex
from pathlib import Path


class SandboxError(ValueError):
    """Base class. ValueError 派生にしておくと LangChain の tool error
    ハンドラが素直に拾って LLM に message を渡す。"""


class CommandNotAllowed(SandboxError):
    """run_shell の allowlist 違反。"""


class PathOutsideRoot(SandboxError):
    """read_file の root 配下脱出 (symlink 経由を含む)。"""


def parse_shell_command(command: str) -> list[str]:
    """shlex.split で argv に分解。`shell=True` を使わない (=パイプ・
    リダイレクト・複合コマンドが構造的に不可能) ための前処理。

    空コマンドは `CommandNotAllowed` で弾く — `create_subprocess_exec`
    に空 argv を渡すと OSError になるので、その前にメッセージ性のある
    例外で raise した方が LLM が retry 時に状況を理解しやすい。
    """
    parts = shlex.split(command)
    if not parts:
        raise CommandNotAllowed("empty command")
    return parts


def ensure_command_allowed(command: str, allowlist: set[str]) -> list[str]:
    """argv に分解し、`argv[0]` が allowlist にあるか確認する。

    OK なら argv (list[str]) を返す。NG なら `CommandNotAllowed`。
    `allowlist` は呼び出し側で env から読み込んで set 化する想定。
    完全一致のみ — prefix マッチや regex はサポートしない (粒度を下げる
    と「rm」も「rmdir」も通る等の罠が生じる)。
    """
    argv = parse_shell_command(command)
    cmd_name = argv[0]
    if cmd_name not in allowlist:
        raise CommandNotAllowed(
            f"command not allowed: {cmd_name!r} "
            f"(allowed: {sorted(allowlist)})"
        )
    return argv


def ensure_path_in_root(path: str, root: str) -> Path:
    """`path` を `root` からの相対パスとして解釈し、resolve 後も `root`
    配下に収まることを確認した上で絶対 Path を返す。

    LLM には「root 配下の相対パスを渡す」前提で tool を提供しているが、
    `..` や絶対パスや symlink でその外を覗こうとされても弾けるよう、
    resolve() 結果と root.resolve() を比較する。
    """
    if not path:
        raise PathOutsideRoot("empty path")
    root_resolved = Path(root).resolve()
    # path が絶対パスでも、root を抜けようとすれば下の relative_to で弾ける。
    # 相対パスは root に対して join した上で resolve する。
    candidate = (root_resolved / path).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as e:
        raise PathOutsideRoot(
            f"path outside file root: {path!r} (root={str(root_resolved)!r})"
        ) from e
    return candidate
