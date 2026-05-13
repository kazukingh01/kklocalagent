"""LangChain tool 定義 + ack phrase マッピング (issue #19).

各 tool は `@tool` デコレータで LangChain Tool に変換され、agent_node に
`create_react_agent` 経由でバインドされる。LLM (Gemma 4) は description を
読んで「いつどの tool を呼ぶか」を確率的に判断する — コード側はルーティング
するだけで判断には関与しない (詳細は issue #19 設計コメント参照)。

ack phrase は tool 実行中の無音をカバーする目的の合成 chunk。`TOOL_ACK_PHRASES`
で tool 名 → 文字列にマップし、None なら ack 挿入をスキップする (即応 tool 向け)。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

from langchain_core.tools import tool

# 日本標準時。voice agent は日本語前提で動くので JST 固定が自然。海外向けに
# 切り出すときは TZ env を見るように変える。
JST = timezone(timedelta(hours=9))


@tool
def get_current_datetime() -> str:
    """現在の日時を JST で ISO 8601 形式 (例: 2026-05-13T15:30:00+09:00) で返す。

    日付・時刻・曜日・今が何月か・現在年などを聞かれたときに呼ぶ。
    """
    return datetime.now(JST).isoformat(timespec="seconds")


def all_tools() -> list:
    """登録 tool の一覧。`create_react_agent` に渡す。

    issue #19 の段階的な追加に伴い後段の PR で `run_shell` / `read_file` /
    `web_search` / `list_drive_files` / `read_drive_file` が積まれる予定。
    """
    return [get_current_datetime]


# tool name → ack phrase。None なら ack 挿入なし。
#
# 即応 tool (date / shell / file) は None でよく、遅い tool (web 検索や Drive
# アクセス) だけ ack を入れて voice agent の無音を埋める。文言は環境変数で
# 上書き可能にしておくと運用しやすい — 各 tool 単位の ack 環境変数は対応する
# tool が wire された PR で追加する。
TOOL_ACK_PHRASES: dict[str, str | None] = {
    "get_current_datetime": None,
}
