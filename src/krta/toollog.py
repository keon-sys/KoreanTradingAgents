"""Append-only `message_tool.log`: every tool input, raw tool output and agent message, redacted."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from krta.redact import Redactor

KST = ZoneInfo("Asia/Seoul")
KINDS = ("request", "data", "agent")


class ToolLog:
    def __init__(self, path: Path, redactor: Redactor):
        self.path = Path(path)
        self._redactor = redactor
        self._lock = threading.Lock()

    def write(self, kind: Literal["request", "data", "agent"], source: str, payload: str) -> None:
        if kind not in KINDS:
            raise ValueError(f"알 수 없는 로그 종류: {kind}")
        stamp = datetime.now(KST).isoformat(timespec="seconds")
        # Not truncated: the log is the raw material reports are checked against.
        entry = self._redactor.redact(f"=== {stamp} [{kind}] {source} ===\n{payload}\n")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(entry)
