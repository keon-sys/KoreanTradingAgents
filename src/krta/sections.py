"""Report sections fed to the analysts, with explicit markers for missing data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Section:
    name: str
    text: str
    missing: list[str] = field(default_factory=list)


def missing_line(what: str, why: str) -> str:
    return f"데이터 없음: {what} ({why})"


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\r\n", " ").replace("\n", " ")


def md_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    lines += ["| " + " | ".join(_cell(row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join(lines)
