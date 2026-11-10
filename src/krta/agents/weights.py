"""Horizon weighting table (spec §8), rendered for every agent prompt."""

from __future__ import annotations

# (구간, 주 근거, 보조 근거) — spec §8 verbatim.
WEIGHTS: tuple[tuple[str, str, str], ...] = (
    ("단기", "수급, 테마·재료, 차트·거래량, 시장", "공시(악재 필터), 밸류(참고)"),
    ("중기", "6명 고르게", "—"),
    ("장기", "펀더멘털·밸류, 공시", "수급·차트(진입 타이밍)"),
)
MARK = "◀ 이번 구간"


def weights_table(bucket: str) -> str:
    """Markdown table of spec §8 with the row for `bucket` bolded and marked."""
    if bucket not in {row[0] for row in WEIGHTS}:
        raise ValueError(f"알 수 없는 기간 구간: {bucket!r}")
    lines = ["| 구간 | 주 근거 | 보조 근거 |", "|---|---|---|"]
    for name, main, aux in WEIGHTS:
        if name == bucket:
            lines.append(f"| **{name}** {MARK} | **{main}** | **{aux}** |")
        else:
            lines.append(f"| {name} | {main} | {aux} |")
    return "\n".join(lines)
