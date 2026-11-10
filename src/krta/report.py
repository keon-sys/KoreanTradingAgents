"""Run output: ten markdown reports under `reports/` plus `decision.json`.

`final_trade_decision.md` is rendered from the PM's structured `Decision`."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from krta.agents.analysts import ANALYSTS
from krta.agents.schemas import RATING_KO, Decision
from krta.collect import UNCONFIRMED_BANNER, DataBundle
from krta.inputs import KST, TradingCalendar, resolve_date
from krta.sections import md_table, missing_line

if TYPE_CHECKING:
    from krta.graph import RunState

DISCLAIMER = "매수 비중·손절가 등은 보유 현황을 모르고 쓴 일반 제안입니다."
PAST_DATE_NOTE = "과거 날짜 분석: 재무제표 정정 전 원본·당시 상장 목록은 재현하지 않음"
UNCONFIRMED_WARNING = f"> {UNCONFIRMED_BANNER}: 이 판단은 장 마감 확정 전 데이터 기준이며 메모리 채점에서 빠진다."
NO_PRICE = "근거 없음"
NONE_ITEM = "- 없음"

EVENT_CALENDAR = "event_calendar.md"
INVESTMENT_PLAN = "investment_plan.md"
TRADER_PLAN = "trader_investment_plan.md"
FINAL_DECISION = "final_trade_decision.md"
DECISION_JSON = "decision.json"
REPORT_FILES: tuple[str, ...] = (
    EVENT_CALENDAR,
    *(spec.report_file for spec in ANALYSTS.values()),
    INVESTMENT_PLAN,
    TRADER_PLAN,
    FINAL_DECISION,
)


def run_dir(data_dir: Path, b: DataBundle) -> Path:
    t = b.target
    return Path(data_dir) / "logs-kr" / f"{t.code}.{t.market}" / b.date.isoformat() / b.horizon.raw


def default_analysis_date(now: datetime | None = None, cal: TradingCalendar | None = None) -> date:
    """The calendar default date of a dateless run: today after 18:00 KST on a session,
    else the previous session (before the KRX publication check)."""
    return resolve_date(None, now or datetime.now(KST), cal or TradingCalendar()).date


def _won(v: int | None) -> str:
    return NO_PRICE if v is None else f"{v:,}원"


def _bullets(items: list[str]) -> str:
    items = [i.strip() for i in items if i.strip()]
    return "\n".join(f"- {i}" for i in items) if items else NONE_ITEM


def _joined(items: list[str]) -> str:
    items = [i.strip() for i in items if i.strip()]
    return "<br>".join(f"• {i}" for i in items) if items else "없음"


def render_final_decision(b: DataBundle, d: Decision, *, default_date: date) -> str:
    h = b.horizon
    out: list[str] = []
    if b.unconfirmed:
        out += [UNCONFIRMED_WARNING, ""]
    out += [
        f"# 최종 판단: {b.target.name} ({b.target.code}.{b.target.market}) {b.date.isoformat()}",
        "",
        "## 결론",
        f"- rating: {RATING_KO[d.rating]} ({d.rating.value})",
        f"- 목표가: {_won(d.target_price)}",
        f"- 손절가: {_won(d.stop_loss)}",
        f"- 기간: {h.raw} ({h.trading_days}거래일, {h.bucket})",
        "- 재점검 조건:",
        *(f"  - {t}" for t in d.review_triggers if t.strip()),
    ]
    if not any(t.strip() for t in d.review_triggers):
        out.append("  - 없음")
    out += ["", d.summary.strip()] if d.summary.strip() else []
    notes = [n for n in (b.date_note, PAST_DATE_NOTE if b.date < default_date else None) if n]
    if notes:
        out += ["", *(f"> {n}" for n in notes)]
    table = md_table(
        [
            {"구분": "가격 상황", "내용": d.price_context.strip() or "없음"},
            {"구분": "강세 근거", "내용": _joined(d.bull_points)},
            {"구분": "약세 근거", "내용": _joined(d.bear_points)},
        ],
        ["구분", "내용"],
    )
    out += [
        "",
        "## 핵심 근거",
        table,
        "",
        "## 다가오는 이벤트",
        _bullets(d.upcoming_events),
        "",
        "## 빠진 데이터",
        _bullets(d.missing_data),
        "",
        "## 치명적 공시 대응",
        d.critical_disclosures_addressed.strip() or "해당 없음",
        "",
        "---",
        DISCLAIMER,
    ]
    return "\n".join(out) + "\n"


def decision_record(b: DataBundle, d: Decision) -> dict:
    return {
        "code": b.target.code,
        "market": b.target.market,
        "name": b.target.name,
        "date": b.date.isoformat(),
        "horizon": b.horizon.raw,
        "trading_days": b.horizon.trading_days,
        "bucket": b.horizon.bucket,
        "unconfirmed": b.unconfirmed,
        "date_note": b.date_note,
        "rating_ko": RATING_KO[d.rating],
        **d.model_dump(mode="json"),
    }


def write_atomic(path: Path, text: str) -> None:
    """Write-then-rename so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_reports(dir: Path, state: RunState, *, default_date: date | None = None) -> list[Path]:
    """Write the ten reports and `decision.json` into `dir`; return the written paths.

    `default_date` is the date a dateless run would analyse right now; an earlier analysis
    date gets the past-date limitation note. Computed from the current KST time if omitted."""
    d = state["decision"]
    if d is None:
        raise ValueError("최종 판단(decision)이 없어 리포트를 쓸 수 없음")
    b = state["bundle"]
    if default_date is None:
        default_date = default_analysis_date()

    texts = {
        EVENT_CALENDAR: b.calendar_md,
        **{
            spec.report_file: state["reports"].get(key) or missing_line(f"{spec.title_ko} 리포트", "생성되지 않음")
            for key, spec in ANALYSTS.items()
        },
        INVESTMENT_PLAN: state["investment_plan"],
        TRADER_PLAN: state["trader_plan"],
        FINAL_DECISION: render_final_decision(b, d, default_date=default_date),
    }
    dir = Path(dir)
    written = []
    for name in REPORT_FILES:
        path = dir / "reports" / name
        write_atomic(path, texts[name])
        written.append(path)
    path = dir / DECISION_JSON
    write_atomic(path, json.dumps(decision_record(b, d), ensure_ascii=False, indent=2) + "\n")
    written.append(path)
    return written
