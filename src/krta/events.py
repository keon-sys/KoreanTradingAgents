"""Event calendar computed by code (not LLM): expiries, index reviews, FOMC/BOK, holidays."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from importlib import resources

from krta.inputs import Horizon, TradingCalendar

MIN_SESSIONS = 20
QUAD_MONTHS = (3, 6, 9, 12)
KOSPI200_REVIEW_MONTHS = (6, 12)
MSCI_REVIEW_MONTHS = (2, 5, 8, 11)
_SCHEDULE_LABELS = {"fomc": "FOMC", "bok": "금통위"}


@dataclass
class Event:
    date: date
    name: str
    estimated: bool
    note: str = ""


class EventList(list[Event]):
    """build_calendar 결과. 이벤트가 아닌 캘린더 단위 안내 문구는 `notes`에 둔다."""

    def __init__(self, events=(), notes: list[str] | None = None) -> None:
        super().__init__(events)
        self.notes: list[str] = list(notes or [])


# ---------------------------------------------------------------- schedules.toml


@lru_cache(maxsize=1)
def _read_schedules_toml() -> dict:
    text = resources.files("krta.data").joinpath("schedules.toml").read_text(encoding="utf-8")
    return tomllib.loads(text)


def load_schedules() -> dict[str, dict[str, list[date]]]:
    """{"fomc"|"bok": {"<연도>": [date, ...]}} — `source` 등 연도가 아닌 키는 뺀다."""
    out: dict[str, dict[str, list[date]]] = {}
    for section in _SCHEDULE_LABELS:
        table = _read_schedules_toml().get(section, {})
        out[section] = {
            year: sorted(date.fromisoformat(s) for s in days)
            for year, days in table.items()
            if year.isdigit()
        }
    return out


# ---------------------------------------------------------------- rules


def _second_thursday(year: int, month: int) -> date:
    first = date(year, month, 1)
    first_thu = first + timedelta(days=(3 - first.weekday()) % 7)
    return first_thu + timedelta(days=7)


def _months(a: date, b: date) -> list[tuple[int, int]]:
    out = []
    y, m = a.year, a.month
    while (y, m) <= (b.year, b.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _first_of_next_month(year: int, month: int) -> date:
    return date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)


def _expiry(year: int, month: int, cal: TradingCalendar) -> date:
    d = _second_thursday(year, month)
    return d if cal.is_session(d) else cal.previous_session(d)


def build_calendar(
    start: date,
    horizon: Horizon,
    cal: TradingCalendar,
    earnings_hint: list[date],
) -> EventList:
    """start ~ start 이후 max(horizon.trading_days, 20)번째 거래일까지의 이벤트.

    earnings_hint는 호출자가 고른 날짜를 범위와 상관없이 그대로 싣는다.
    배당 공시는 접수일이 기준일이 아니므로 이벤트로 싣지 않는다(호출자가 notes에 안내).
    """
    end = cal.next_sessions(start, max(horizon.trading_days, MIN_SESSIONS))[-1]

    def in_range(d: date) -> bool:
        return start <= d <= end

    events: list[Event] = []
    for y, m in _months(start, end):
        exp = _expiry(y, m, cal)
        if in_range(exp):
            name = "선물옵션 동시만기" if m in QUAD_MONTHS else "선물옵션 만기"
            events.append(Event(exp, name, False))
        if m in KOSPI200_REVIEW_MONTHS:
            rebal = cal.next_sessions(exp, 1)[0]
            if in_range(rebal):
                events.append(Event(rebal, "코스피200·코스닥150 정기변경 반영", True))
        if m in MSCI_REVIEW_MONTHS:
            last = cal.previous_session(_first_of_next_month(y, m))
            if in_range(last):
                events.append(
                    Event(last, "MSCI 분기 리뷰 반영", True, "발표일·반영일은 MSCI 공지로 확인")
                )

    schedules = load_schedules()
    notes: list[str] = []
    for year in range(start.year, end.year + 1):
        missing = [
            label
            for key, label in _SCHEDULE_LABELS.items()
            if str(year) not in schedules.get(key, {})
        ]
        if missing:
            notes.append(f"{year} {'/'.join(missing)} 일정 데이터 없음")
    for key, name, note in (
        ("fomc", "FOMC 금리 결정", "미국 날짜 기준 성명 발표일, 한국시간 다음날 새벽"),
        ("bok", "한국은행 금통위 통화정책방향 결정", ""),
    ):
        for days in schedules.get(key, {}).values():
            events.extend(Event(d, name, False, note) for d in days if in_range(d))

    events.extend(
        Event(d, "실적 발표 예상", True, "작년 같은 분기 잠정실적 공시일 기준")
        for d in earnings_hint
    )

    sessions = set(cal.sessions_between(start, end))
    d = start
    while d <= end:
        if d.weekday() < 5 and d not in sessions:
            events.append(Event(d, "KRX 휴장", False))
        d += timedelta(days=1)

    events.sort(key=lambda e: e.date)
    return EventList(events, notes)


# ---------------------------------------------------------------- render


def _cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def render_calendar(events: list[Event]) -> str:
    """날짜순 마크다운 표. 추정 이벤트는 이름 뒤에 "(추정)"."""
    lines = []
    if events:
        lines += ["| 날짜 | 이벤트 | 비고 |", "| --- | --- | --- |"]
        for e in sorted(events, key=lambda e: e.date):
            name = f"{e.name} (추정)" if e.estimated else e.name
            lines.append(f"| {e.date.isoformat()} | {_cell(name)} | {_cell(e.note)} |")
    else:
        lines.append("기간 안 일정 없음")
    notes = getattr(events, "notes", [])
    if notes:
        lines.append("")
        lines += [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"
