from datetime import date

import pytest

from krta import events
from krta.events import Event, build_calendar, load_schedules, render_calendar
from krta.inputs import TradingCalendar, parse_horizon


@pytest.fixture(scope="module")
def cal():
    return TradingCalendar()


def _names(ev):
    return {(e.date, e.name) for e in ev}


def test_second_thursday_expiry_and_quad(cal):
    ev = build_calendar(date(2026, 11, 10), parse_horizon("2m"), cal, [])
    names = _names(ev)
    assert (date(2026, 11, 12), "선물옵션 만기") in names
    assert (date(2026, 12, 10), "선물옵션 동시만기") in names
    assert (date(2026, 12, 11), "코스피200·코스닥150 정기변경 반영") in names
    rebal = next(e for e in ev if e.name == "코스피200·코스닥150 정기변경 반영")
    assert rebal.estimated is True
    expiry = next(e for e in ev if e.name == "선물옵션 만기")
    assert expiry.estimated is False


def test_expiry_shifts_before_holiday(cal):
    # 2025-10-09(둘째 목요일)는 한글날이고 10/3~10/9가 연휴라 직전 거래일은 10/2.
    assert not cal.is_session(date(2025, 10, 9))
    ev = build_calendar(date(2025, 10, 1), parse_horizon("5d"), cal, [])
    expiry = [e for e in ev if e.name == "선물옵션 만기"]
    assert [e.date for e in expiry] == [date(2025, 10, 2)]
    assert (date(2025, 10, 9), "KRX 휴장") in _names(ev)


def test_msci_marked_estimated(cal):
    ev = build_calendar(date(2026, 11, 10), parse_horizon("1m"), cal, [])
    msci = [e for e in ev if e.name.startswith("MSCI")]
    assert [e.date for e in msci] == [date(2026, 11, 30)]
    assert msci[0].estimated is True
    assert msci[0].note == "발표일·반영일은 MSCI 공지로 확인"


def test_range_is_at_least_20_sessions(cal):
    # 5d여도 최소 20거래일: 2026-11-10 이후 20번째 거래일은 12-08이라 12-09 FOMC는 범위 밖.
    assert cal.next_sessions(date(2026, 11, 10), 20)[-1] == date(2026, 12, 8)
    ev = build_calendar(date(2026, 11, 10), parse_horizon("5d"), cal, [])
    assert all(date(2026, 11, 10) <= e.date <= date(2026, 12, 8) for e in ev)
    assert (date(2026, 11, 26), "한국은행 금통위 통화정책방향 결정") in _names(ev)
    assert not any(e.name.startswith("FOMC") for e in ev)


def test_fomc_and_bok_from_schedule(cal):
    ev = build_calendar(date(2026, 11, 10), parse_horizon("2m"), cal, [])
    fomc = [e for e in ev if e.name == "FOMC 금리 결정"]
    assert [e.date for e in fomc] == [date(2026, 12, 9)]
    assert "한국시간 다음날 새벽" in fomc[0].note
    assert fomc[0].estimated is False


def test_holidays_in_range(cal):
    ev = build_calendar(date(2026, 11, 10), parse_horizon("2m"), cal, [])
    holidays = sorted(e.date for e in ev if e.name == "KRX 휴장")
    assert holidays == [date(2026, 12, 25), date(2026, 12, 31), date(2027, 1, 1)]


def test_missing_year_gives_note_not_event(cal, monkeypatch):
    monkeypatch.setattr(
        events,
        "load_schedules",
        lambda: {"fomc": {"2026": [date(2026, 12, 9)]}, "bok": {"2026": [date(2026, 11, 26)]}},
    )
    ev = build_calendar(date(2026, 11, 10), parse_horizon("2m"), cal, [])
    assert ev.notes == ["2027 FOMC/금통위 일정 데이터 없음"]
    assert not any("미확보" in e.name for e in ev)
    assert "2027 FOMC/금통위 일정 데이터 없음" in render_calendar(ev)


def test_missing_year_only_one_section(cal, monkeypatch):
    monkeypatch.setattr(
        events,
        "load_schedules",
        lambda: {"fomc": {"2026": [], "2027": [date(2027, 1, 27)]}, "bok": {"2026": []}},
    )
    ev = build_calendar(date(2026, 11, 10), parse_horizon("2m"), cal, [])
    assert ev.notes == ["2027 금통위 일정 데이터 없음"]


def test_no_note_when_years_covered(cal):
    ev = build_calendar(date(2026, 11, 10), parse_horizon("1m"), cal, [])
    assert ev.notes == []


def test_earnings_hint_arg(cal):
    ev = build_calendar(date(2026, 11, 10), parse_horizon("2m"), cal, [date(2026, 11, 20)])
    earn = next(e for e in ev if e.name == "실적 발표 예상")
    assert earn.date == date(2026, 11, 20) and earn.estimated is True
    assert not any("배당" in e.name for e in ev)


def test_schedules_file_has_sources_and_valid_dates():
    s = load_schedules()
    assert set(s) == {"fomc", "bok"}
    for section in s.values():
        assert "2026" in section
        for year, days in section.items():
            assert all(d.year == int(year) for d in days)
            assert days == sorted(days)
    assert date(2026, 12, 9) in s["fomc"]["2026"]
    assert len(s["fomc"]["2026"]) == 8 and len(s["bok"]["2026"]) == 8
    raw = events._read_schedules_toml()
    for name in ("fomc", "bok"):
        assert raw[name]["source"].startswith("https://")


def test_render_calendar_sorted_with_estimated_mark():
    ev = [
        Event(date(2026, 12, 11), "코스피200·코스닥150 정기변경 반영", True),
        Event(date(2026, 11, 12), "선물옵션 만기", False),
    ]
    md = render_calendar(ev)
    lines = md.splitlines()
    assert lines[0].startswith("| 날짜 |")
    body = [ln for ln in lines if ln.startswith("| 2026")]
    assert body[0].startswith("| 2026-11-12 | 선물옵션 만기 |")
    assert "코스피200·코스닥150 정기변경 반영 (추정)" in body[1]
    assert "(추정)" not in body[0]


def test_render_calendar_empty():
    assert "일정 없음" in render_calendar([])
