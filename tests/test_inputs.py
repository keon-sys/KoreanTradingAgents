from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from krta.inputs import (
    Horizon,
    InputError,
    OutOfScopeError,
    ResolvedDate,
    TradingCalendar,
    check_listed,
    normalize_code,
    parse_horizon,
    resolve_date,
    resolve_market,
)

KST = ZoneInfo("Asia/Seoul")


def test_parse_horizon_buckets():
    assert parse_horizon("5d") == Horizon("5d", 5, "단기")
    assert parse_horizon("4w").trading_days == 20 and parse_horizon("4w").bucket == "단기"
    assert parse_horizon("6m") == Horizon("6m", 126, "중기")
    assert parse_horizon("2y") == Horizon("2y", 504, "장기")
    for bad in ["0d", "5", "5h", "-1d"]:
        with pytest.raises(ValueError):
            parse_horizon(bad)


def test_parse_horizon_bucket_boundaries():
    assert parse_horizon("20d").bucket == "단기"
    assert parse_horizon("21d").bucket == "중기"
    assert parse_horizon("126d").bucket == "중기"
    assert parse_horizon("127d").bucket == "장기"
    assert parse_horizon("1y").trading_days == 252
    for bad in ["", "1000d", "5D", "d", "5dd", " 5 d", "00d"]:
        with pytest.raises(ValueError):
            parse_horizon(bad)


def test_default_date_before_and_after_18():
    cal = TradingCalendar()
    assert resolve_date(None, datetime(2026, 11, 9, 17, 59, tzinfo=KST), cal).date == date(2026, 11, 6)
    assert resolve_date(None, datetime(2026, 11, 9, 18, 0, tzinfo=KST), cal).date == date(2026, 11, 9)


def test_default_date_is_confirmed_and_has_no_note():
    r = resolve_date(None, datetime(2026, 11, 9, 10, tzinfo=KST), TradingCalendar())
    assert r == ResolvedDate(date(2026, 11, 6), False, None)


def test_default_date_on_holiday_uses_previous_session():
    # 2026-11-08 is a Sunday; even after 18:00 the default is the prior session.
    r = resolve_date(None, datetime(2026, 11, 8, 20, tzinfo=KST), TradingCalendar())
    assert r.date == date(2026, 11, 6) and r.unconfirmed is False


def test_now_is_converted_to_kst():
    # 09:00 UTC == 18:00 KST on the same day.
    r = resolve_date(None, datetime(2026, 11, 9, 9, 0, tzinfo=timezone.utc), TradingCalendar())
    assert r.date == date(2026, 11, 9)


def test_naive_now_rejected():
    with pytest.raises(ValueError):
        resolve_date(None, datetime(2026, 11, 9, 20), TradingCalendar())


def test_explicit_holiday_moves_back_with_note():
    r = resolve_date(date(2026, 10, 3), datetime(2026, 11, 9, 20, tzinfo=KST), TradingCalendar())  # 개천절(토)
    assert r.date == date(2026, 10, 2) and "휴장일" in r.note
    assert r.note == "요청일 2026-10-03은 휴장일이라 직전 거래일 2026-10-02로 분석"
    assert r.unconfirmed is False


def test_explicit_past_session_kept():
    r = resolve_date(date(2026, 11, 6), datetime(2026, 11, 9, 10, tzinfo=KST), TradingCalendar())
    assert r == ResolvedDate(date(2026, 11, 6), False, None)


def test_explicit_today_intraday_is_unconfirmed():
    r = resolve_date(date(2026, 11, 9), datetime(2026, 11, 9, 10, tzinfo=KST), TradingCalendar())
    assert r.unconfirmed is True
    assert r.date == date(2026, 11, 9)


def test_explicit_today_after_close_is_confirmed():
    r = resolve_date(date(2026, 11, 9), datetime(2026, 11, 9, 18, 0, tzinfo=KST), TradingCalendar())
    assert r == ResolvedDate(date(2026, 11, 9), False, None)


def test_future_date_rejected():
    with pytest.raises(ValueError):
        resolve_date(date(2026, 11, 20), datetime(2026, 11, 9, 20, tzinfo=KST), TradingCalendar())


def test_resolved_date_is_frozen():
    r = ResolvedDate(date(2026, 11, 6), False, None)
    with pytest.raises(AttributeError):
        r.date = date(2026, 11, 9)  # type: ignore[misc]


def test_calendar_methods():
    cal = TradingCalendar()
    assert cal.is_session(date(2026, 11, 9)) is True
    assert cal.is_session(date(2026, 11, 8)) is False
    assert cal.is_session(date(2026, 10, 5)) is False  # 개천절 대체공휴일
    assert cal.previous_session(date(2026, 11, 9)) == date(2026, 11, 6)
    assert cal.previous_session(date(2026, 10, 3)) == date(2026, 10, 2)
    assert cal.next_sessions(date(2026, 11, 6), 2) == [date(2026, 11, 9), date(2026, 11, 10)]
    assert cal.next_sessions(date(2026, 10, 2), 1) == [date(2026, 10, 6)]
    assert cal.next_sessions(date(2026, 11, 6), 0) == []
    assert cal.sessions_between(date(2026, 11, 6), date(2026, 11, 10)) == [
        date(2026, 11, 6), date(2026, 11, 9), date(2026, 11, 10)
    ]
    assert cal.sessions_between(date(2026, 11, 7), date(2026, 11, 8)) == []
    with pytest.raises(ValueError):
        cal.next_sessions(date(2026, 11, 6), -1)
    with pytest.raises(ValueError):
        cal.sessions_between(date(2026, 11, 10), date(2026, 11, 6))


def test_calendar_out_of_range_raises_value_error():
    cal = TradingCalendar()
    with pytest.raises(ValueError):
        cal.is_session(date(1900, 1, 2))


def test_codes():
    assert normalize_code("005930.ks") == ("005930", "KS")
    assert resolve_market("005935", None, {"005930", "005935"}, set()) == "KS"  # 우선주
    with pytest.raises(OutOfScopeError):
        resolve_market("069500", None, {"005930"}, {"035720"})
    with pytest.raises(InputError):
        resolve_market("035720", "KS", {"005930"}, {"035720"})
    with pytest.raises(InputError):
        resolve_market("035720", None, {"035720"}, {"035720"})


def test_codes_more():
    assert normalize_code("005930") == ("005930", None)
    assert normalize_code("035720.KQ") == ("035720", "KQ")
    for bad in ["5930", "0059300", "005930.KX", "AAPL", "005930.", "005930.KS.KS", ""]:
        with pytest.raises(InputError):
            normalize_code(bad)
    assert resolve_market("035720", "KQ", {"005930"}, {"035720"}) == "KQ"
    assert resolve_market("035720", None, {"005930"}, {"035720"}) == "KQ"
    with pytest.raises(OutOfScopeError, match="ETF"):
        resolve_market("069500", "KS", {"005930"}, {"035720"})


def test_out_of_scope_message():
    with pytest.raises(OutOfScopeError) as ei:
        resolve_market("069500", None, set(), set())
    assert str(ei.value) == "KRX 주식 기본정보에 없는 코드(ETF·ETN·리츠 등은 범위 밖)"


def test_before_listing_rejected():
    with pytest.raises(InputError):
        check_listed(date(2020, 1, 2), date(2019, 12, 30))


def test_listing_day_and_after_accepted():
    assert check_listed(date(2020, 1, 2), date(2020, 1, 2)) is None
    assert check_listed(date(2020, 1, 2), date(2026, 11, 9)) is None
