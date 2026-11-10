import json
from datetime import date

import pandas as pd
import pytest

from krta.agents.analysts import ANALYSTS
from krta.agents.schemas import Decision, Rating
from krta.collect import DataBundle, Target
from krta.inputs import TradingCalendar, parse_horizon
from krta.memory import MAX_ATTEMPTS, Memory, memory_path
from krta.net import FetchError
from krta.sections import Section

CAL = TradingCalendar()
D0 = date(2026, 11, 2)  # Mon; 5 sessions after: 11-03, 04, 05, 06, 09
SESSIONS = [D0, *CAL.next_sessions(D0, 9)]


def bundle(code="005930", market="KS", day=D0, horizon="5d", unconfirmed=False) -> DataBundle:
    return DataBundle(
        target=Target(code, market, "종목", "00000000", date(2000, 1, 4)),
        date=day,
        window_end=CAL.next_sessions(day, parse_horizon(horizon).trading_days)[-1],
        horizon=parse_horizon(horizon),
        unconfirmed=unconfirmed,
        date_note=None,
        peers=[],
        calendar_md="",
        sections={key: Section(key, "") for key in ANALYSTS},
    )


def decision(rating=Rating.BUY, target=110, stop=95) -> Decision:
    return Decision(
        rating=rating, target_price=target, stop_loss=stop, review_triggers=[], price_context="",
        bull_points=[], bear_points=[], upcoming_events=[], missing_data=[],
        critical_disclosures_addressed="해당 없음", summary="",
    )


def bars(closes, highs=None, lows=None) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in SESSIONS[: len(closes)]], name="date")
    highs = highs or closes
    lows = lows or closes
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": [1] * len(closes)},
        index=idx,
    )


class FakeHistory:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        self.calls: list[tuple[str, date, int]] = []

    def __call__(self, symbol: str, end: date, days: int) -> pd.DataFrame:
        self.calls.append((symbol, end, days))
        if symbol not in self.frames:
            raise FetchError("yahoo", f"{symbol} 없음")
        df = self.frames[symbol]
        return df[df.index <= pd.Timestamp(end)].tail(days)


STOCK = bars([100, 101, 104, 112, 103, 108, 120],
             highs=[100, 102, 105, 113, 104, 109, 121], lows=[100, 99, 101, 108, 96, 105, 118])
INDEX = bars([2000, 2010, 2020, 2030, 2040, 2040, 2100])


def history():
    return FakeHistory({"005930.KS": STOCK, "^KS11": INDEX, "035720.KQ": STOCK, "^KQ11": INDEX})


def lines(mem: Memory) -> list[dict]:
    return [json.loads(x) for x in mem.path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_memory_path(tmp_path):
    assert memory_path(tmp_path) == tmp_path / "memory-kr" / "decisions.jsonl"


def test_record_fields(tmp_path):
    mem = Memory(memory_path(tmp_path))
    mem.record(bundle(), decision())
    (row,) = lines(mem)
    assert row == {
        "code": "005930", "market": "KS", "date": "2026-11-02", "horizon": "5d", "trading_days": 5,
        "rating": "BUY", "target_price": 110, "stop_loss": 95, "unconfirmed": False, "settled": None,
    }


def test_record_replaces_same_run_key(tmp_path):
    mem = Memory(memory_path(tmp_path))
    mem.record(bundle(), decision(rating=Rating.BUY))
    mem.record(bundle(horizon="1m"), decision(rating=Rating.HOLD))
    mem.record(bundle(), decision(rating=Rating.SELL))
    rows = lines(mem)
    assert [(r["horizon"], r["rating"]) for r in rows] == [("1m", "HOLD"), ("5d", "SELL")]


def test_unconfirmed_not_recorded_for_settlement(tmp_path):
    mem = Memory(memory_path(tmp_path))
    mem.record(bundle(unconfirmed=True), decision())
    h = history()
    assert mem.settle(date(2026, 11, 20), h, CAL) == 0
    (row,) = lines(mem)
    assert row["unconfirmed"] is True and row["settled"] is None
    assert h.calls == []
    assert mem.lessons("005930", date(2026, 12, 1)) == ""


def test_settle_after_window_only(tmp_path):
    mem = Memory(memory_path(tmp_path))
    mem.record(bundle(), decision())
    h = history()

    assert mem.settle(date(2026, 11, 7), h, CAL) == 0  # 4 sessions passed
    assert mem.settle(date(2026, 11, 9), h, CAL) == 0  # last window session is today: not closed
    assert h.calls == []
    assert mem.settle(date(2026, 11, 10), h, CAL) == 1
    assert mem.settle(date(2026, 11, 11), h, CAL) == 0  # already settled

    (row,) = lines(mem)
    s = row["settled"]
    # window close: 11-09 (index 5): stock 100 -> 108, index 2000 -> 2040
    assert s["return"] == pytest.approx(0.08)
    assert s["bench_return"] == pytest.approx(0.02)
    assert s["excess"] == pytest.approx(0.06)
    assert s["hit_target"] is True  # high 113 >= 110 on 11-05
    assert s["hit_stop"] is False  # low 96 > 95
    assert s["settled_on"] == "2026-11-10"
    assert s["window_end"] == "2026-11-09"
    assert {c[0] for c in h.calls} == {"005930.KS", "^KS11"}
    assert all(c[1] == date(2026, 11, 9) for c in h.calls)


def test_settle_kosdaq_uses_kosdaq_index(tmp_path):
    mem = Memory(memory_path(tmp_path))
    mem.record(bundle(code="035720", market="KQ"), decision())
    h = history()
    assert mem.settle(date(2026, 11, 10), h, CAL) == 1
    assert {c[0] for c in h.calls} == {"035720.KQ", "^KQ11"}


def test_settle_short_side_target_and_stop(tmp_path):
    mem = Memory(memory_path(tmp_path))
    mem.record(bundle(), decision(rating=Rating.SELL, target=97, stop=114))
    assert mem.settle(date(2026, 11, 10), history(), CAL) == 1
    s = lines(mem)[0]["settled"]
    assert s["hit_target"] is True  # low 96 <= 97
    assert s["hit_stop"] is False  # high 113 < 114


def test_settle_null_prices(tmp_path):
    mem = Memory(memory_path(tmp_path))
    mem.record(bundle(), decision(target=None, stop=None))
    assert mem.settle(date(2026, 11, 10), history(), CAL) == 1
    s = lines(mem)[0]["settled"]
    assert s["hit_target"] is None and s["hit_stop"] is None


def test_settle_skips_when_data_missing(tmp_path):
    mem = Memory(memory_path(tmp_path))
    mem.record(bundle(), decision())
    assert mem.settle(date(2026, 11, 10), FakeHistory({"005930.KS": STOCK}), CAL) == 0
    short = FakeHistory({"005930.KS": STOCK.iloc[:4], "^KS11": INDEX})  # window end row missing
    assert mem.settle(date(2026, 11, 10), short, CAL) == 0
    assert lines(mem)[0]["settled"] is None


def test_partial_line_is_ignored_and_not_glued(tmp_path):
    path = memory_path(tmp_path)
    mem = Memory(path)
    mem.record(bundle(), decision())
    with path.open("a", encoding="utf-8") as f:
        f.write('{"code": "0059')  # crash mid-append
    mem.record(bundle(day=date(2026, 11, 3)), decision())
    dates = [r["date"] for r in mem.entries()]
    assert dates == ["2026-11-02", "2026-11-03"]
    assert mem.settle(date(2026, 11, 20), history(), CAL) == 2


def test_lessons_exclude_future_dates(tmp_path):
    mem = Memory(memory_path(tmp_path))
    mem.record(bundle(), decision())
    assert mem.settle(date(2026, 11, 10), history(), CAL) == 1

    # decision date (11-02) < as_of, but scoring finished on 11-10 >= as_of: excluded
    assert mem.lessons("005930", date(2026, 11, 5)) == ""
    assert mem.lessons("005930", date(2026, 11, 10)) == ""
    text = mem.lessons("005930", date(2026, 11, 11))
    assert "2026-11-02" in text and "매수(BUY)" in text
    assert "+8.0%" in text and "+2.0%" in text and "+6.0%" in text
    assert mem.lessons("035720", date(2026, 11, 11)) == ""


def test_lessons_newest_first_and_limit(tmp_path):
    mem = Memory(memory_path(tmp_path))
    for d in SESSIONS[:3]:
        mem.record(bundle(day=d, horizon="1d"), decision())
    mem.record(bundle(day=date(2026, 11, 10), horizon="5d"), decision())  # window not passed
    assert mem.settle(date(2026, 11, 10), history(), CAL) == 3
    text = mem.lessons("005930", date(2026, 11, 11))
    found = [line[2:12] for line in text.splitlines()]
    assert found == ["2026-11-04", "2026-11-03", "2026-11-02"]
    assert len(mem.lessons("005930", date(2026, 11, 11), limit=2).splitlines()) == 2


def _with_row(df: pd.DataFrame, day: date, like: date) -> pd.DataFrame:
    extra = df.loc[[pd.Timestamp(like)]].set_axis(pd.DatetimeIndex([pd.Timestamp(day)], name="date"))
    return pd.concat([df, extra]).sort_index()


def test_settle_tolerates_stale_extra_row(tmp_path):
    # Yahoo row on a KRX holiday (11-07, Sat) inside the window must not push out the start row
    stock = _with_row(STOCK, date(2026, 11, 7), like=date(2026, 11, 6))
    index = _with_row(INDEX, date(2026, 11, 7), like=date(2026, 11, 6))
    mem = Memory(memory_path(tmp_path))
    mem.record(bundle(), decision())
    h = FakeHistory({"005930.KS": stock, "^KS11": index})
    assert mem.settle(date(2026, 11, 10), h, CAL) == 1
    s = lines(mem)[0]["settled"]
    assert s["return"] == pytest.approx(0.08)
    assert s["bench_return"] == pytest.approx(0.02)


def test_settle_tolerates_missing_intermediate_session(tmp_path):
    gap = STOCK.drop(pd.Timestamp(date(2026, 11, 5)))  # the 113 high was on 11-05
    mem = Memory(memory_path(tmp_path))
    mem.record(bundle(), decision())
    assert mem.settle(date(2026, 11, 10), FakeHistory({"005930.KS": gap, "^KS11": INDEX}), CAL) == 1
    s = lines(mem)[0]["settled"]
    assert s["return"] == pytest.approx(0.08)
    assert s["hit_target"] is False  # remaining highs top out at 109


def test_settle_gives_up_after_max_attempts(tmp_path):
    mem = Memory(memory_path(tmp_path))
    mem.record(bundle(), decision())
    assert mem.settle(date(2026, 11, 9), FakeHistory({}), CAL) == 0  # not due: no attempt counted
    assert "attempts" not in lines(mem)[0]

    failing = FakeHistory({})  # every symbol raises FetchError (e.g. delisted)
    for i in range(MAX_ATTEMPTS):
        assert mem.settle(date(2026, 11, 10 + i), failing, CAL) == 0
        assert lines(mem)[0]["attempts"] == i + 1
    calls = len(failing.calls)
    assert mem.settle(date(2026, 11, 20), failing, CAL) == 0
    assert len(failing.calls) == calls

    good = history()
    assert mem.settle(date(2026, 11, 20), good, CAL) == 0
    assert good.calls == []
    assert lines(mem)[0]["settled"] is None
