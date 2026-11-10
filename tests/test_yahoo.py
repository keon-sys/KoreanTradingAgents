import logging
from datetime import date

import pandas as pd
import pytest

from krta.data import yahoo
from krta.data.yahoo import MACRO_SYMBOLS, history
from krta.net import FetchError

END = date(2026, 11, 10)


def yf_frame(days, tz="Asia/Seoul"):
    """Shaped like yfinance Ticker.history(): capitalised columns, tz-aware index, extras."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days]).tz_localize(tz)
    n = len(days)
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [110.0 + i for i in range(n)],
            "Low": [90.0 + i for i in range(n)],
            "Close": [105.0 + i for i in range(n)],
            "Volume": [1000 + i for i in range(n)],
            "Dividends": [0.0] * n,
            "Stock Splits": [0.0] * n,
        },
        index=idx,
    ).rename_axis("Date")


class FakeFetch:
    def __init__(self, frame=None, exc=None):
        self.frame, self.exc, self.calls = frame, exc, []

    def __call__(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        if self.exc is not None:
            raise self.exc
        return self.frame


def test_history_clips_after_end():
    fetch = FakeFetch(yf_frame(["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11"]))
    df = history("005930.KS", END, 30, fetch=fetch)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "date"
    assert [d.date() for d in df.index] == [date(2026, 11, 6), date(2026, 11, 9), date(2026, 11, 10)]
    assert df.index.tz is None
    assert df.loc["2026-11-10", "close"] == 107.0
    symbol, start, end = fetch.calls[0]
    assert symbol == "005930.KS" and end == END and start < END


def test_history_keeps_last_n_trading_rows():
    days = pd.bdate_range("2026-01-01", "2026-11-10").strftime("%Y-%m-%d").tolist()
    fetch = FakeFetch(yf_frame(days))
    df = history("^KS11", END, 20, fetch=fetch)
    assert len(df) == 20
    assert df.index[-1].date() == END
    # Requested window is wide enough to hold `days` trading rows across weekends/holidays.
    _, start, _ = fetch.calls[0]
    assert (END - start).days >= 20 * 7 // 5


def test_history_uses_exchange_local_date():
    # 00:00 Asia/Seoul on 11-10 is 11-09 15:00 UTC; the row belongs to the 11-10 session.
    fetch = FakeFetch(yf_frame(["2026-11-10"]))
    df = history("^KS11", END, 5, fetch=fetch)
    assert [d.date() for d in df.index] == [END]


def test_history_drops_rows_without_close_and_duplicate_dates():
    frame = yf_frame(["2026-11-06", "2026-11-09", "2026-11-09", "2026-11-10"])
    frame.iloc[3, frame.columns.get_loc("Close")] = float("nan")
    df = history("KRW=X", END, 10, fetch=FakeFetch(frame))
    assert [d.date() for d in df.index] == [date(2026, 11, 6), date(2026, 11, 9)]
    assert df.loc["2026-11-09", "close"] == 107.0  # last duplicate wins


@pytest.mark.parametrize(
    "frame",
    [None, pd.DataFrame(), yf_frame(["2026-11-11", "2026-11-12"])],
    ids=["none", "empty", "only-after-end"],
)
def test_history_empty_is_fetch_error(frame):
    with pytest.raises(FetchError) as e:
        history("^SOX", END, 30, fetch=FakeFetch(frame))
    assert e.value.source == "yahoo" and "^SOX" in e.value.message


def test_history_missing_column_is_fetch_error():
    frame = yf_frame(["2026-11-10"]).drop(columns=["Volume"])
    with pytest.raises(FetchError) as e:
        history("^SOX", END, 30, fetch=FakeFetch(frame))
    assert e.value.source == "yahoo"


def test_history_fetch_exception_becomes_fetch_error():
    with pytest.raises(FetchError) as e:
        history("^GSPC", END, 30, fetch=FakeFetch(exc=RuntimeError("boom")))
    assert e.value.source == "yahoo" and "boom" in e.value.message
    assert e.value.__cause__ is None and e.value.__context__ is None


def test_history_rejects_nonpositive_days():
    with pytest.raises(ValueError):
        history("^GSPC", END, 0, fetch=FakeFetch(yf_frame(["2026-11-10"])))


def test_macro_symbols():
    assert MACRO_SYMBOLS == {
        "원달러": "KRW=X",
        "S&P500": "^GSPC",
        "나스닥": "^IXIC",
        "필라델피아반도체": "^SOX",
        "상해종합": "000001.SS",
        "코스피": "^KS11",
        "코스닥": "^KQ11",
    }


def test_default_fetch_moves_yfinance_cache_to_temp(monkeypatch, tmp_path):
    import yfinance as yf
    import yfinance.cache as yf_cache

    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        def history(self, **kwargs):
            seen["kwargs"] = kwargs
            return yf_frame(["2026-11-10"])

    monkeypatch.setattr(yahoo, "_cache_dir", None)
    monkeypatch.setattr(yahoo.tempfile, "mkdtemp", lambda prefix: str(tmp_path / prefix))
    monkeypatch.setattr(yf_cache, "set_cache_location", lambda d: seen.setdefault("cache", d))
    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    monkeypatch.setattr(logging.getLogger("yfinance"), "level", logging.NOTSET)

    df = history("^IXIC", END, 5)
    assert len(df) == 1
    assert seen["cache"].startswith(str(tmp_path))
    assert seen["symbol"] == "^IXIC"
    # yfinance `end` is exclusive: ask for the day after so `end` itself is included.
    assert seen["kwargs"]["end"] == "2026-11-11"
    assert seen["kwargs"]["raise_errors"] is True
    assert logging.getLogger("yfinance").level == logging.CRITICAL


def test_default_fetch_uses_unadjusted_close(monkeypatch, tmp_path):
    import yfinance as yf
    import yfinance.cache as yf_cache

    seen = {}
    # auto_adjust=False shape: raw OHLC plus `Adj Close`, which must be ignored.
    raw = yf_frame(["2026-11-09", "2026-11-10"]).drop(columns=["Dividends", "Stock Splits"])
    raw.insert(4, "Adj Close", [1.0, 2.0])

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            seen["kwargs"] = kwargs
            return raw

    monkeypatch.setattr(yahoo, "_cache_dir", None)
    monkeypatch.setattr(yahoo.tempfile, "mkdtemp", lambda prefix: str(tmp_path / prefix))
    monkeypatch.setattr(yf_cache, "set_cache_location", lambda d: None)
    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    monkeypatch.setattr(logging.getLogger("yfinance"), "level", logging.NOTSET)

    df = history("005930.KS", END, 5)
    assert seen["kwargs"]["auto_adjust"] is False
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].tolist() == [105.0, 106.0]
