"""Daily price history from Yahoo Finance (single stocks, indices, FX) via yfinance.

yfinance does its own HTTP, so it is kept behind the injectable `fetch` callable; tests
replace it. Rows after the analysis date are dropped.
"""

from __future__ import annotations

import logging
import math
import tempfile
import threading
import warnings
from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd

from krta.net import FetchError

SOURCE = "yahoo"

MACRO_SYMBOLS = {
    "원달러": "KRW=X",
    "S&P500": "^GSPC",
    "나스닥": "^IXIC",
    "필라델피아반도체": "^SOX",
    "상해종합": "000001.SS",
    "코스피": "^KS11",
    "코스닥": "^KQ11",
}

COLUMNS = ["open", "high", "low", "close", "volume"]

# Calendar days requested per trading row wanted: 7/5 for weekends, plus slack for
# holidays (KRX/SSE close ~15 weekdays a year), plus a fixed margin for short windows.
_CALENDAR_PER_TRADING_DAY = 7 / 5 * 1.15
_CALENDAR_MARGIN_DAYS = 14
_MAX_MESSAGE = 300

Fetch = Callable[[str, date, date], pd.DataFrame]

_cache_lock = threading.Lock()
_cache_dir: str | None = None


def history(symbol: str, end: date, days: int, *, fetch: Fetch | None = None) -> pd.DataFrame:
    """Last `days` daily rows of `symbol` up to and including `end`.

    Columns exactly `open, high, low, close, volume`; index `date` is a tz-naive
    DatetimeIndex at midnight of the exchange-local session date (so callers can
    resample). Raises FetchError on fetch failure or when no row is left.
    """
    if days <= 0:
        raise ValueError(f"days must be positive: {days}")
    start = end - timedelta(days=math.ceil(days * _CALENDAR_PER_TRADING_DAY) + _CALENDAR_MARGIN_DAYS)
    fetch = fetch if fetch is not None else _yfinance_fetch
    try:
        raw = fetch(symbol, start, end)
        error = None
    except FetchError:
        raise
    except Exception as exc:  # yfinance raises many unrelated types
        raw, error = None, f"{symbol} 조회 실패: {type(exc).__name__}: {exc}"
    if error is not None:
        # Raised outside `except` so the original exception is not chained.
        raise FetchError(SOURCE, error[:_MAX_MESSAGE]) from None
    df = _normalize(symbol, raw)
    df = df[df.index <= pd.Timestamp(end)]
    if df.empty:
        raise FetchError(SOURCE, f"{symbol}: {end.isoformat()} 이전 일봉 없음")
    return df.tail(days)


def _normalize(symbol: str, raw: pd.DataFrame | None) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise FetchError(SOURCE, f"{symbol}: 일봉 없음")
    df = raw.rename(columns=lambda c: str(c).strip().lower())
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise FetchError(SOURCE, f"{symbol}: 컬럼 없음 {missing}")
    df = df[COLUMNS].copy()
    index = pd.DatetimeIndex(df.index)
    # Keep the exchange-local calendar date before dropping the timezone.
    if index.tz is not None:
        index = index.tz_localize(None)
    df.index = index.normalize().rename("date")
    df = df[df["close"].notna()]
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def _yfinance_fetch(symbol: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf
    import yfinance.cache as yf_cache

    _prepare_yfinance(yf_cache)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # yfinance `end` is exclusive. Unadjusted OHLC so prices match KRX quotes;
        # `_normalize` keeps the raw `Close` and drops `Adj Close`.
        return yf.Ticker(symbol).history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            raise_errors=True,
        )


def _prepare_yfinance(yf_cache) -> None:
    """Point yfinance's sqlite caches (tz, cookie, ISIN) at a fresh temp dir and mute it.

    The container may run with a read-only root filesystem; the default location may not be
    writable, and nothing from yfinance should persist between runs. Must happen before
    yfinance touches its cache, hence once, under a lock.
    """
    global _cache_dir
    with _cache_lock:
        if _cache_dir is None:
            _cache_dir = tempfile.mkdtemp(prefix="krta-yfinance-")
            yf_cache.set_cache_location(_cache_dir)
            # Failures surface as exceptions (raise_errors=True); its log lines are noise.
            logging.getLogger("yfinance").setLevel(logging.CRITICAL)
