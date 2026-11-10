"""Input parsing: horizon, analysis date (KRX trading calendar), stock code and market."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

KST = ZoneInfo("Asia/Seoul")
# 투자자별 매매 등 장 마감 뒤 데이터가 확정되는 시각(KST).
CONFIRM_TIME = time(18, 0)


class OutOfScopeError(Exception):
    """입력이 이 도구의 범위(KOSPI·KOSDAQ 상장 주식) 밖일 때."""


class InputError(ValueError):
    """요청 자체가 잘못됐을 때(종목코드 형식, 시장 접미사, 상장일 이전 분석일). 종료 코드 2.

    ValueError의 하위 클래스라 입력 파싱 단계의 다른 ValueError와 함께 잡힌다. 수집 중에는
    이 예외만 입력 오류로 보고, 다른 ValueError(데이터 형식 오류 등)는 실행 실패로 본다."""


# ---------------------------------------------------------------- horizon

Bucket = Literal["단기", "중기", "장기"]

_HORIZON_RE = re.compile(r"^(\d{1,3})([dwmy])$")
_UNIT_DAYS = {"d": 1, "w": 5, "m": 21, "y": 252}


@dataclass(frozen=True)
class Horizon:
    raw: str
    trading_days: int
    bucket: Bucket


def _bucket(trading_days: int) -> Bucket:
    if trading_days <= 20:
        return "단기"
    if trading_days <= 126:
        return "중기"
    return "장기"


def parse_horizon(s: str) -> Horizon:
    m = _HORIZON_RE.fullmatch(s)
    if m is None:
        raise ValueError(f"기간 형식 오류: {s!r} (예: 5d, 4w, 6m, 1y)")
    n, unit = int(m.group(1)), m.group(2)
    if n <= 0:
        raise ValueError(f"기간은 1 이상이어야 함: {s!r}")
    trading_days = n * _UNIT_DAYS[unit]
    # 출력 경로에 쓰이므로 "05d" 같은 표기는 "5d"로 정규화한다.
    return Horizon(f"{n}{unit}", trading_days, _bucket(trading_days))


# ---------------------------------------------------------------- calendar

_CAL_START = "2000-01-01"
_CAL_END = "2050-12-31"  # XKRX bound_max


@lru_cache(maxsize=1)
def _sessions() -> pd.DatetimeIndex:
    # exchange_calendars는 휴장일을 패키지에 내장하고 있어 네트워크를 쓰지 않는다.
    cal = xcals.get_calendar("XKRX", start=_CAL_START, end=_CAL_END)
    return pd.DatetimeIndex(cal.sessions).normalize()


class TradingCalendar:
    """KRX(XKRX) 거래일 달력. 날짜는 모두 KST 기준 `date`."""

    def __init__(self) -> None:
        self._s = _sessions()
        self._first = self._s[0].date()
        self._last = self._s[-1].date()

    def _check(self, d: date) -> pd.Timestamp:
        if not (self._first <= d <= self._last):
            raise ValueError(f"거래일 달력 범위({self._first}~{self._last}) 밖의 날짜: {d}")
        return pd.Timestamp(d)

    def is_session(self, d: date) -> bool:
        return self._check(d) in self._s

    def previous_session(self, d: date) -> date:
        """d보다 앞선(d 제외) 가장 가까운 거래일."""
        i = self._s.searchsorted(self._check(d), side="left") - 1
        if i < 0:
            raise ValueError(f"{d} 이전 거래일이 달력 범위에 없음")
        return self._s[i].date()

    def next_sessions(self, d: date, n: int) -> list[date]:
        """d 다음(d 제외) 거래일 n개."""
        if n < 0:
            raise ValueError(f"n은 0 이상이어야 함: {n}")
        i = self._s.searchsorted(self._check(d), side="right")
        if i + n > len(self._s):
            raise ValueError(f"{d} 이후 거래일 {n}개가 달력 범위({self._last})를 넘음")
        return [t.date() for t in self._s[i : i + n]]

    def sessions_between(self, a: date, b: date) -> list[date]:
        """a~b(양끝 포함) 사이 거래일."""
        if a > b:
            raise ValueError(f"시작일 {a}이 종료일 {b}보다 늦음")
        lo = self._s.searchsorted(self._check(a), side="left")
        hi = self._s.searchsorted(self._check(b), side="right")
        return [t.date() for t in self._s[lo:hi]]


# ---------------------------------------------------------------- analysis date


@dataclass(frozen=True)
class ResolvedDate:
    date: date
    unconfirmed: bool
    note: str | None


def resolve_date(requested: date | None, now: datetime, cal: TradingCalendar) -> ResolvedDate:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now에는 시간대가 있어야 함")
    now_kst = now.astimezone(KST)
    today = now_kst.date()
    after_close = now_kst.time() >= CONFIRM_TIME

    if requested is None:
        if cal.is_session(today) and after_close:
            return ResolvedDate(today, False, None)
        return ResolvedDate(cal.previous_session(today), False, None)

    if requested > today:
        raise ValueError(f"미래 날짜는 분석할 수 없음: {requested} (오늘 {today} KST)")
    if not cal.is_session(requested):
        p = cal.previous_session(requested)
        return ResolvedDate(p, False, f"요청일 {requested.isoformat()}은 휴장일이라 직전 거래일 {p.isoformat()}로 분석")
    if requested == today and not after_close:
        return ResolvedDate(requested, True, None)
    return ResolvedDate(requested, False, None)


# ---------------------------------------------------------------- code / market

Market = Literal["KS", "KQ"]

_CODE_RE = re.compile(r"^(\d{6})(?:\.(KS|KQ))?$", re.IGNORECASE)


def normalize_code(raw: str) -> tuple[str, str | None]:
    m = _CODE_RE.fullmatch(raw.strip())
    if m is None:
        raise InputError(f"종목코드 형식 오류: {raw!r} (예: 005930, 005930.KS, 035720.KQ)")
    suffix = m.group(2)
    return m.group(1), (suffix.upper() if suffix else None)


def resolve_market(
    code: str, suffix: str | None, kospi_codes: set[str], kosdaq_codes: set[str]
) -> Market:
    markets: list[Market] = []
    if code in kospi_codes:
        markets.append("KS")
    if code in kosdaq_codes:
        markets.append("KQ")
    if not markets:
        raise OutOfScopeError("KRX 주식 기본정보에 없는 코드(ETF·ETN·리츠 등은 범위 밖)")
    if suffix is not None:
        suffix = suffix.upper()
        if suffix not in markets:
            raise InputError(f"{code}은 {'/'.join(markets)} 종목인데 접미사가 .{suffix}임")
        return suffix  # type: ignore[return-value]
    if len(markets) > 1:
        raise InputError(f"{code}이 KOSPI·KOSDAQ 양쪽에 있음: .KS 또는 .KQ를 붙여 지정")
    return markets[0]


def check_listed(listing_date: date, d: date) -> None:
    if d < listing_date:
        raise InputError(f"분석일 {d}은 상장일 {listing_date} 이전")
