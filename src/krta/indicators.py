"""Technical indicators computed by code so the LLM only reads results."""

from __future__ import annotations

from typing import Literal

import pandas as pd

SHORTAGE = "데이터 부족"
DAILY_ONLY = "일봉에서만 계산"
LIMIT_THRESHOLD = 0.295  # 가격제한폭 ±30%의 근접 판정
LIMIT_LOOKBACK = 60  # 거래일
_EPS = 1e-9
_MA_WINDOWS = (5, 20, 60, 120)
# 52주 고저: 일봉 252, 주봉 52, 월봉 12개.
_YEAR_BARS = {"D": 252, "W": 52, "M": 12}
_RESAMPLE_RULE = {"W": "W-FRI", "M": "ME"}

Timeframe = Literal["D", "W", "M"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: str(c).lower() for c in df.columns}
    out = df.rename(columns=cols)
    for need in ("close", "volume"):
        if need not in out.columns:
            raise ValueError(f"일봉 DataFrame에 {need} 열이 없음 (열: {list(df.columns)})")
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("일봉 DataFrame의 인덱스는 DatetimeIndex여야 함")
    out = out.sort_index()
    out = out[out["close"].notna()]
    if "high" not in out.columns:
        out["high"] = out["close"]
    if "low" not in out.columns:
        out["low"] = out["close"]
    return out[["high", "low", "close", "volume"]]


def _resample(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    if timeframe == "D":
        return df
    agg = df.resample(_RESAMPLE_RULE[timeframe]).agg(
        {"high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return agg[agg["close"].notna()]


def _limit_hits(df: pd.DataFrame) -> list[dict] | str:
    if len(df) < LIMIT_LOOKBACK + 1:
        return SHORTAGE
    change = df["close"].pct_change().iloc[-LIMIT_LOOKBACK:]
    hits = change[change.abs() >= LIMIT_THRESHOLD - _EPS]
    return [{"date": ts.date(), "change_pct": round(float(r) * 100, 2)} for ts, r in hits.items()]


def technical_summary(df: pd.DataFrame, timeframe: Timeframe = "D") -> dict:
    """일봉 df(DatetimeIndex, close·volume 필수, high·low 선택; 열 이름 대소문자 무관)로 지표 계산.

    W·M은 주봉(금요일 마감)·월봉으로 리샘플해 같은 키로 계산하고, 각 창은 봉 단위다.
    창이 모자라면 값 대신 SHORTAGE. limit_hits는 일봉에서만 계산한다.
    """
    if timeframe not in _YEAR_BARS:
        raise ValueError(f"timeframe은 D/W/M 중 하나: {timeframe!r}")
    daily = _normalize(df)
    bars = _resample(daily, timeframe)
    close, volume, n = bars["close"], bars["volume"], len(bars)

    out: dict = {"close": float(close.iloc[-1]) if n else SHORTAGE}
    for w in _MA_WINDOWS:
        out[f"ma{w}"] = float(close.iloc[-w:].mean()) if n >= w else SHORTAGE

    # 당일(마지막 봉) 거래량 / 직전 20봉 평균 거래량
    base = volume.iloc[-21:-1].mean() if n >= 21 else None
    out["vol_ratio_20"] = float(volume.iloc[-1] / base) if base else SHORTAGE
    # 최근 20개 봉 수익률의 표준편차(표본, ddof=1)
    out["volatility_20"] = float(close.pct_change().iloc[-20:].std()) if n >= 21 else SHORTAGE

    year = _YEAR_BARS[timeframe]
    out["high_52w"] = float(bars["high"].iloc[-year:].max()) if n >= year else SHORTAGE
    out["low_52w"] = float(bars["low"].iloc[-year:].min()) if n >= year else SHORTAGE

    out["limit_hits"] = _limit_hits(daily) if timeframe == "D" else DAILY_ONLY
    return out
