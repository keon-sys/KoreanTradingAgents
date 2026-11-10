import numpy as np
import pandas as pd
import pytest

from krta.indicators import SHORTAGE, technical_summary


def df_of_length(n, start="2025-01-02", close=None, volume=None):
    idx = pd.bdate_range(start, periods=n)
    close = np.linspace(10000, 12000, n) if close is None else np.asarray(close, dtype=float)
    volume = np.full(n, 1000.0) if volume is None else np.asarray(volume, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": volume},
        index=idx,
    )


def test_short_history_gives_data_shortage():
    s = technical_summary(df_of_length(30))
    assert s["ma60"] == "데이터 부족" and s["ma20"] != "데이터 부족"
    assert s["ma120"] == SHORTAGE and s["high_52w"] == SHORTAGE and s["limit_hits"] == SHORTAGE
    assert set(s) == {
        "close", "ma5", "ma20", "ma60", "ma120", "vol_ratio_20",
        "volatility_20", "high_52w", "low_52w", "limit_hits",
    }


def test_values_on_known_series():
    close = np.arange(1, 301, dtype=float) * 100  # 100..30000
    vol = np.full(300, 1000.0)
    vol[-1] = 3000.0
    s = technical_summary(df_of_length(300, close=close, volume=vol))
    assert s["close"] == 30000.0
    assert s["ma5"] == pytest.approx(np.mean(close[-5:]))
    assert s["ma120"] == pytest.approx(np.mean(close[-120:]))
    # 당일 거래량 / 직전 20거래일 평균
    assert s["vol_ratio_20"] == pytest.approx(3.0)
    rets = pd.Series(close).pct_change().iloc[-20:]
    assert s["volatility_20"] == pytest.approx(rets.std())
    assert s["high_52w"] == pytest.approx(30000 * 1.01)
    assert s["low_52w"] == pytest.approx(close[-252] * 0.99)
    assert s["limit_hits"] == []


def test_limit_hits_detected():
    close = np.full(80, 10000.0)
    close[70:] = 13000.0  # 70번째 날 +30% 마감
    df = df_of_length(80, close=close)
    s = technical_summary(df)
    assert len(s["limit_hits"]) == 1
    hit = s["limit_hits"][0]
    assert hit["date"] == df.index[70].date()
    assert hit["change_pct"] == pytest.approx(30.0)


def test_limit_hits_threshold_and_window():
    close = np.full(100, 10000.0)
    close[10] = 7000.0  # -30%지만 최근 60거래일 밖
    close[50] = 12900.0  # +29.0%: 기준 미달 (다음 날 복귀 -22.5%)
    close[90:] = 7050.0  # -29.5%: 경계 포함
    s = technical_summary(df_of_length(100, close=close))
    assert [h["change_pct"] for h in s["limit_hits"]] == [pytest.approx(-29.5)]


def test_weekly_and_monthly_resample():
    df = df_of_length(300)
    w = technical_summary(df, "W")
    m = technical_summary(df, "M")
    # 300영업일 = 주봉 61개, 월봉 15개
    assert w["ma60"] != SHORTAGE and w["ma120"] == SHORTAGE
    assert m["ma5"] != SHORTAGE and m["ma20"] == SHORTAGE
    assert w["close"] == pytest.approx(12000.0) and m["close"] == pytest.approx(12000.0)
    assert w["limit_hits"] == m["limit_hits"] == "일봉에서만 계산"
    weekly = df["Volume"].resample("W-FRI").sum()
    assert w["vol_ratio_20"] == pytest.approx(weekly.iloc[-1] / weekly.iloc[-21:-1].mean())


def test_lowercase_columns_accepted():
    df = df_of_length(30).rename(columns=str.lower)
    assert technical_summary(df)["ma20"] != SHORTAGE


def test_bad_input():
    with pytest.raises(ValueError):
        technical_summary(df_of_length(30), "X")
    with pytest.raises(ValueError):
        technical_summary(df_of_length(30).drop(columns=["Close"]))
