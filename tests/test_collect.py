import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fakes import FakeLLM

from krta.collect import SECTION_KEYS, DataBundle, Sources, Target, collect
from krta.data.dart import DartKeyError
from krta.data.krx_openapi import KrxAuthError
from krta.data.toss import REASON_403, TossUnavailable
from krta.data.news import KST, NewsItem
from krta.inputs import OutOfScopeError, ResolvedDate, TradingCalendar, parse_horizon
from krta.net import FetchError
from krta.redact import Redactor
from krta.toollog import ToolLog

FIXTURES = Path(__file__).parent / "fixtures"
D = date(2026, 11, 6)  # Friday, KRX session
CAL = TradingCalendar()


def load(*parts):
    return json.loads((FIXTURES.joinpath(*parts)).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- fakes


def _base_row(code, name, list_dd="19750611", shares="5,969,782,550"):
    return {
        "ISU_CD": f"KR7{code}003", "ISU_SRT_CD": code, "ISU_NM": name, "ISU_ABBRV": name,
        "LIST_DD": list_dd, "SECUGRP_NM": "주권", "KIND_STKCERT_TP_NM": "보통주",
        "LIST_SHRS": shares,
    }


class FakeKrx:
    def __init__(self, *, fail_trades=False):
        self.fail_trades = fail_trades
        self.calls = []

    def base_info(self, market, d):
        self.calls.append(("base_info", market, d))
        if market == "KS":
            return [
                _base_row("005930", "삼성전자"),
                _base_row("000660", "SK하이닉스", "19961226", "728,002,365"),
                _base_row("005935", "삼성전자우"),
            ]
        return [_base_row("035720", "카카오", "19991111", "443,256,000")]

    def daily_trades(self, market, d):
        self.calls.append(("daily_trades", market, d))
        if self.fail_trades:
            raise FetchError("krx_openapi", "HTTP 503: sto/stk_bydd_trd", status=503)
        name = "stk_bydd_trd.json" if market == "KS" else "ksq_bydd_trd.json"
        return load("krx", name)["OutBlock_1"]

    def index_daily(self, market, d):
        self.calls.append(("index_daily", market, d))
        rows = load("krx", "kospi_dd_trd.json")["OutBlock_1"] if market == "KS" else []
        # Sector indices whose level depends on the date, so 20-day changes differ.
        k = (d - date(2026, 1, 1)).days
        rows = rows + [
            {"IDX_NM": "전기전자", "CLSPRC_IDX": f"{1000 + 3 * k:.2f}"},
            {"IDX_NM": "음식료품", "CLSPRC_IDX": f"{2000 - k:.2f}"},
        ]
        return rows

    def futures_daily(self, d):
        return []


def _flow_rows(start, end, foreign=10_000_000_000):
    out = []
    for i, d in enumerate(CAL.sessions_between(start, end)):
        sign = 1 if i % 3 else -1
        out.append({
            "date": d, "외국인": sign * foreign, "기관합계": -sign * 4_000_000_000,
            "금융투자": -1_000_000_000, "보험": 0, "투신": -1_000_000_000, "사모": 0,
            "은행": 0, "기타금융": 0, "연기금": -2_000_000_000, "기타법인": 0,
            "개인": -sign * 6_000_000_000,
        })
    return out


class FakeToss:
    def available(self):
        return True

    def investor_flow(self, code, start, end):
        return _flow_rows(start, end)

    def investor_flow_market(self, market, start, end):
        return _flow_rows(start, end, foreign=500_000_000_000)

    def shorting(self, code, start, end):
        return [
            {"date": d, "공매도거래량": 1000, "공매도거래대금": 98_500_000,
             "공매도잔고수량": None, "공매도잔고금액": None,
             "공매도거래량비중": 0.5, "공매도거래대금비중": None}
            for d in CAL.sessions_between(start, end)
        ]

    def foreign_holding(self, code, start, end):
        return [
            {"date": d, "외국인보유수량": 2_990_000_000, "외국인지분율": 50.5, "한도소진율": 50.5}
            for d in CAL.sessions_between(start, end)
        ]

    def program_trading(self, code, start, end):
        return [
            {"date": d, "차익매수": 10, "차익매도": 30, "차익순매수": -20,
             "비차익매수": 500, "비차익매도": 400, "비차익순매수": 100}
            for d in CAL.sessions_between(start, end)
        ]

    def credit_balance(self, code, start, end):
        # Published T+1: the analysis day itself is not there yet.
        days = [d for d in CAL.sessions_between(start, end) if d < end]
        return [
            {"date": d, "융자신규": 100, "융자상환": 90, "융자잔고": 1_000_000 + 10 * i,
             "융자잔고율": 0.42, "융자공여율": 9.0, "대주신규": None, "대주상환": None,
             "대주잔고": None, "대주잔고율": None, "대주공여율": None}
            for i, d in enumerate(days)
        ]

    def securities_lending(self, code, start, end):
        return [
            {"date": d, "대차체결": 200, "대차상환": 100, "대차잔고수량": 15_000_000 + 100 * i,
             "대차잔고금액": 1_200_000_000_000}
            for i, d in enumerate(CAL.sessions_between(start, end))
        ]


class FailingToss:
    def available(self):
        return True

    def _fail(self, *a, **k):
        raise TossUnavailable(REASON_403)

    investor_flow = investor_flow_market = shorting = foreign_holding = _fail
    program_trading = credit_balance = securities_lending = _fail


def _filing(dt, name, rcept_no=None):
    return {
        "rcept_dt": dt, "report_nm": name, "rcept_no": rcept_no or f"{dt}000001",
        "flr_nm": "삼성전자", "is_correction": "정정" in name,
    }


def _fin(sj, account_id, name, th, fr="", bf="", add="", fr_add=""):
    return {
        "rcept_no": "20260310000001", "sj_div": sj, "account_id": account_id,
        "account_nm": name, "thstrm_amount": th, "frmtrm_amount": fr,
        "bfefrmtrm_amount": bf, "thstrm_add_amount": add, "frmtrm_add_amount": fr_add,
    }


class FakeDart:
    def __init__(self, *, filings=None, key_error=False):
        self.key_error = key_error
        self._filings = filings if filings is not None else [
            _filing("20261020", "주요사항보고서(전환사채권발행결정)", "20261015000555"),
            _filing("20261005", "현금ㆍ현물배당을위한주주명부폐쇄(기준일)결정"),
            _filing("20251030", "연결재무제표기준영업(잠정)실적(공정공시)"),
            _filing("20250915", "회사분할결정"),
        ]
        self.calls = []

    def corp_code(self, stock_code):
        if self.key_error:
            raise DartKeyError("DART 키 문제(상태 010)")
        return {"005930": "00126380", "000660": "00164779", "035720": "00258801"}[stock_code]

    def company(self, corp):
        self.calls.append(("company", corp))
        return {"corp_name": "삼성전자", "induty_code": "264", "acc_mt": "12"}

    def filings(self, corp, start, end, *, as_of):
        self.calls.append(("filings", corp, start, end))
        return [f for f in self._filings if f["rcept_dt"] <= as_of.strftime("%Y%m%d")]

    def major_report(self, kind, corp, start, end, *, as_of):
        if kind == "CB":
            rows = load("dart", "cvbdIsDecsn.json")["list"]
            return [r for r in rows if r["rcept_no"][:8] <= as_of.strftime("%Y%m%d")]
        return []

    def financials(self, corp, year, reprt_code, *, as_of):
        self.calls.append(("financials", corp, year, reprt_code))
        if reprt_code == "11011" and year == 2025:
            return [
                _fin("IS", "ifrs-full_Revenue", "매출액", "300,000,000,000,000",
                     "258,000,000,000,000", "302,000,000,000,000"),
                _fin("IS", "dart_OperatingIncomeLoss", "영업이익", "33,000,000,000,000",
                     "6,500,000,000,000", "43,000,000,000,000"),
                _fin("IS", "ifrs-full_ProfitLossAttributableToOwnersOfParent",
                     "지배기업의 소유주에게 귀속되는 당기순이익(손실)", "30,000,000,000,000",
                     "14,000,000,000,000", "54,000,000,000,000"),
                _fin("BS", "ifrs-full_Liabilities", "부채총계", "100,000,000,000,000"),
                _fin("BS", "ifrs-full_Equity", "자본총계", "400,000,000,000,000"),
                _fin("BS", "ifrs-full_EquityAttributableToOwnersOfParent",
                     "지배기업 소유주지분", "390,000,000,000,000"),
            ]
        if reprt_code == "11014" and year == 2026:
            return [
                _fin("IS", "ifrs-full_ProfitLossAttributableToOwnersOfParent",
                     "지배기업의 소유주에게 귀속되는 당기순이익(손실)", "10,000,000,000,000",
                     add="28,000,000,000,000", fr_add="18,000,000,000,000"),
                _fin("BS", "ifrs-full_EquityAttributableToOwnersOfParent",
                     "지배기업 소유주지분", "400,000,000,000,000"),
            ]
        return []

    def largest_holders(self, corp, year, reprt_code, *, as_of):
        if (year, reprt_code) != (2026, "11014"):
            return []
        return [
            {"nm": "삼성생명보험", "relate": "최대주주", "stock_knd": "보통주",
             "trmend_posesn_stock_co": "508,157,148", "trmend_posesn_stock_qota_rt": "8.51"},
            {"nm": "삼성물산", "relate": "계열회사", "stock_knd": "보통주",
             "trmend_posesn_stock_co": "298,818,100", "trmend_posesn_stock_qota_rt": "5.01"},
            {"nm": "계", "relate": "-", "stock_knd": "보통주",
             "trmend_posesn_stock_co": "806,975,248", "trmend_posesn_stock_qota_rt": "13.52"},
        ]

    def major_holders(self, corp, *, as_of):
        return [{"rcept_dt": "2026-10-01", "repror": "국민연금공단", "stkqy": "430,000,000",
                 "stkrt": "7.20", "stkqy_irds": "-1,000,000", "stkrt_irds": "-0.02",
                 "report_resn": "보유비율 변동"}]

    def exec_holdings(self, corp, *, as_of):
        return []

    def dividends(self, corp, year, reprt_code, *, as_of):
        return [{"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,446"}]


class FakeNews:
    def __init__(self, n=25):
        self.n = n
        self.calls = []

    def search(self, query, as_of, days=14):
        self.calls.append((query, as_of, days))
        base = datetime(as_of.year, as_of.month, as_of.day, 15, tzinfo=KST)
        return [
            NewsItem(f"{query} 기사 {i}", "연합뉴스", base - timedelta(hours=i), f"https://x/{i}")
            for i in range(self.n)
        ]


class FakeHistory:
    def __init__(self):
        self.calls = []

    def __call__(self, symbol, end, days):
        self.calls.append((symbol, end, days))
        idx = pd.bdate_range(end=pd.Timestamp(end), periods=days)
        n = len(idx)
        close = pd.Series([100.0 + i * 0.5 for i in range(n)], index=idx)
        return pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1, "close": close,
             "volume": [1_000_000.0] * n},
            index=idx.rename("date"),
        )


def sources(**over):
    s = {
        "krx": FakeKrx(), "toss": FakeToss(), "dart": FakeDart(),
        "news": FakeNews(),
        "history": FakeHistory(), "cal": CAL,
    }
    s.update(over)
    return Sources(**s)


def peer_llm(text="SK하이닉스\n없는회사"):
    return FakeLLM(lambda system, user: text)


def run(tmp_path, *, src=None, llm=None, horizon="1m", unconfirmed=False, code="005930",
        note=None, d=D):
    log = ToolLog(tmp_path / "message_tool.log", Redactor([]))
    b = collect(
        code, ResolvedDate(d, unconfirmed, note), parse_horizon(horizon),
        src or sources(), llm or peer_llm(), log,
    )
    return b, (tmp_path / "message_tool.log")


# --------------------------------------------------------------------------- tests


def test_bundle_has_six_sections_and_calendar(tmp_path):
    b, _ = run(tmp_path, note="요청일 메모")
    assert isinstance(b, DataBundle)
    assert set(b.sections) == set(SECTION_KEYS) == {
        "market", "flow", "theme", "technical", "disclosure", "fundamentals"}
    assert b.target == Target("005930", "KS", "삼성전자", "00126380", date(1975, 6, 11))
    assert (b.date, b.unconfirmed, b.date_note) == (D, False, "요청일 메모")
    assert all(s.text.strip() for s in b.sections.values())
    assert "선물옵션 만기" in b.calendar_md and "| 날짜 |" in b.calendar_md
    assert "⚠" not in b.sections["market"].text


def test_toss_unavailable_marks_flow_missing_but_continues(tmp_path):
    b, _ = run(tmp_path, src=sources(toss=FailingToss()))
    flow = b.sections["flow"]
    assert "데이터 없음" in flow.text and "허용 IP 미등록" in flow.text
    assert flow.missing and all(m.startswith("데이터 없음") for m in flow.missing)
    assert b.sections["technical"].text
    assert "ma20" in b.sections["technical"].text
    assert "데이터 없음" in b.sections["market"].text  # market-wide investor flow too


def test_without_toss_flow_items_are_missing_with_key_names(tmp_path):
    for toss in (None, type("NoKeys", (FakeToss,), {"available": lambda self: False})()):
        b, _ = run(tmp_path, src=sources(toss=toss))
        flow = b.sections["flow"].text
        for what in ("투자자별 순매수", "외국인 보유", "공매도", "대차잔고", "프로그램 매매", "신용잔고"):
            assert f"데이터 없음: {what} (토스증권 API 키 없음(TOSS_CLIENT_ID, TOSS_CLIENT_SECRET))" in flow
        assert "데이터 없음: 시장 전체 투자자별 순매수" in b.sections["market"].text


def test_program_trading_credit_and_lending_from_toss(tmp_path):
    b, _ = run(tmp_path)
    text = b.sections["flow"].text
    assert "데이터 없음: 프로그램 매매" not in text and "데이터 없음: 신용잔고" not in text
    assert "데이터 없음: 대차잔고" not in text
    prog = text.split("### 프로그램 매매")[1].split("###")[0]
    assert "| 최근 5거래일 | -100 | 500 | 400 |" in prog
    credit = text.split("### 신용잔고")[1].split("###")[0]
    assert "신용잔고 최신일 2026-11-05" in credit and "0.42%" in credit and "공여율 9.00%" in credit
    assert "5거래일 전 대비 +50주" in credit
    lend = text.split("### 대차잔고")[1].split("###")[0]
    assert "5거래일 전 대비 +500주" in lend


def test_short_balance_missing_not_zero(tmp_path):
    b, _ = run(tmp_path)
    part = b.sections["flow"].text.split("### 공매도")[1].split("###")[0]
    assert "데이터 없음: 공매도 잔고 (토스 API는 공매도 잔고를 제공하지 않음)" in part
    assert "| 2026-11-06 | 1,000 | 98,500,000 | 0.50% |  |" in part


def test_provisional_none_values_are_blank_not_zero(tmp_path):
    class Provisional(FakeToss):
        def investor_flow(self, code, start, end):
            rows = super().investor_flow(code, start, end)
            for k in ("금융투자", "보험", "투신", "사모", "은행", "기타금융", "연기금", "기타법인", "개인"):
                rows[-1][k] = None
            return rows

    b, _ = run(tmp_path, src=sources(toss=Provisional()))
    part = b.sections["flow"].text.split("### 투자자별 순매수")[1].split("### 외국인")[0]
    assert "(값 없는 날 1)" in part and "0이 아님" in part
    daily = [ln for ln in part.splitlines() if ln.startswith("| 2026-11-06 |")]
    assert daily and daily[-1].endswith("|  |  |  |  |  |  |  |  |  |")
    assert "None" not in part


def test_dart_key_error_aborts(tmp_path):
    with pytest.raises(DartKeyError):
        run(tmp_path, src=sources(dart=FakeDart(key_error=True)))


def test_krx_auth_error_on_base_info_aborts(tmp_path):
    class NoAuthKrx(FakeKrx):
        def base_info(self, market, d):
            raise KrxAuthError("KRX Open API 인증 실패")

    with pytest.raises(KrxAuthError):
        run(tmp_path, src=sources(krx=NoAuthKrx()))


def test_daily_trades_failure_is_soft(tmp_path):
    b, _ = run(tmp_path, src=sources(krx=FakeKrx(fail_trades=True)))
    assert "데이터 없음" in b.sections["theme"].text and "HTTP 503" in b.sections["theme"].text
    assert b.sections["technical"].text


def test_peers_only_validated_names(tmp_path):
    llm = peer_llm("1. SK하이닉스\n- 없는회사\n삼성전자\n")
    b, _ = run(tmp_path, llm=llm)
    assert [p.name for p in b.peers] == ["SK하이닉스"]
    p = b.peers[0]
    assert (p.code, p.market, p.corp_code, p.listing_date) == (
        "000660", "KS", "00164779", date(1996, 12, 26))
    assert "동료 종목은 LLM 추정 후 상장 목록으로 검증" in b.sections["theme"].text
    _system, user = llm.calls[0]
    assert "삼성전자" in user and "264" in user and "삼성전자 기사 19" in user
    assert "삼성전자 기사 20" not in user  # 20 headlines at most


def test_peer_list_capped_at_eight(tmp_path):
    many = "\n".join(["없는회사"] * 8 + ["SK하이닉스"])
    b, _ = run(tmp_path, llm=peer_llm(many))
    assert b.peers == []


def test_unconfirmed_banner(tmp_path):
    b, _ = run(tmp_path, horizon="5d", unconfirmed=True)
    for s in b.sections.values():
        assert s.text.startswith("⚠ 분석일 당일 데이터는 미확정")
    assert "신용잔고 최신일 2026-11-05" in b.sections["flow"].text


def test_critical_disclosure_listed(tmp_path):
    b, _ = run(tmp_path)
    text = b.sections["disclosure"].text
    head, tail = text.split("치명적 공시 후보:")
    assert "전환사채" in tail.splitlines()[1] and "CB" in tail.splitlines()[1]
    assert "회사분할결정" not in tail
    # CB dilution: 1,912,045 / 5,969,782,550 listed shares
    assert "0.03%" in head
    # the 2026-11-10 CB row is after the analysis date and must not appear
    assert "53,000" not in text


def test_out_of_scope_code_raises(tmp_path):
    with pytest.raises(OutOfScopeError):
        run(tmp_path, code="069500")


def test_us_macro_uses_previous_day_others_analysis_date(tmp_path):
    h = FakeHistory()
    run(tmp_path, src=sources(history=h))
    ends = {sym: end for sym, end, _ in h.calls}
    for sym in ("^GSPC", "^IXIC", "^SOX"):
        assert ends[sym] == D - timedelta(days=1)
    for sym in ("KRW=X", "000001.SS", "^KS11", "^KQ11", "005930.KS"):
        assert ends[sym] == D
    assert ("005930.KS", D, 400) in h.calls


def test_thin_news_noted_but_used(tmp_path):
    b, _ = run(tmp_path, src=sources(news=FakeNews(n=2)))
    text = b.sections["theme"].text
    assert "데이터 없음: 뉴스 (기사 수 부족(2건))" in text
    assert "삼성전자 기사 1" in text


def test_news_window_by_horizon(tmp_path):
    n = FakeNews()
    run(tmp_path, src=sources(news=n), horizon="1y")
    assert ("삼성전자", D, 30) in n.calls


def test_technical_timeframes_and_unadjusted_note(tmp_path):
    short, _ = run(tmp_path / "s", horizon="5d")
    long, _ = run(tmp_path / "l", horizon="1y")
    assert "수정주가 아님" in short.sections["technical"].text
    assert "일봉" in short.sections["technical"].text and "주봉" not in short.sections["technical"].text
    t = long.sections["technical"].text
    assert "주봉" in t and "월봉" in t


def test_flow_cumulative_and_estimated_avg_price(tmp_path):
    b, _ = run(tmp_path)
    text = b.sections["flow"].text
    assert "누적 순매수" in text and "추정 평균 단가" in text and "추정" in text


def test_valuation_per_pbr(tmp_path):
    b, _ = run(tmp_path)
    text = b.sections["fundamentals"].text
    # TTM 지배순이익 = 30조 + 28조 - 18조 = 40조; 시총 588,012,345,678,000
    assert "14.70" in text  # PER
    # PBR = 시총 / 최근 지배자본 400조
    assert "1.47" in text
    assert "13.52" in text  # 최대주주·특수관계인 지분 합 (8.51 + 5.01)
    assert "회사분할결정" in text  # 지배구조 이력


def test_calendar_only_in_range_dividends(tmp_path):
    b, _ = run(tmp_path)
    # past dividend filing goes to the disclosure section, not the calendar
    assert "배당기준일" not in b.calendar_md
    assert "- 최근 1년 배당 관련 공시 1건(최근 2026-10-05) — 기준일은 공시 섹션·본문 확인" in b.calendar_md
    assert "주주명부폐쇄" in b.sections["disclosure"].text
    # 2025-10-30 잠정실적 + 1y = 2026-10-30 is before the analysis date: not a hint
    assert "실적 발표 예상" not in b.calendar_md


def test_earnings_hint_in_range(tmp_path):
    dart = FakeDart(filings=[_filing("20251113", "연결재무제표기준영업(잠정)실적(공정공시)")])
    b, _ = run(tmp_path, src=sources(dart=dart))
    assert "| 2026-11-13 | 실적 발표 예상 (추정)" in b.calendar_md


def test_sources_logged_to_toollog(tmp_path):
    _, log = run(tmp_path)
    text = log.read_text(encoding="utf-8")
    for src in ("krx_openapi.base_info", "google_news", "dart.filings", "toss.investor_flow",
                "yahoo.history 005930.KS", "krx_openapi.daily_trades"):
        assert f"[data] {src}" in text, src


def test_dart_key_error_mid_run_aborts(tmp_path):
    class KeyFailsLater(FakeDart):
        def filings(self, corp, start, end, *, as_of):
            raise DartKeyError("DART 키 문제(상태 011)")

    with pytest.raises(DartKeyError):
        run(tmp_path, src=sources(dart=KeyFailsLater()))


def test_corp_code_fetch_failure_is_soft_with_reason(tmp_path):
    class CorpDown(FakeDart):
        def corp_code(self, stock_code):
            raise FetchError("dart", "HTTP 503: corpCode.xml", status=503)

    b, _ = run(tmp_path, src=sources(dart=CorpDown()))
    assert b.target.corp_code == ""
    assert "DART 고유번호 조회 실패: HTTP 503" in b.sections["disclosure"].text
    assert "DART 고유번호 조회 실패" in b.sections["fundamentals"].text
    assert b.sections["technical"].text


def test_base_info_falls_back_to_previous_session_when_unpublished(tmp_path):
    class Unpublished(FakeKrx):
        def base_info(self, market, d):
            return [] if d == D else super().base_info(market, d)

    krx = Unpublished()
    b, _ = run(tmp_path, src=sources(krx=krx), unconfirmed=True)
    assert b.target.code == "005930" and b.date == D


@pytest.mark.parametrize("horizon, end", [("1m", date(2026, 12, 7)), ("5d", date(2026, 11, 13))])
def test_window_end_is_last_session_of_horizon(tmp_path, horizon, end):
    b, _ = run(tmp_path, horizon=horizon)
    assert b.window_end == end


def test_sector_table_excludes_broad_indices(tmp_path):
    b, _ = run(tmp_path)
    text = b.sections["theme"].text
    sector = text.split("섹터 쏠림")[1].split("###")[0]
    assert "전기전자" in sector and "코스피 200" not in sector
    assert sector.index("전기전자") < sector.index("음식료품")  # sorted by 20-day change


def test_correction_compares_changed_fields_only(tmp_path):
    class Corrected(FakeDart):
        def major_report(self, kind, corp, start, end, *, as_of):
            if kind != "CB":
                return []
            orig = {"rcept_no": "20261015000555", "bd_tm": "5", "cv_prc": "52,300",
                    "bd_fta": "100,000,000,000"}
            fix = {**orig, "rcept_no": "20261030000777", "cv_prc": "48,000"}
            return [orig, fix]

    dart = Corrected(filings=[
        _filing("20261015", "주요사항보고서(전환사채권발행결정)", "20261015000555"),
        _filing("20261030", "[기재정정]주요사항보고서(전환사채권발행결정)", "20261030000777"),
        _filing("20261102", "[기재정정]사업보고서 (2025.12)", "20261102000888"),
    ])
    b, _ = run(tmp_path, src=sources(dart=dart))
    text = b.sections["disclosure"].text
    part = text.split("### 정정공시 비교")[1].split("###")[0]
    assert "원보고 20261015000555 → 최종 정정 20261030000777" in part
    assert "| cv_prc | 52,300 | 48,000 |" in part
    assert "bd_fta" not in part  # unchanged fields are not listed
    assert "사업보고서 (2025.12): 정정 내용 비교 불가(본문 확인 필요)" in part


ANNUAL_NI = "지배기업의 소유주에게 귀속되는 당기순이익(손실)"


class QuarterDart(FakeDart):
    """Annual 2025 as in FakeDart; the latest quarter report is replaced by `quarter`."""

    def __init__(self, quarter_code, quarter_rows, annual_rows=None, **kw):
        super().__init__(**kw)
        self.quarter_code, self.quarter_rows, self.annual_rows = quarter_code, quarter_rows, annual_rows

    def financials(self, corp, year, reprt_code, *, as_of):
        if reprt_code == "11011" and year == 2025 and self.annual_rows is not None:
            return self.annual_rows
        if reprt_code == "11011":
            return super().financials(corp, year, reprt_code, as_of=as_of)
        if (year, reprt_code) == (2026, self.quarter_code):
            return self.quarter_rows
        return []


def _valuation_line(b):
    return next(ln for ln in b.sections["fundamentals"].text.splitlines()
                if ln.startswith("최근 4분기 지배순이익"))


def test_ttm_without_cumulative_fields_uses_annual(tmp_path):
    # 3Q report with only the 3-month amount: must not be treated as year-to-date.
    rows = [_fin("IS", "ifrs-full_ProfitLossAttributableToOwnersOfParent", ANNUAL_NI, "10,000,000,000,000")]
    rows[0]["frmtrm_q_amount"] = "8,000,000,000,000"  # 3-month figures only, no *_add_amount
    b, _ = run(tmp_path, src=sources(dart=QuarterDart("11014", rows)))
    line = _valuation_line(b)
    assert "30,000,000,000,000원" in line and "분기 누적값 없음" in line
    assert "19.60" in b.sections["fundamentals"].text  # 588.01조 / 30조


def test_ttm_first_quarter_three_month_is_ytd(tmp_path):
    row = _fin("IS", "ifrs-full_ProfitLossAttributableToOwnersOfParent", ANNUAL_NI, "9,000,000,000,000")
    row["frmtrm_q_amount"] = "6,000,000,000,000"
    b, _ = run(tmp_path, src=sources(dart=QuarterDart("11013", [row])))
    assert "33,000,000,000,000원 (최근 연간 + 당해 누적 - 전년 동기 누적)" in _valuation_line(b)


def test_ttm_never_mixes_parent_and_total_net_income(tmp_path):
    # Annual has 지배주주순이익; the quarter only 당기순이익 -> annual figure, not a mix.
    rows = [_fin("IS", "ifrs-full_ProfitLoss", "당기순이익", "1", add="99,000,000,000,000",
                 fr_add="1,000,000,000,000")]
    b, _ = run(tmp_path, src=sources(dart=QuarterDart("11014", rows)))
    line = _valuation_line(b)
    assert "30,000,000,000,000원" in line and "분기 누적값 없음" in line


def test_ttm_total_net_income_basis_is_stated(tmp_path):
    annual = [_fin("IS", "ifrs-full_ProfitLoss", "당기순이익", "30,000,000,000,000"),
              _fin("BS", "ifrs-full_Equity", "자본총계", "400,000,000,000,000")]
    b, _ = run(tmp_path, src=sources(dart=QuarterDart("11014", [], annual_rows=annual)))
    line = _valuation_line(b)
    assert "지배주주순이익 없음: 당기순이익 사용" in line
    assert "지배자본 없음: 자본총계 사용" in line
    assert "1.47" in b.sections["fundamentals"].text


def test_foreign_holding_missing_values_not_zero(tmp_path):
    class Gappy(FakeToss):
        def foreign_holding(self, code, start, end):
            rows = super().foreign_holding(code, start, end)
            for r in rows[-3:]:
                r["외국인지분율"] = None
            for r in rows:
                r["한도소진율"] = None
            return rows

    b, _ = run(tmp_path, src=sources(toss=Gappy()))
    part = b.sections["flow"].text.split("### 외국인 보유 비중")[1].split("###")[0]
    assert "0.00" not in part and "None" not in part
    assert "| 외국인지분율(%) | 50.50 (2026-08-10) | 50.50 (2026-11-03) | 50.50 | 50.50 | 3 |" in part
    assert "데이터 없음: 외국인 한도소진율 (값 없음)" in part


def test_holder_without_ratio_skipped_and_noted(tmp_path):
    class NoRate(FakeDart):
        def largest_holders(self, corp, year, reprt_code, *, as_of):
            rows = super().largest_holders(corp, year, reprt_code, as_of=as_of)
            if rows:
                rows[1]["trmend_posesn_stock_qota_rt"] = "-"
            return rows

    b, _ = run(tmp_path, src=sources(dart=NoRate()))
    assert "지분 합: 8.51% (2026 3분기 기준, 1건 비율 없음: 합에서 제외)" in b.sections["fundamentals"].text


def test_empty_market_listing_is_fetch_error_not_out_of_scope(tmp_path):
    class NoKosdaq(FakeKrx):
        def base_info(self, market, d):
            return [] if market == "KQ" else super().base_info(market, d)

    with pytest.raises(FetchError, match=r"KRX 종목 기본정보가 비어 있음\(KQ, 2026-11-05\)"):
        run(tmp_path, src=sources(krx=NoKosdaq()))


def test_one_market_unpublished_on_d_uses_previous_session(tmp_path):
    class KqLate(FakeKrx):
        def base_info(self, market, d):
            return [] if (market, d) == ("KQ", D) else super().base_info(market, d)

    b, _ = run(tmp_path, src=sources(krx=KqLate()), code="035720")
    assert b.target.market == "KQ"


def test_peer_llm_failure_degrades_without_echo(tmp_path):
    def boom(system, user):
        raise RuntimeError("upstream said SENSITIVE-DETAIL-123")

    b, log = run(tmp_path, llm=FakeLLM(boom))
    assert b.peers == []
    for key in ("theme", "fundamentals"):
        assert "데이터 없음: 동료 종목 (LLM 추정 실패)" in b.sections[key].text
    everything = "".join(x.text for x in b.sections.values()) + log.read_text(encoding="utf-8")
    assert "SENSITIVE-DETAIL-123" not in everything
    assert b.sections["technical"].text


def test_program_and_lending_none_values_do_not_abort(tmp_path):
    class Holes(FakeToss):
        def program_trading(self, code, start, end):
            rows = super().program_trading(code, start, end)
            rows[-1]["차익순매수"] = None
            return rows

        def securities_lending(self, code, start, end):
            rows = super().securities_lending(code, start, end)
            rows[-1]["대차잔고수량"] = None
            return rows

    b, _ = run(tmp_path, src=sources(toss=Holes()))
    text = b.sections["flow"].text
    prog = text.split("### 프로그램 매매")[1].split("###")[0]
    assert "| 최근 5거래일 | -80 (값 없는 날 1) | 500 | 320 (값 없는 날 1) |" in prog
    assert "| 2026-11-06 |  | 100 |  |" in prog
    lend = text.split("### 대차잔고")[1].split("###")[0]
    assert "최근 대차잔고 2026-11-05: 15,005,800주" in lend
    assert "| 2026-11-06 | 200 | 100 |  |" in lend
    assert "None" not in prog and "None" not in lend


def test_program_and_lending_all_none_is_missing(tmp_path):
    class Empty(FakeToss):
        def program_trading(self, code, start, end):
            rows = super().program_trading(code, start, end)
            for r in rows:
                r["차익순매수"] = r["비차익순매수"] = None
            return rows

        def securities_lending(self, code, start, end):
            rows = super().securities_lending(code, start, end)
            for r in rows:
                r["대차잔고수량"] = None
            return rows

    b, _ = run(tmp_path, src=sources(toss=Empty()))
    text = b.sections["flow"].text
    assert "데이터 없음: 프로그램 매매 (순매수 값 없음)" in text
    assert "데이터 없음: 대차잔고 (잔고 값 없음)" in text


# --------------------------------------------------------------------------- report periods


class LateAnnualDart(FakeDart):
    """FY2025 사업보고서 missing; FY2024 present (FakeDart's annual rows); 2026 1Q present."""

    def financials(self, corp, year, reprt_code, *, as_of):
        self.calls.append(("financials", corp, year, reprt_code))
        if reprt_code == "11011":
            return FakeDart.financials(self, corp, 2025, "11011", as_of=as_of) if year == 2024 else []
        if (year, reprt_code) == (2026, "11013"):
            return [_fin("IS", "ifrs-full_ProfitLossAttributableToOwnersOfParent", ANNUAL_NI,
                         "9,000,000,000,000", add="9,000,000,000,000", fr_add="6,000,000,000,000")]
        return []


def test_quarter_after_older_annual_is_not_mixed_into_ttm(tmp_path):
    dart = LateAnnualDart()
    b, _ = run(tmp_path, src=sources(dart=dart), d=date(2026, 5, 20))
    text = b.sections["fundamentals"].text
    line = _valuation_line(b)
    assert "30,000,000,000,000원 (최근 연간(2024))" in line
    assert "당해 누적" not in line
    assert "FY2025 사업보고서 없음 또는 분석일 기준 미공개" in text
    assert "최근 분기: 2026" not in text
    # Only quarters of FY2024 + 1 are looked at.
    quarter_years = {c[2] for c in dart.calls if c[0] == "financials" and c[3] != "11011"}
    assert quarter_years == {2025}


def test_latest_annual_has_no_fallback_note(tmp_path):
    b, _ = run(tmp_path)
    assert "사업보고서 없음 또는 분석일 기준 미공개" not in b.sections["fundamentals"].text


def test_annual_fetch_failure_is_one_failure_line_not_absence(tmp_path):
    class Failing(FakeDart):
        def financials(self, corp, year, reprt_code, *, as_of):
            if reprt_code == "11011":
                raise FetchError("dart", "HTTP 503: fnlttSinglAcntAll", status=503)
            return super().financials(corp, year, reprt_code, as_of=as_of)

    b, _ = run(tmp_path, src=sources(dart=Failing()))
    sec = b.sections["fundamentals"]
    assert "사업보고서 없음" not in sec.text
    assert [m for m in sec.missing if "재무제표" in m] == [
        "데이터 없음: 재무제표(2025 사업보고서) (HTTP 503: fnlttSinglAcntAll)"]


def test_peer_annual_fetch_failure_is_noted_as_failure(tmp_path):
    class PeerFails(FakeDart):
        def financials(self, corp, year, reprt_code, *, as_of):
            if corp == "00164779" and reprt_code == "11011":
                raise FetchError("dart", "timeout")
            return super().financials(corp, year, reprt_code, as_of=as_of)

    b, _ = run(tmp_path, src=sources(dart=PeerFails()))
    row = next(ln for ln in b.sections["fundamentals"].text.splitlines()
               if ln.startswith("| SK하이닉스"))
    assert "조회 실패(timeout)" in row and "사업보고서 없음" not in row


def test_calendar_never_lists_dividend_record_date_from_filing_receipt(tmp_path):
    # A dividend filing received on the analysis date is in range, but its receipt date is
    # not the record date: only the notes line mentions it.
    dart = FakeDart(filings=[_filing("20261106", "현금ㆍ현물배당결정")])
    b, _ = run(tmp_path, src=sources(dart=dart))
    assert "배당기준일" not in b.calendar_md
    assert "- 최근 1년 배당 관련 공시 1건(최근 2026-11-06)" in b.calendar_md


def test_peer_quarter_failure_falls_back_to_annual_with_note(tmp_path):
    class PeerQuarterFails(FakeDart):
        def financials(self, corp, year, reprt_code, *, as_of):
            if corp == "00164779" and reprt_code != "11011":
                raise FetchError("dart", "timeout")
            return super().financials(corp, year, reprt_code, as_of=as_of)

    b, _ = run(tmp_path, src=sources(dart=PeerQuarterFails()))
    row = next(ln for ln in b.sections["fundamentals"].text.splitlines()
               if ln.startswith("| SK하이닉스"))
    assert "30,000,000,000,000" in row  # FY2025 annual only, no quarter adjustment
    assert "2026 3분기 조회 실패(timeout): 최근 연간" in row
