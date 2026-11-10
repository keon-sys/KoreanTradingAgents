"""Data bundle collection: every data source -> six analyst sections + event calendar.

All numbers in the sections are computed here; the analysts only read them. Any
`FetchError`, `TossUnavailable`, `PortalUnavailable` or `KrxAuthError` turns the affected
item into a `missing_line` and collection continues. Two failures abort the whole run: the
listing lookup that fixes the target (no target, nothing to analyse) and `DartKeyError`.

Collection is sequential on purpose: `HttpClient`'s min-interval state is not
thread-safe.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timedelta
from typing import Any, ClassVar, Literal

import pandas as pd

from krta.data.dart import Dart
from krta.data.krx_openapi import KrxAuthError, KrxOpenApi, to_float, to_int
from krta.data.krx_portal import KrxPortal, PortalUnavailable
from krta.data.news import GoogleNews, NewsItem
from krta.data.toss import FLOW_KEYS, TossApi, TossUnavailable
from krta.data.yahoo import MACRO_SYMBOLS
from krta.events import MIN_SESSIONS, build_calendar, render_calendar
from krta.indicators import technical_summary
from krta.inputs import (
    Horizon,
    ResolvedDate,
    TradingCalendar,
    check_listed,
    normalize_code,
    resolve_market,
)
from krta.llm import LLM
from krta.net import FetchError
from krta.sections import Section, md_table, missing_line
from krta.toollog import ToolLog

SECTION_KEYS = ("market", "flow", "theme", "technical", "disclosure", "fundamentals")
UNCONFIRMED_BANNER = "⚠ 분석일 당일 데이터는 미확정"
PEER_NOTE = "동료 종목은 LLM 추정 후 상장 목록으로 검증"
CRITICAL_HEADER = "치명적 공시 후보:"
UNADJUSTED_NOTE = "수정주가 아님: 액면분할·병합 등은 가격 공백(급등락)으로 보인다"

# Look-back windows [계획 결정].
HISTORY_DAYS = 400  # trading rows of the stock's daily bars
CROSS_SECTION_DAYS = 20  # trading-value rank / theme concentration
FLOW_DAYS = {"단기": 60, "중기": 60, "장기": 250}  # investor flow, shorting, foreign holding, ...
DISCLOSURE_DAYS = {"단기": 90, "중기": 365, "장기": 3 * 365}
GOVERNANCE_DAYS = 3 * 365 + 1
NEWS_DAYS = {"단기": 14, "중기": 14, "장기": 30}
MACRO_DAYS = 61  # enough rows for a 60-day change
MIN_NEWS = 3
MAX_PEERS = 8
PEER_HEADLINES = 20
SHOWN_ROWS = 20

US_SYMBOLS = frozenset({"^GSPC", "^IXIC", "^SOX"})
MARKET_INDEX = {"KS": "^KS11", "KQ": "^KQ11"}
MARKET_NAME = {"KS": "코스피", "KQ": "코스닥"}
# Market-wide news for the market analyst (spec §6.1: index-wide policy, memory cycle).
MARKET_NEWS_QUERIES = {
    "지수 정책·규제": "증시 공매도 OR 밸류업 OR 금리 OR 세제",
    "메모리 반도체 가격": "메모리 반도체 가격",
}

# report_nm keywords. Labels are what the PM prompt reads.
CRITICAL_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("유상증자", ("유상증자",)),
    ("CB", ("전환사채",)),
    ("BW", ("신주인수권부사채",)),
    ("감사의견 비적정", ("비적정", "의견거절", "부적정", "감사의견한정", "감사의견 한정")),
    ("관리종목", ("관리종목",)),
    ("거래정지·상장폐지", ("거래정지", "매매정지", "상장폐지", "상장적격성")),
)

SOURCE_KRX = "krx_openapi"

_SOFT_ERRORS = (FetchError, TossUnavailable, PortalUnavailable, KrxAuthError)
NO_TOSS = "토스증권 API 키 없음(TOSS_CLIENT_ID, TOSS_CLIENT_SECRET)"

Market = Literal["KS", "KQ"]


@dataclass
class Target:
    code: str
    market: Market
    name: str
    corp_code: str
    listing_date: date


@dataclass
class DataBundle:
    target: Target
    date: date
    horizon: Horizon
    window_end: date  # last session of the horizon window (horizon.trading_days after date)
    unconfirmed: bool
    date_note: str | None
    peers: list[Target]
    calendar_md: str
    sections: dict[str, Section]


@dataclass
class Sources:
    """Already-built data clients.

    Construction rules for the caller:
    - `toss` (investor flows, foreign holding, short selling, program trading, credit,
      lending) gets its OWN `HttpClient(..., min_interval=krta.data.toss.MIN_INTERVAL)`
      and the SAME `Redactor` as the `ToolLog`: TossApi registers its access token with
      that redactor at run time. None (or no keys) -> those items become missing lines.
    - `portal` (KrxPortal) is kept for compatibility but no longer called: plaintext
      login is refused (spec §17). Leave it None.
    - On an unconfirmed run, build `KrxOpenApi(..., unconfirmed_date=<analysis date>)` so
      the still-changing day is not cached.
    - `history(symbol, end, days)` is `krta.data.yahoo.history`: `days` counts trading
      rows; prices are unadjusted.
    """

    krx: KrxOpenApi
    dart: Dart
    news: GoogleNews
    history: Callable[[str, date, int], pd.DataFrame]
    cal: TradingCalendar
    toss: TossApi | None = None
    portal: KrxPortal | None = None


def collect(
    raw_code: str,
    resolved: ResolvedDate,
    horizon: Horizon,
    src: Sources,
    llm: LLM,
    toollog: ToolLog,
) -> DataBundle:
    return _Collector(resolved, horizon, src, llm, toollog).run(raw_code)


# =========================================================================== helpers


class _Unavailable(Exception):
    """Internal soft failure (e.g. no DART corp code) with a displayable reason."""


def _reason(exc: Exception) -> str:
    if isinstance(exc, FetchError):
        return exc.message
    if isinstance(exc, (TossUnavailable, PortalUnavailable)):
        return exc.reason
    return str(exc)


class _Sec:
    def __init__(self, key: str):
        self.key = key
        self.lines: list[str] = []
        self.missing: list[str] = []

    def h(self, title: str) -> None:
        self.lines += ["", f"### {title}"]

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def table(self, rows: list[dict], columns: list[str]) -> None:
        self.lines.append(md_table(rows, columns))

    def miss(self, what: str, why: str) -> None:
        line = missing_line(what, why)
        self.missing.append(line)
        self.lines.append(line)

    def build(self, banner: bool) -> Section:
        body = "\n".join(self.lines).strip()
        if banner:
            body = f"{UNCONFIRMED_BANNER}\n\n{body}"
        return Section(self.key, body + "\n", list(self.missing))


def _dump(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        return value.to_csv()

    def default(o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        return str(o)

    return json.dumps(value, ensure_ascii=False, default=default)


def _won(v: float | None) -> str:
    return "" if v is None else f"{round(v):,}"


def _pct(x: float | None, signed: bool = True) -> str:
    if x is None:
        return ""
    return f"{x:+.2f}%" if signed else f"{x:.2f}%"


def _num(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:,.2f}"
    return str(x)


def _int_or_none(s: Any) -> int | None:
    try:
        return to_int(None if s is None else str(s))
    except ValueError:
        return None


def _float_or_none(s: Any) -> float | None:
    try:
        return to_float(None if s is None else str(s))
    except ValueError:
        return None


def _known_sum(values: Iterable[int | None]) -> tuple[int | None, int]:
    """(sum of the known values or None if none is known, number of unknown values)."""
    vals = list(values)
    known = [v for v in vals if v is not None]
    return (sum(known) if known else None), len(vals) - len(known)


def _ymd(text: Any) -> date | None:
    s = str(text or "").strip().replace("-", "").replace("/", "").replace(".", "")
    try:
        return datetime.strptime(s[:8], "%Y%m%d").date()
    except ValueError:
        return None


def _plus_year(d: date) -> date:
    try:
        return d.replace(year=d.year + 1)
    except ValueError:  # Feb 29
        return d.replace(year=d.year + 1, day=28)


def _change(closes: pd.Series, n: int) -> float | None:
    """Percent change of the last close vs the close n rows earlier."""
    if len(closes) <= n:
        return None
    base = float(closes.iloc[-1 - n])
    return (float(closes.iloc[-1]) / base - 1) * 100 if base else None


def _filing_date(row: dict) -> date | None:
    return _ymd(row.get("rcept_dt")) or _ymd(str(row.get("rcept_no") or "")[:8])


def _matches(name: str, keywords: Iterable[str]) -> bool:
    return any(k in name for k in keywords)


def _critical_label(name: str) -> str | None:
    for label, keywords in CRITICAL_KEYWORDS:
        if _matches(name, keywords):
            return label
    return None


_BULLET = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s*")


def _peer_candidates(text: str) -> list[str]:
    lines = [_BULLET.sub("", ln).strip().strip("\"'`") for ln in text.splitlines()]
    return [ln for ln in lines if ln][:MAX_PEERS]


# --------------------------------------------------------------------------- financials

# (label, statement kinds, account ids, account names)
_ACCOUNTS: tuple[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    ("매출액", ("IS", "CIS"), ("ifrs-full_Revenue",), ("매출액", "수익(매출액)", "영업수익", "매출")),
    ("영업이익", ("IS", "CIS"), ("dart_OperatingIncomeLoss",), ("영업이익", "영업이익(손실)")),
    ("당기순이익", ("IS", "CIS"), ("ifrs-full_ProfitLoss",), ("당기순이익", "당기순이익(손실)")),
    ("지배주주순이익", ("IS", "CIS"), ("ifrs-full_ProfitLossAttributableToOwnersOfParent",),
     ("지배기업의 소유주에게 귀속되는 당기순이익(손실)", "지배기업 소유주지분", "지배기업의 소유주")),
    ("자산총계", ("BS",), ("ifrs-full_Assets",), ("자산총계",)),
    ("부채총계", ("BS",), ("ifrs-full_Liabilities",), ("부채총계",)),
    ("자본총계", ("BS",), ("ifrs-full_Equity",), ("자본총계",)),
    ("지배주주지분", ("BS",), ("ifrs-full_EquityAttributableToOwnersOfParent",),
     ("지배기업 소유주지분", "지배기업의 소유주에게 귀속되는 자본")),
    ("영업활동현금흐름", ("CF",), ("ifrs-full_CashFlowsFromUsedInOperatingActivities",),
     ("영업활동현금흐름", "영업활동으로 인한 현금흐름")),
    ("투자활동현금흐름", ("CF",), ("ifrs-full_CashFlowsFromUsedInInvestingActivities",),
     ("투자활동현금흐름", "투자활동으로 인한 현금흐름")),
    ("재무활동현금흐름", ("CF",), ("ifrs-full_CashFlowsFromUsedInFinancingActivities",),
     ("재무활동현금흐름", "재무활동으로 인한 현금흐름")),
)
_QUARTER_CODES = (("11014", "3분기"), ("11012", "반기"), ("11013", "1분기"))


def _account(rows: list[dict], label: str) -> dict | None:
    for lab, sjs, ids, names in _ACCOUNTS:
        if lab != label:
            continue
        cand = [r for r in rows if str(r.get("sj_div", "")) in sjs]
        for r in cand:
            if r.get("account_id") in ids:
                return r
        for r in cand:
            if str(r.get("account_nm", "")).strip() in names:
                return r
    return None


def _amount(rows: list[dict], label: str, field: str) -> int | None:
    r = _account(rows, label)
    return None if r is None else _int_or_none(r.get(field))


def _parent_ni(rows: list[dict], field: str) -> tuple[int | None, bool]:
    """(지배주주순이익, is_fallback): falls back to 당기순이익 for separate statements."""
    v = _amount(rows, "지배주주순이익", field)
    if v is not None:
        return v, False
    return _amount(rows, "당기순이익", field), True


_FIRST_QUARTER = "11013"  # 1Q: the 3-month figures are the year-to-date figures


def _ttm_parent_ni(year: int, annual: list[dict],
                   quarter: tuple[str, list[dict]] | None) -> tuple[int | None, str]:
    """(최근 4분기 순이익, how) from the FY`year` annual and a quarter of `year` + 1.

    Annual and quarter use the same account (no mixing)."""
    base, fb = _parent_ni(annual, "thstrm_amount")
    label = "당기순이익" if fb else "지배주주순이익"
    basis = " (지배주주순이익 없음: 당기순이익 사용)" if fb else ""
    if base is None or quarter is None:
        return base, f"최근 연간({year})" + basis
    code, rows = quarter
    ytd = _amount(rows, label, "thstrm_add_amount")
    prev = _amount(rows, label, "frmtrm_add_amount")
    if code == _FIRST_QUARTER:
        ytd = ytd if ytd is not None else _amount(rows, label, "thstrm_amount")
        prev = prev if prev is not None else _amount(rows, label, "frmtrm_q_amount")
    if ytd is None or prev is None:
        return base, f"최근 연간({year}, 분기 누적값 없음)" + basis
    return base + ytd - prev, "최근 연간 + 당해 누적 - 전년 동기 누적" + basis


# =========================================================================== collector


class _Collector:
    def __init__(self, resolved: ResolvedDate, horizon: Horizon, src: Sources, llm: LLM,
                 toollog: ToolLog):
        self.d = resolved.date
        self.resolved = resolved
        self.h = horizon
        self.src = src
        self.cal = src.cal
        self.llm = llm
        self.log = toollog
        self.secs = {k: _Sec(k) for k in SECTION_KEYS}
        # Shared intermediate results.
        self.base_rows: dict[str, list[dict]] = {}
        self.target_row: dict = {}
        self.company: dict = {}
        self.news: list[NewsItem] = []
        self.rejected_peers: list[str] = []
        self.history: pd.DataFrame | None = None
        self.history_error: str | None = None
        self.filings: list[dict] | None = None
        self.filings_error: str | None = None
        self.latest_quote: dict[str, dict] = {}  # code -> daily_trades row on the latest day
        self.latest_quote_day: date | None = None
        self.major: dict[str, list[dict]] = {}
        self.corp_error: str | None = None
        self.peer_error: str | None = None

    # ------------------------------------------------------------------ plumbing

    def _get(self, source: str, fn: Callable[[], Any], *, secs: Iterable[_Sec] = (),
             what: str = "", excerpt: Callable[[Any], Any] | None = None) -> tuple[Any, str | None]:
        """Run one source call; log raw output (or an excerpt) to the tool log."""
        try:
            value = fn()
        except (*_SOFT_ERRORS, _Unavailable) as exc:
            reason = _reason(exc)
            self.log.write("data", source, f"실패: {reason}")
            for s in secs:
                s.miss(what or source, reason)
            return None, reason
        self.log.write("data", source, _dump(excerpt(value) if excerpt else value))
        return value, None

    def _sessions(self, n: int) -> list[date]:
        """The n sessions ending at the analysis date (inclusive), oldest first."""
        return self.cal.sessions_between(self.d - timedelta(days=2 * n + 30), self.d)[-n:]

    def _toss(self) -> TossApi:
        if self.src.toss is None or not self.src.toss.available():
            raise _Unavailable(NO_TOSS)
        return self.src.toss

    def _corp(self) -> str:
        if not self.target.corp_code:
            raise _Unavailable(self.corp_error or "DART 고유번호 없음")
        return self.target.corp_code

    # ------------------------------------------------------------------ run

    def run(self, raw_code: str) -> DataBundle:
        self.target = self._resolve_target(raw_code)
        self._collect_news()
        self.peers = self._find_peers()
        self._collect_history()
        self._market()
        self._flow()
        self._theme()
        self._technical()
        self._collect_filings()
        self._disclosure()
        self._fundamentals()
        calendar_md = self._calendar()
        banner = self.resolved.unconfirmed
        return DataBundle(
            target=self.target, date=self.d, horizon=self.h,
            window_end=self.cal.next_sessions(self.d, self.h.trading_days)[-1],
            unconfirmed=self.resolved.unconfirmed, date_note=self.resolved.note,
            peers=self.peers, calendar_md=calendar_md,
            sections={k: s.build(banner) for k, s in self.secs.items()},
        )

    # ------------------------------------------------------------------ target

    def _resolve_target(self, raw_code: str) -> Target:
        code, suffix = normalize_code(raw_code)
        day = self.d
        # Failures here abort: without the listing there is no target.
        rows = {m: self.src.krx.base_info(m, day) for m in ("KS", "KQ")}
        if not rows["KS"] and not rows["KQ"]:
            # Not published yet for the analysis date (e.g. intraday): use the previous day.
            day = self.cal.previous_session(self.d)
            rows = {m: self.src.krx.base_info(m, day) for m in ("KS", "KQ")}
        for m, rs in rows.items():
            if not rs and day == self.d:
                # One market empty on D: try the previous session before giving up.
                prev = self.cal.previous_session(self.d)
                rs = rows[m] = self.src.krx.base_info(m, prev)
                if not rs:
                    raise FetchError(SOURCE_KRX, f"KRX 종목 기본정보가 비어 있음({m}, {prev})")
            elif not rs:
                raise FetchError(SOURCE_KRX, f"KRX 종목 기본정보가 비어 있음({m}, {day})")
        self.base_rows = rows
        codes = {m: {str(r.get("ISU_SRT_CD", "")) for r in rs} for m, rs in rows.items()}
        market = resolve_market(code, suffix, codes["KS"], codes["KQ"])
        row = next(r for r in rows[market] if r.get("ISU_SRT_CD") == code)
        self.target_row = row
        listing = _ymd(row.get("LIST_DD"))
        if listing is None:
            raise ValueError(f"KRX 기본정보 상장일 형식 오류: {row.get('LIST_DD')!r}")
        check_listed(listing, self.d)
        self.log.write("data", f"krx_openapi.base_info KS/KQ {day.isoformat()} (발췌)", _dump({
            "rows": {m: len(rs) for m, rs in rows.items()}, "target": row,
        }))
        corp, self.corp_error = self._corp_code(code)
        return Target(code, market, str(row.get("ISU_ABBRV", "")), corp, listing)

    def _corp_code(self, code: str) -> tuple[str, str | None]:
        """(corp code, None) or ("", reason). DartKeyError propagates."""
        try:
            return self.src.dart.corp_code(code), None
        except LookupError:
            return "", "DART 고유번호 없음"
        except FetchError as exc:
            self.log.write("data", f"dart.corp_code {code}", f"실패: {exc.message}")
            return "", f"DART 고유번호 조회 실패: {exc.message}"

    # ------------------------------------------------------------------ news / peers

    def _collect_news(self) -> None:
        days = NEWS_DAYS[self.h.bucket]
        items, err = self._get(
            f"google_news {self.target.name} {days}d",
            lambda: self.src.news.search(self.target.name, self.d, days),
        )
        self.news_error = err
        self.news = items or []

    def _find_peers(self) -> list[Target]:
        company, _ = self._get(f"dart.company {self.target.corp_code}",
                               lambda: self.src.dart.company(self._corp()))
        self.company = company or {}
        titles = [i.title for i in self.news[:PEER_HEADLINES]]
        system = (
            "너는 한국 주식시장 애널리스트다. 주어진 회사와 같은 테마·업종에서 함께 움직이는 "
            "코스피·코스닥 상장 동료 종목을 고른다. 종목 이름만 한 줄에 하나씩, 최대 "
            f"{MAX_PEERS}개 쓴다. KRX 약식 종목명으로 쓰고 번호·설명·종목코드는 붙이지 않는다. "
            "대상 회사 자신은 뺀다."
        )
        user = "\n".join([
            f"회사명: {self.target.name}",
            f"DART 업종코드(KSIC): {self.company.get('induty_code', '') or '알 수 없음'}",
            "최근 뉴스 제목:",
            *(f"- {t}" for t in titles),
        ])
        try:
            answer = self.llm.complete(system, user, quick=True)
        except Exception as exc:  # noqa: BLE001 - peers are optional; never echo the text
            self.peer_error = "LLM 추정 실패"
            self.log.write("data", "peers (LLM 추정, 상장 목록 검증)",
                           f"실패: LLM 추정 실패({type(exc).__name__})")
            return []
        by_name: dict[str, tuple[Market, dict]] = {}
        for m in ("KS", "KQ"):
            for r in self.base_rows.get(m, []):
                by_name.setdefault(str(r.get("ISU_ABBRV", "")), (m, r))  # type: ignore[arg-type]
        peers: list[Target] = []
        seen = {self.target.code}
        for name in _peer_candidates(answer):
            hit = by_name.get(name)
            listing = _ymd(hit[1].get("LIST_DD")) if hit else None
            if hit is None or listing is None:
                self.rejected_peers.append(name)
                continue
            m, r = hit
            code = str(r.get("ISU_SRT_CD", ""))
            if code in seen:
                continue
            seen.add(code)
            peers.append(Target(code, m, name, self._corp_code(code)[0], listing))
        self.log.write("data", "peers (LLM 추정, 상장 목록 검증)", _dump({
            "accepted": peers, "rejected": self.rejected_peers,
        }))
        return peers

    # ------------------------------------------------------------------ history

    def _collect_history(self) -> None:
        sym = f"{self.target.code}.{self.target.market}"
        self.history, self.history_error = self._get(
            f"yahoo.history {sym} {self.d.isoformat()} {HISTORY_DAYS}",
            lambda: self.src.history(sym, self.d, HISTORY_DAYS),
        )

    def _closes(self) -> dict[date, float]:
        if self.history is None:
            return {}
        return {ts.date(): float(v) for ts, v in self.history["close"].items()}

    # ------------------------------------------------------------------ market

    def _market(self) -> None:
        s = self.secs["market"]
        s.add("## 시장·지수 데이터")
        s.h("지수·환율·해외 지표 (Yahoo, 종가 기준)")
        s.add("미국 지수(S&P500·나스닥·필라델피아반도체)는 분석일 전날(미국 날짜) 종가까지 사용"
              " — 미국 정규장은 같은 날짜 KRX 마감 뒤에 끝난다.")
        rows = []
        closes_by_sym: dict[str, pd.Series] = {}
        for label, sym in MACRO_SYMBOLS.items():
            end = self.d - timedelta(days=1) if sym in US_SYMBOLS else self.d
            df, _ = self._get(f"yahoo.history {sym} {end.isoformat()} {MACRO_DAYS}",
                                lambda sym=sym, end=end: self.src.history(sym, end, MACRO_DAYS),
                                secs=[s], what=f"{label}({sym})")
            if df is None:
                continue
            c = df["close"]
            closes_by_sym[sym] = c
            rows.append({
                "지표": label, "심볼": sym, "기준일": c.index[-1].date().isoformat(),
                "종가": _num(float(c.iloc[-1])), "1일": _pct(_change(c, 1)),
                "5일": _pct(_change(c, 5)), "20일": _pct(_change(c, 20)),
                "60일": _pct(_change(c, 60)),
            })
        if rows:
            s.table(rows, ["지표", "심볼", "기준일", "종가", "1일", "5일", "20일", "60일"])

        # Market-wide investor flow.
        mk = self.target.market
        s.h(f"{MARKET_NAME[mk]} 시장 전체 투자자별 순매수 (원, 토스증권; 외국인은 등록·미등록 합계)")
        sessions = self._sessions(FLOW_DAYS[self.h.bucket])
        flow, _ = self._get(
            f"toss.investor_flow_market {mk} {sessions[0]}~{self.d}",
            lambda: self._toss().investor_flow_market(mk, sessions[0], self.d),
            secs=[s], what="시장 전체 투자자별 순매수",
        )
        if flow is not None:
            self._flow_tables(s, flow, ("외국인", "기관합계", "개인"), daily=5)

        s.h("시장 전반 뉴스 (헤드라인)")
        for label, query in MARKET_NEWS_QUERIES.items():
            days = NEWS_DAYS[self.h.bucket]
            items, _ = self._get(f"google_news {query} {days}d",
                                 lambda q=query, days=days: self.src.news.search(q, self.d, days),
                                 secs=[s], what=f"뉴스({label})")
            if items is None:
                continue
            s.add(f"- {label}:")
            if not items:
                s.add("  - (기사 없음)")
            for it in items[:10]:
                s.add(f"  - {it.published.date().isoformat()} {it.source}: {it.title}")

        # Relative strength vs the stock's own market index.
        s.h(f"상대 강도 ({self.target.name} vs {MARKET_NAME[self.target.market]})")
        idx = closes_by_sym.get(MARKET_INDEX[self.target.market])
        if self.history is None:
            s.miss("상대 강도", f"종목 일봉 없음: {self.history_error}")
        elif idx is None:
            s.miss("상대 강도", "시장 지수 일봉 없음")
        else:
            stock = self.history["close"]
            rs_rows = []
            for n in (20, 60):
                a, b = _change(stock, n), _change(idx, n)
                rs_rows.append({
                    "기간": f"{n}거래일", "종목 수익률": _pct(a), "지수 수익률": _pct(b),
                    "차이(%p)": "" if a is None or b is None else f"{a - b:+.2f}",
                })
            s.table(rs_rows, ["기간", "종목 수익률", "지수 수익률", "차이(%p)"])

    # ------------------------------------------------------------------ flow

    def _flow_tables(self, s: _Sec, rows: list[dict], keys: Iterable[str], daily: int) -> None:
        keys = list(keys)
        if not rows:
            s.add("(행 없음)")
            return
        cum = []
        gaps = False
        for n in sorted({5, 20, len(rows)}):
            if n > len(rows):
                continue
            part = rows[-n:]
            row: dict[str, str] = {"구간": f"최근 {n}거래일"}
            for k in keys:
                total, unknown = _known_sum(r[k] for r in part)
                row[k] = _won(total) + (f" (값 없는 날 {unknown})" if unknown and total is not None else "")
                gaps = gaps or bool(unknown)
            cum.append(row)
        s.add("누적 순매수:")
        s.table(cum, ["구간", *keys])
        if gaps:
            s.add("빈 칸·'값 없는 날'은 토스가 제공하지 않은 값이다(0이 아님). 당일 잠정치에는 "
                  "개인·기타법인·기관 세부가 없다.")
        s.add(f"일별 순매수 (최근 {min(daily, len(rows))}거래일):")
        s.table([{"date": r["date"].isoformat(), **{k: _won(r[k]) for k in keys}}
                 for r in rows[-daily:]], ["date", *keys])

    def _flow(self) -> None:
        s = self.secs["flow"]
        t = self.target
        s.add("## 수급 데이터 (토스증권 Open API)")
        sessions = self._sessions(FLOW_DAYS[self.h.bucket])
        start = sessions[0]
        keys = [k for k in FLOW_KEYS if k != "date"]

        s.h(f"투자자별 순매수 (주, 거래량 기준, {start}~{self.d})")
        s.add("토스는 종목별 투자자 거래대금(원)을 주지 않는다. 수량은 KRX·NXT 통합, "
              "외국인은 등록외국인 기준이다.")
        flow, _ = self._get(f"toss.investor_flow {t.code} {start}~{self.d}",
                            lambda: self._toss().investor_flow(t.code, start, self.d),
                            secs=[s], what="투자자별 순매수")
        if flow is not None:
            self._flow_tables(s, flow, keys, daily=SHOWN_ROWS)
            self._avg_cost(s, flow, keys)
            self._turns(s, flow)
            if self.history is not None:
                s.add(f"같은 기간 주가 등락률(20거래일): {_pct(_change(self.history['close'], 20))}")

        s.h("외국인 보유 비중")
        fh, _ = self._get(f"toss.foreign_holding {t.code} {start}~{self.d}",
                          lambda: self._toss().foreign_holding(t.code, start, self.d),
                          secs=[s], what="외국인 보유")
        if fh is not None:
            if not fh:
                s.miss("외국인 보유", "행 없음")
            else:
                stats = []
                for key in ("외국인지분율", "한도소진율"):
                    pts = [(r["date"], r[key]) for r in fh if r[key] is not None]
                    if not pts:
                        s.miss(f"외국인 {key}", "값 없음")
                        continue
                    vals = [v for _, v in pts]
                    stats.append({"항목": f"{key}(%)", "시작": f"{_num(vals[0])} ({pts[0][0]})",
                                  "최근": f"{_num(vals[-1])} ({pts[-1][0]})",
                                  "최고": _num(max(vals)), "최저": _num(min(vals)),
                                  "값 없는 날": len(fh) - len(pts)})
                if stats:
                    s.table(stats, ["항목", "시작", "최근", "최고", "최저", "값 없는 날"])
                qty = next((r for r in reversed(fh) if r["외국인보유수량"] is not None), None)
                if qty is not None:
                    s.add(f"최근 외국인 보유수량 {_won(qty['외국인보유수량'])}주 ({qty['date']})")

        s.h("공매도")
        sh, _ = self._get(f"toss.shorting {t.code} {start}~{self.d}",
                          lambda: self._toss().shorting(t.code, start, self.d),
                          secs=[s], what="공매도")
        if sh is not None:
            published = [r for r in sh if r["공매도잔고수량"] is not None]
            if published:
                p = published[-1]
                s.add(f"공매도 잔고 최신 공개일 {p['date'].isoformat()}: 수량 {_won(p['공매도잔고수량'])}주, "
                      f"금액 {_won(p['공매도잔고금액'])}원 (잔고는 며칠 늦게 공개된다)")
            else:
                s.miss("공매도 잔고", "토스 API는 공매도 잔고를 제공하지 않음")
            s.add(f"공매도 거래 (최근 {min(SHOWN_ROWS, len(sh))}거래일, 비중은 시간외 포함 당일 거래 대비):")
            s.table([{"date": r["date"].isoformat(), "거래량": _won(r["공매도거래량"]),
                      "거래대금": _won(r["공매도거래대금"]),
                      "거래량 비중": _pct(r.get("공매도거래량비중"), signed=False),
                      "거래대금 비중": _pct(r.get("공매도거래대금비중"), signed=False)}
                     for r in sh[-SHOWN_ROWS:]],
                    ["date", "거래량", "거래대금", "거래량 비중", "거래대금 비중"])

        self._lending(s, start)
        self._program(s, start)
        self._credit(s, start)

    def _lending(self, s: _Sec, start: date) -> None:
        t = self.target
        s.h("대차잔고 (기관 간 대차거래)")
        rows, _ = self._get(f"toss.securities_lending {t.code} {start}~{self.d}",
                            lambda: self._toss().securities_lending(t.code, start, self.d),
                            secs=[s], what="대차잔고")
        if rows is None:
            return
        if not rows:
            s.miss("대차잔고", "행 없음")
            return
        known = [r for r in rows if r["대차잔고수량"] is not None]
        if not known:
            s.miss("대차잔고", "잔고 값 없음")
            return
        last = known[-1]
        s.add(f"최근 대차잔고 {last['date'].isoformat()}: {_won(last['대차잔고수량'])}주, "
              f"{_won(last['대차잔고금액'])}원")
        changes = []
        for n in (5, 20):
            if len(known) > n:
                base = known[-1 - n]["대차잔고수량"]
                diff = last["대차잔고수량"] - base
                changes.append(f"{n}거래일 전 대비 {diff:+,}주"
                               + (f" ({diff / base * 100:+.2f}%)" if base else ""))
        if changes:
            s.add("잔고 변화(값 있는 날 기준): " + ", ".join(changes))
        s.add(f"일별 대차 (최근 {min(SHOWN_ROWS, len(rows))}거래일, 주; 빈 칸은 값 없음):")
        s.table([{"date": r["date"].isoformat(), "체결": _won(r["대차체결"]),
                  "상환": _won(r["대차상환"]), "잔고": _won(r["대차잔고수량"])}
                 for r in rows[-SHOWN_ROWS:]], ["date", "체결", "상환", "잔고"])

    def _program(self, s: _Sec, start: date) -> None:
        t = self.target
        s.h("프로그램 매매 (주, KRX만 집계·NXT 제외)")
        rows, _ = self._get(f"toss.program_trading {t.code} {start}~{self.d}",
                            lambda: self._toss().program_trading(t.code, start, self.d),
                            secs=[s], what="프로그램 매매")
        if rows is None:
            return
        if not rows:
            s.miss("프로그램 매매", "행 없음")
            return
        if all(r["차익순매수"] is None and r["비차익순매수"] is None for r in rows):
            s.miss("프로그램 매매", "순매수 값 없음")
            return

        def total(r: dict) -> int | None:
            a, b = r["차익순매수"], r["비차익순매수"]
            return None if a is None or b is None else a + b

        def cell(values: list[int | None]) -> str:
            v, unknown = _known_sum(values)
            return _won(v) + (f" (값 없는 날 {unknown})" if unknown and v is not None else "")

        cum = []
        for n in sorted({5, 20, len(rows)}):
            if n > len(rows):
                continue
            part = rows[-n:]
            arb = [r["차익순매수"] for r in part]
            non = [r["비차익순매수"] for r in part]
            # 합계 = 차익 + 비차익 over days where both are known.
            cum.append({"구간": f"최근 {n}거래일", "차익": cell(arb), "비차익": cell(non),
                        "합계": cell([total(r) for r in part])})
        s.add("누적 순매수 (값 없는 날은 0이 아니라 제외):")
        s.table(cum, ["구간", "차익", "비차익", "합계"])
        s.add(f"일별 순매수 (최근 {min(SHOWN_ROWS, len(rows))}거래일, 빈 칸은 값 없음):")
        s.table([{"date": r["date"].isoformat(), "차익": _won(r["차익순매수"]),
                  "비차익": _won(r["비차익순매수"]), "합계": _won(total(r))}
                 for r in rows[-SHOWN_ROWS:]], ["date", "차익", "비차익", "합계"])

    def _credit(self, s: _Sec, start: date) -> None:
        t = self.target
        s.h("신용잔고 (융자·대주, 다음 영업일 새벽 공개)")
        rows, _ = self._get(f"toss.credit_balance {t.code} {start}~{self.d}",
                            lambda: self._toss().credit_balance(t.code, start, self.d),
                            secs=[s], what="신용잔고")
        if rows is None:
            return
        loans = [r for r in rows if r["융자잔고"] is not None]
        if not loans:
            s.miss("신용잔고", "행 없음" if not rows else "융자 잔고 값 없음")
            return
        last = loans[-1]
        s.add(f"신용잔고 최신일 {last['date'].isoformat()}: 융자잔고 {_won(last['융자잔고'])}주 "
              f"(상장주식 대비 {_pct(last['융자잔고율'], signed=False)}, 공여율 "
              f"{_pct(last['융자공여율'], signed=False)})")
        changes = []
        for n in (5, 20):
            if len(loans) > n:
                changes.append(f"{n}거래일 전 대비 {last['융자잔고'] - loans[-1 - n]['융자잔고']:+,}주")
        if changes:
            s.add("융자잔고 변화: " + ", ".join(changes))
        s.add(f"일별 신용 (최근 {min(SHOWN_ROWS, len(rows))}거래일, 주; 빈 칸은 값 없음):")
        s.table([{"date": r["date"].isoformat(), "융자신규": _won(r["융자신규"]),
                  "융자상환": _won(r["융자상환"]), "융자잔고": _won(r["융자잔고"]),
                  "대주잔고": _won(r["대주잔고"])} for r in rows[-SHOWN_ROWS:]],
                ["date", "융자신규", "융자상환", "융자잔고", "대주잔고"])

    def _avg_cost(self, s: _Sec, flow: list[dict], keys: list[str]) -> None:
        closes = self._closes()
        s.add("")
        s.add("주체별 추정 평균 단가 (추정치: 순매수일만 사용, Σ(순매수수량 × 당일 종가) ÷ "
              "Σ순매수수량, 값 없는 날 제외, 종가는 수정주가 아님):")
        if not closes:
            s.miss("주체별 추정 평균 단가", f"종목 일봉 없음: {self.history_error}")
            return
        out = []
        for k in keys:
            amt = qty = 0.0
            days = 0
            for r in flow:
                c = closes.get(r["date"])
                v = r[k]
                if v is not None and v > 0 and c:
                    amt += v * c
                    qty += v
                    days += 1
            out.append({"주체": k, "순매수일 수": days, "순매수수량 합": _won(qty),
                        "추정 평균 단가": _won(amt / qty) if qty else ""})
        s.table(out, ["주체", "순매수일 수", "순매수수량 합", "추정 평균 단가"])

    def _turns(self, s: _Sec, flow: list[dict]) -> None:
        s.add("")
        s.add("수급 전환 지점 (5거래일 누적 순매수의 부호가 바뀐 날, 값 없는 날이 낀 구간은 건너뜀):")
        for k in ("외국인", "기관합계"):
            vals = [r[k] for r in flow]
            turns = []
            prev = None
            for i in range(4, len(vals)):
                window = vals[i - 4:i + 1]
                if any(v is None for v in window):
                    continue
                total = sum(window)
                sign = (total > 0) - (total < 0)
                if prev is not None and sign and prev and sign != prev:
                    turns.append(f"{flow[i]['date'].isoformat()}({'매수' if sign > 0 else '매도'} 전환)")
                if sign:
                    prev = sign
            s.add(f"- {k}: {', '.join(turns[-5:]) if turns else '전환 없음'}")

    # ------------------------------------------------------------------ theme

    def _theme(self) -> None:
        s = self.secs["theme"]
        t = self.target
        s.add("## 테마·재료 데이터")
        s.add(f"{PEER_NOTE}.")
        if self.peer_error:
            s.miss("동료 종목", self.peer_error)
        elif self.peers:
            s.add("동료 종목: " + ", ".join(f"{p.name}({p.code}.{p.market})" for p in self.peers))
        else:
            s.add("동료 종목: 검증된 종목 없음")
        if self.rejected_peers:
            s.add("상장 목록에 없어 제외한 LLM 추정: " + ", ".join(self.rejected_peers))

        group = [t, *self.peers]
        codes = {p.code for p in group}
        sessions = self._sessions(CROSS_SECTION_DAYS + 1)
        days: dict[date, dict] = {}
        errors: list[str] = []
        unpublished: list[date] = []
        for day in sessions:
            info: dict[str, Any] = {"total": {}, "rows": {}}
            for mk in ("KS", "KQ"):
                rows, err = self._get(
                    f"krx_openapi.daily_trades {mk} {day.isoformat()} (발췌)",
                    lambda mk=mk, day=day: self.src.krx.daily_trades(mk, day),
                    excerpt=lambda rs: {"rows": len(rs), "selected": [
                        r for r in rs if r.get("ISU_CD") in codes]},
                )
                if err is not None:
                    errors.append(f"{day} {mk}: {err}")
                    continue
                if not rows:
                    unpublished.append(day)
                    continue
                vals = [(str(r.get("ISU_CD")), _int_or_none(r.get("ACC_TRDVAL")) or 0) for r in rows]
                info["total"][mk] = sum(v for _, v in vals)
                for r in rows:
                    if r.get("ISU_CD") in codes:
                        info["rows"][r["ISU_CD"]] = r
                if mk == t.market:
                    ranked = sorted(vals, key=lambda cv: cv[1], reverse=True)
                    pos = next((i for i, (c, _) in enumerate(ranked, 1) if c == t.code), None)
                    info["rank"] = (pos, len(ranked))
            if info["total"]:
                days[day] = info

        s.h(f"거래대금 순위 ({MARKET_NAME[t.market]} 전체 중, 최근 {CROSS_SECTION_DAYS}거래일)")
        if errors:
            s.miss("일별 시세", f"{len(errors)}건 조회 실패, 예: {errors[0]}")
        if unpublished:
            s.add(f"시세 미공개(빈 응답) 날짜: {', '.join(sorted({d.isoformat() for d in unpublished}))}")
        shown = [d for d in sessions if d in days][-CROSS_SECTION_DAYS:]
        rank_rows = []
        values = []
        for day in shown:
            info = days[day]
            row = info["rows"].get(t.code)
            pos, n = info.get("rank", (None, None))
            val = _int_or_none(row.get("ACC_TRDVAL")) if row else None
            if val is not None:
                values.append(val)
            rank_rows.append({
                "date": day.isoformat(), "거래대금": _won(val), "순위": pos or "", "종목 수": n or "",
                "시장 거래대금 합": _won(info["total"].get(t.market)),
            })
        if rank_rows:
            s.table(rank_rows, ["date", "거래대금", "순위", "종목 수", "시장 거래대금 합"])
            if len(values) >= 2 and sum(values[:-1]):
                avg = sum(values[:-1]) / len(values[:-1])
                s.add(f"최근일 거래대금 / 직전 {len(values) - 1}거래일 평균: {values[-1] / avg:.2f}배")
        elif not errors:
            s.miss("거래대금 순위", "조회 구간 시세 없음")

        both = [d for d in shown if len(days[d]["total"]) == 2]
        s.h("테마 자금 쏠림 (동료+자신 거래대금 합 / 코스피+코스닥 전체 거래대금)")
        if both:
            share_rows = []
            for day in both:
                info = days[day]
                g = sum(_int_or_none(r.get("ACC_TRDVAL")) or 0 for r in info["rows"].values())
                total = sum(info["total"].values())
                share_rows.append({"date": day.isoformat(), "테마 거래대금": _won(g),
                                   "전체 거래대금": _won(total),
                                   "비중": _pct(g / total * 100 if total else None, signed=False)})
            s.table(share_rows, ["date", "테마 거래대금", "전체 거래대금", "비중"])
        else:
            s.miss("테마 자금 쏠림", "양 시장 시세가 모두 있는 날 없음")

        s.h("테마 내 위치 (최근일 기준: 대장주/2등주/후발주 판단 자료)")
        if shown:
            last, first = shown[-1], sessions[0]
            self.latest_quote_day = last
            self.latest_quote = dict(days[last]["rows"])
            base = days.get(first, {}).get("rows", {})
            pos_rows = []
            for p in group:
                r = days[last]["rows"].get(p.code)
                if r is None:
                    pos_rows.append({"종목": p.name, "코드": p.code, "시장": p.market})
                    continue
                c, c0 = _int_or_none(r.get("TDD_CLSPRC")), _int_or_none(base.get(p.code, {}).get("TDD_CLSPRC"))
                pos_rows.append({
                    "종목": p.name, "코드": p.code, "시장": p.market, "종가": _won(c),
                    "1일 등락률": _pct(to_float(r.get("FLUC_RT"))),
                    f"{CROSS_SECTION_DAYS}거래일 등락률": _pct((c / c0 - 1) * 100 if c and c0 else None),
                    "거래대금": _won(_int_or_none(r.get("ACC_TRDVAL"))),
                    "시가총액": _won(_int_or_none(r.get("MKTCAP"))),
                    "_mc": _int_or_none(r.get("MKTCAP")) or 0,
                    "_tv": _int_or_none(r.get("ACC_TRDVAL")) or 0,
                })
            for key, col in (("_mc", "시총 순위"), ("_tv", "거래대금 순위")):
                order = sorted((r for r in pos_rows if key in r), key=lambda r: r[key], reverse=True)
                for i, r in enumerate(order, 1):
                    r[col] = f"{i}/{len(order)}"
            s.add(f"기준일 {last.isoformat()}")
            s.table(pos_rows, ["종목", "코드", "시장", "종가", "1일 등락률",
                               f"{CROSS_SECTION_DAYS}거래일 등락률", "거래대금", "시가총액",
                               "시총 순위", "거래대금 순위"])
        else:
            s.miss("테마 내 위치", "시세 없음")

        self._sectors(s, sessions[0])

        s.h(f"종목 뉴스 헤드라인 (최근 {NEWS_DAYS[self.h.bucket]}일, 본문 없음)")
        if self.news_error is not None:
            s.miss("뉴스", self.news_error)
        else:
            if len(self.news) < MIN_NEWS:
                s.miss("뉴스", f"기사 수 부족({len(self.news)}건)")
            if self.news:
                s.table([{"date": i.published.date().isoformat(), "언론사": i.source, "제목": i.title}
                         for i in self.news[:SHOWN_ROWS]], ["date", "언론사", "제목"])

    def _sectors(self, s: _Sec, base_day: date) -> None:
        s.h(f"KRX 업종 지수 {CROSS_SECTION_DAYS}거래일 등락률 (섹터 쏠림)")
        out = []
        for mk in ("KS", "KQ"):
            now, _ = self._get(f"krx_openapi.index_daily {mk} {self.d.isoformat()}",
                                 lambda mk=mk: self.src.krx.index_daily(mk, self.d),
                                 secs=[s], what=f"{MARKET_NAME[mk]} 업종 지수")
            then, _ = (None, None) if now is None else self._get(
                f"krx_openapi.index_daily {mk} {base_day.isoformat()}",
                lambda mk=mk: self.src.krx.index_daily(mk, base_day),
                secs=[s], what=f"{MARKET_NAME[mk]} 업종 지수({base_day})")
            if not now or not then:
                continue
            old = {r.get("IDX_NM"): to_float(r.get("CLSPRC_IDX")) for r in then}
            for r in now:
                name = str(r.get("IDX_NM", ""))
                if re.match(r"^(코스피|코스닥)(\s|$)", name) or _matches(name, ("대형주", "중형주", "소형주")):
                    continue  # broad / size indices, not sectors
                c, c0 = to_float(r.get("CLSPRC_IDX")), old.get(name)
                if c and c0:
                    out.append({"지수": name, "시장": MARKET_NAME[mk], "종가": _num(c),
                                "등락률": (c / c0 - 1) * 100})
        if not out:
            s.miss("업종 지수", "비교할 업종 지수 없음")
            return
        out.sort(key=lambda r: r["등락률"], reverse=True)
        fmt = [{**r, "등락률": _pct(r["등락률"])} for r in out]
        cols = ["지수", "시장", "종가", "등락률"]
        if len(fmt) <= 10:
            s.add(f"전체 ({base_day}→{self.d}, 등락률 순):")
            s.table(fmt, cols)
            return
        s.add(f"상위 5 ({base_day}→{self.d}):")
        s.table(fmt[:5], cols)
        s.add("하위 5:")
        s.table(fmt[-5:][::-1], cols)

    # ------------------------------------------------------------------ technical

    def _technical(self) -> None:
        s = self.secs["technical"]
        s.add("## 차트·거래량 데이터 (Yahoo 일봉)")
        s.add(UNADJUSTED_NOTE)
        if self.history is None:
            s.miss("일봉", self.history_error or "조회 실패")
            return
        df = self.history
        last = df.index[-1].date()
        if last != self.d:
            s.add(f"분석일 일봉 없음: 마지막 행 {last.isoformat()}")
        elif self.resolved.unconfirmed:
            s.add("마지막 행(분석일)은 장중 값일 수 있음")
        frames = {"단기": ("D",), "중기": ("D", "W"), "장기": ("W", "M")}[self.h.bucket]
        labels = {"D": "일봉", "W": "주봉", "M": "월봉"}
        for tf in frames:
            summary = technical_summary(df, tf)
            s.h(f"{labels[tf]} 지표")
            hits = summary.pop("limit_hits")
            s.table([{"지표": k, "값": _num(v)} for k, v in summary.items()], ["지표", "값"])
            if tf == "D":
                if isinstance(hits, str):
                    s.add(f"상한가·하한가(±30%) 근접 이력: {hits}")
                elif hits:
                    s.add("상한가·하한가(±30%) 근접 이력(최근 60거래일): " + ", ".join(
                        f"{h['date']} {h['change_pct']:+.2f}%" for h in hits))
                else:
                    s.add("상한가·하한가(±30%) 근접 이력(최근 60거래일): 없음")
        s.h("지지·저항 참고 (고가·저가)")
        hl = []
        for n in (20, 60, 120):
            part = df.iloc[-n:]
            hl.append({"구간": f"최근 {n}거래일", "최고가": _won(float(part["high"].max())),
                       "최저가": _won(float(part["low"].min()))})
        s.table(hl, ["구간", "최고가", "최저가"])
        s.h("최근 10거래일 일봉")
        s.table([{"date": ts.date().isoformat(), "시가": _won(r["open"]), "고가": _won(r["high"]),
                  "저가": _won(r["low"]), "종가": _won(r["close"]), "거래량": _won(r["volume"])}
                 for ts, r in df.iloc[-10:].iterrows()],
                ["date", "시가", "고가", "저가", "종가", "거래량"])

    # ------------------------------------------------------------------ filings

    def _collect_filings(self) -> None:
        start = self.d - timedelta(days=GOVERNANCE_DAYS)
        self.filings, self.filings_error = self._get(
            f"dart.filings {self.target.corp_code} {start}~{self.d}",
            lambda: self.src.dart.filings(self._corp(), start, self.d, as_of=self.d),
        )

    def _filings_since(self, days: int) -> list[dict]:
        start = self.d - timedelta(days=days)
        out = []
        for f in self.filings or []:
            fd = _filing_date(f)
            if fd is not None and start <= fd <= self.d:
                out.append(f)
        return out

    def _filing_rows(self, rows: list[dict]) -> list[dict]:
        return [{"접수일": str(_filing_date(f) or ""), "보고서명": f.get("report_nm", ""),
                 "제출인": f.get("flr_nm", ""), "정정": "정정" if f.get("is_correction") else "",
                 "접수번호": f.get("rcept_no", "")} for f in rows]

    # ------------------------------------------------------------------ disclosure

    _MAJOR_FIELDS: ClassVar[dict[str, list[tuple[str, str]]]] = {
        "CB": [("bd_tm", "회차"), ("bd_knd", "종류"), ("bd_fta", "권면총액"), ("cv_prc", "전환가"),
               ("act_mktprcfl_cvprc_lwtrsprc", "리픽싱 최저조정가"), ("cvisstk_cnt", "전환가능주식수"),
               ("cvrqpd_bgd", "전환청구 시작"), ("cvrqpd_edd", "전환청구 종료")],
        "BW": [("bd_tm", "회차"), ("bd_knd", "종류"), ("bd_fta", "권면총액"), ("ex_prc", "행사가"),
               ("act_mktprcfl_cvprc_lwtrsprc", "리픽싱 최저조정가"), ("nstk_isstk_cnt", "행사가능주식수"),
               ("expd_bgd", "행사 시작"), ("expd_edd", "행사 종료")],
        "유상증자": [("nstk_ostk_cnt", "신주(보통)"), ("nstk_estk_cnt", "신주(기타)"),
                 ("ic_mthn", "증자방식"), ("fdpp_fclt", "시설자금"), ("fdpp_op", "운영자금")],
        "자기주식취득": [("aqpln_stk_ostk", "취득예정(보통)"), ("aqpln_prc_ostk", "취득예정금액(보통)"),
                    ("aq_pp", "목적"), ("aq_mth", "방법"), ("aqexpd_bgd", "기간 시작"),
                    ("aqexpd_edd", "기간 종료")],
        "자기주식처분": [("dppln_stk_ostk", "처분예정(보통)"), ("dppln_prc_ostk", "처분예정금액(보통)"),
                    ("dp_pp", "목적"), ("dpprpd_bgd", "기간 시작"), ("dpprpd_edd", "기간 종료")],
    }
    _COMPARE_SKIP: ClassVar[frozenset[str]] = frozenset({"rcept_no", "corp_cls", "corp_code", "corp_name"})

    def _disclosure(self) -> None:
        s = self.secs["disclosure"]
        days = DISCLOSURE_DAYS[self.h.bucket]
        start = self.d - timedelta(days=days)
        s.add(f"## 공시 데이터 (DART, 접수일 {start}~{self.d})")
        if self.filings is None:
            s.miss("공시 목록", self.filings_error or "조회 실패")
        window = self._filings_since(days)
        corrections = {f.get("rcept_no") for f in window if f.get("is_correction")}

        s.h(f"공시 목록 ({len(window)}건)")
        if self.filings is not None:
            s.table(self._filing_rows(window), ["접수일", "보고서명", "제출인", "정정", "접수번호"])

        s.h("주요사항보고서")
        listed = _int_or_none(self.target_row.get("LIST_SHRS"))
        structured: set[str] = set()
        for kind, fields in self._MAJOR_FIELDS.items():
            rows, _ = self._get(
                f"dart.major_report {kind} {self.target.corp_code} {start}~{self.d}",
                lambda kind=kind: self.src.dart.major_report(kind, self._corp(), start, self.d,
                                                             as_of=self.d),
                secs=[s], what=f"주요사항보고서({kind})")
            if rows is None:
                continue
            self.major[kind] = rows
            structured |= {str(r.get("rcept_no")) for r in rows}
            if not rows:
                s.add(f"- {kind}: 없음")
                continue
            out = []
            for r in rows:
                row = {"접수일": str(_filing_date(r) or ""), "접수번호": r.get("rcept_no", ""),
                       "정정": "정정" if r.get("rcept_no") in corrections else ""}
                row.update({label: r.get(f, "") for f, label in fields})
                if kind in ("CB", "BW"):
                    cnt = _int_or_none(r.get("cvisstk_cnt" if kind == "CB" else "nstk_isstk_cnt"))
                    row["잠재 희석률"] = (_pct(cnt / listed * 100, signed=False)
                                     if cnt is not None and listed else "")
                out.append(row)
            cols = ["접수일", "접수번호", "정정", *(label for _, label in fields)]
            if kind in ("CB", "BW"):
                cols.append("잠재 희석률")
                s.add(f"{kind} (잠재 희석률 = 전환·행사 가능 주식수 / 상장주식수 {_won(listed)}주):")
            else:
                s.add(f"{kind}:")
            s.table(out, cols)
        cancels = [f for f in window if "주식소각결정" in str(f.get("report_nm", ""))]
        s.add(f"자기주식소각: {len(cancels)}건" + (
            "" if not cancels else " — " + ", ".join(f"{_filing_date(f)} {f.get('report_nm')}" for f in cancels)))

        s.h("최대주주 변경·합병·분할")
        events = [f for f in window if _matches(str(f.get("report_nm", "")), ("최대주주변경", "최대주주 변경", "합병", "분할"))]
        if events:
            s.table(self._filing_rows(events), ["접수일", "보고서명", "제출인", "정정", "접수번호"])
            s.add("물적분할 여부는 보고서명에 '물적분할'이 없으면 본문 확인 필요")
        else:
            s.add("해당 공시 없음")

        s.h("5% 대량보유 상황보고")
        self._holder_table(s, "dart.major_holders", lambda: self.src.dart.major_holders(self._corp(), as_of=self.d),
                           start, "대량보유",
                           [("repror", "보고자"), ("stkqy", "보유주식수"), ("stkrt", "보유비율"),
                            ("stkqy_irds", "증감수량"), ("stkrt_irds", "증감비율"), ("report_resn", "사유")])
        s.h("임원·주요주주 소유보고")
        self._holder_table(s, "dart.exec_holdings", lambda: self.src.dart.exec_holdings(self._corp(), as_of=self.d),
                           start, "임원·주요주주 소유",
                           [("repror", "보고자"), ("isu_exctv_ofcps", "직위"),
                            ("sp_stock_lmp_cnt", "보유주식수"), ("sp_stock_lmp_irds_cnt", "증감"),
                            ("sp_stock_lmp_rate", "보유비율"), ("sp_stock_lmp_irds_rate", "증감비율")])

        s.h("잠정실적·영업실적 공시")
        prelim = [f for f in window if _matches(str(f.get("report_nm", "")), ("잠정", "영업실적"))]
        if prelim:
            s.table(self._filing_rows(prelim), ["접수일", "보고서명", "제출인", "정정", "접수번호"])
        else:
            s.add("해당 공시 없음")

        self._corrections(s, window, corrections, structured)

        s.h("배당 관련 공시 (최근 1년, 배당기준일은 공시 본문 확인)")
        divs = [f for f in self._filings_since(365) if "배당" in str(f.get("report_nm", ""))]
        if divs:
            s.table(self._filing_rows(divs), ["접수일", "보고서명", "제출인", "정정", "접수번호"])
        elif self.filings is not None:
            s.add("해당 공시 없음")

        s.add("")
        s.add(CRITICAL_HEADER)
        crit = []
        for f in window:
            label = _critical_label(str(f.get("report_nm", "")))
            if label:
                crit.append(f"- {_filing_date(f)} {f.get('report_nm')} [{label}] (접수번호 {f.get('rcept_no')})")
        if self.filings is None:
            crit.append(f"- (공시 목록 조회 실패: {self.filings_error})")
        s.add(*(crit or ["- 없음"]))

    def _holder_table(self, s: _Sec, source: str, fn: Callable[[], list[dict]], start: date,
                      what: str, fields: list[tuple[str, str]]) -> None:
        rows, _ = self._get(f"{source} {self.target.corp_code}", fn, secs=[s], what=what)
        if rows is None:
            return
        rows = [r for r in rows if (fd := _filing_date(r)) is not None and start <= fd <= self.d]
        rows.sort(key=lambda r: _filing_date(r) or date.min, reverse=True)
        if not rows:
            s.add("조회 구간에 보고 없음")
            return
        s.table([{"접수일": str(_filing_date(r)), **{lab: r.get(f, "") for f, lab in fields}}
                 for r in rows[:SHOWN_ROWS]], ["접수일", *(lab for _, lab in fields)])
        if len(rows) > SHOWN_ROWS:
            s.add(f"(최근 {SHOWN_ROWS}건만 표시, 전체 {len(rows)}건)")

    def _corrections(self, s: _Sec, window: list[dict], corrections: set, structured: set[str]) -> None:
        s.h("정정공시 비교")
        lines: list[str] = []
        for kind, rows in self.major.items():
            originals = [r for r in rows if r.get("rcept_no") not in corrections]
            fixes = [r for r in rows if r.get("rcept_no") in corrections]
            for fix_group in self._group_corrections(originals, fixes):
                orig, final = fix_group
                if orig is None:
                    lines.append(f"- {kind} 정정 {final.get('rcept_no')}: 원보고 레코드 없음(조회 범위 밖일 수 있음)")
                    continue
                changed = [{"항목": k, "원보고": orig.get(k, ""), "최종 정정": final.get(k, "")}
                           for k in sorted(set(orig) | set(final))
                           if k not in self._COMPARE_SKIP and orig.get(k) != final.get(k)]
                lines.append(f"- {kind}: 원보고 {orig.get('rcept_no')} → 최종 정정 {final.get('rcept_no')}")
                if changed:
                    lines.append(md_table(changed, ["항목", "원보고", "최종 정정"]))
                else:
                    lines.append("  바뀐 필드 없음")
        for f in window:
            if f.get("is_correction") and str(f.get("rcept_no")) not in structured:
                lines.append(f"- {_filing_date(f)} {f.get('report_nm')}: 정정 내용 비교 불가(본문 확인 필요)")
        s.add(*(lines or ["정정공시 없음"]))

    @staticmethod
    def _group_corrections(originals: list[dict], fixes: list[dict]):
        """Pair each corrected record with its original (same 회차 when present)."""
        def key(r: dict) -> Any:
            return r.get("bd_tm")
        by_orig: dict[int, list[dict]] = {}
        orphans: list[dict] = []
        for fx in sorted(fixes, key=lambda r: str(r.get("rcept_no"))):
            cands = [o for o in originals
                     if str(o.get("rcept_no")) < str(fx.get("rcept_no")) and key(o) == key(fx)]
            if not cands:
                orphans.append(fx)
                continue
            orig = max(cands, key=lambda r: str(r.get("rcept_no")))
            by_orig.setdefault(id(orig), [orig]).append(fx)
        for group in by_orig.values():
            yield group[0], group[-1]
        for fx in orphans:
            yield None, fx

    # ------------------------------------------------------------------ fundamentals

    def _reports(self, corp: str, secs: list[_Sec]) -> tuple[tuple[int, list[dict]] | None,
                                                            tuple[int, str, str, list[dict]] | None,
                                                            str | None]:
        """(latest annual (year, rows), latest quarter of the following year (year, code, label,
        rows), fetch failure reason). A failure is already recorded in `secs`.

        Only quarters of annual year + 1 are used: with an older annual (FY Y-1 not filed by
        the analysis date) a year-Y quarter would make a mixed, not a trailing, sum."""
        annual = None
        for year in (self.d.year - 1, self.d.year - 2):
            rows, err = self._get(f"dart.financials {corp} {year} 11011",
                                  lambda year=year: self.src.dart.financials(corp, year, "11011", as_of=self.d),
                                  secs=secs, what=f"재무제표({year} 사업보고서)")
            if err is not None:
                return None, None, err
            if rows:
                annual = (year, rows)
                break
        if annual is None:
            return None, None, None
        quarter = None
        year = annual[0] + 1
        for code, label in _QUARTER_CODES:
            rows, _ = self._get(f"dart.financials {corp} {year} {code}",
                                lambda code=code: self.src.dart.financials(corp, year, code, as_of=self.d),
                                secs=secs, what=f"재무제표({year} {label})")
            if rows:
                quarter = (year, code, label, rows)
                break
        return annual, quarter, None

    def _valuation(self, mktcap: int | None, year: int, annual: list[dict],
                   quarter: tuple[str, list[dict]] | None) -> dict:
        """`quarter` is (reprt_code, rows) of the latest quarterly report of `year` + 1."""
        ttm, how = _ttm_parent_ni(year, annual, quarter)
        eq = _amount(quarter[1], "지배주주지분", "thstrm_amount") if quarter else None
        if eq is None:
            eq = _amount(annual, "지배주주지분", "thstrm_amount")
        if eq is None:
            eq = _amount(quarter[1], "자본총계", "thstrm_amount") if quarter else None
            if eq is None:
                eq = _amount(annual, "자본총계", "thstrm_amount")
            if eq is not None:
                how += "; 지배자본 없음: 자본총계 사용"
        per = mktcap / ttm if mktcap and ttm and ttm > 0 else None
        pbr = mktcap / eq if mktcap and eq and eq > 0 else None
        return {"ttm": ttm, "how": how, "equity": eq, "per": per, "pbr": pbr}

    def _fundamentals(self) -> None:
        s = self.secs["fundamentals"]
        t = self.target
        s.add("## 펀더멘털·밸류 데이터 (DART, 연결 우선·없으면 별도)")
        if self.company:
            s.add(f"기업개황: 업종코드 {self.company.get('induty_code', '')}, "
                  f"결산월 {self.company.get('acc_mt', '')}")
        try:
            corp = self._corp()
        except _Unavailable as exc:
            s.miss("재무제표", str(exc))
            corp = ""
        annual, quarter, failed = self._reports(corp, [s]) if corp else (None, None, None)

        s.h("재무 추이 (원)")
        if annual is None:
            if corp and failed is None:
                s.miss("재무제표", "최근 2년 사업보고서 없음")
        else:
            year, rows = annual
            if year < self.d.year - 1:
                s.add(f"FY{self.d.year - 1} 사업보고서 없음 또는 분석일 기준 미공개")
            cols = [str(year - 2), str(year - 1), str(year)]
            fields = ["bfefrmtrm_amount", "frmtrm_amount", "thstrm_amount"]
            table = []
            vals: dict[str, list[int | None]] = {}
            for label, *_ in _ACCOUNTS:
                v = [_amount(rows, label, f) for f in fields]
                vals[label] = v
                if any(x is not None for x in v):
                    table.append({"항목": label, **{c: _won(x) for c, x in zip(cols, v, strict=True)}})

            def ratio(a: str, b: str, i: int) -> str:
                x, y = vals[a][i], vals[b][i]
                return _pct(x / y * 100, signed=False) if x is not None and y else ""
            for name, a, b in (("영업이익률", "영업이익", "매출액"), ("순이익률", "당기순이익", "매출액"),
                               ("부채비율", "부채총계", "자본총계")):
                table.append({"항목": name, **{c: ratio(a, b, i) for i, c in enumerate(cols)}})
            s.add(f"{year} 사업보고서 기준 3개 사업연도:")
            s.table(table, ["항목", *cols])
            if quarter:
                qy, _, qlabel, qrows = quarter
                q = []
                for label, *_ in _ACCOUNTS:
                    v = [_amount(qrows, label, f) for f in ("thstrm_amount", "thstrm_add_amount", "frmtrm_add_amount")]
                    if any(x is not None for x in v):
                        q.append({"항목": label, "당기(3개월/기말)": _won(v[0]), "당기 누적": _won(v[1]),
                                  "전년 동기 누적": _won(v[2])})
                s.add(f"최근 분기: {qy} {qlabel}")
                s.table(q, ["항목", "당기(3개월/기말)", "당기 누적", "전년 동기 누적"])
            else:
                s.add("사업보고서 이후 분기보고서 없음")

        s.h("밸류에이션 (PER = 시총 / 최근 4분기 지배순이익, PBR = 시총 / 지배자본)")
        if self.peer_error:
            s.miss("동료 종목", self.peer_error)
        quote = self.latest_quote.get(t.code)
        mktcap = _int_or_none(quote.get("MKTCAP")) if quote else None
        close = _int_or_none(quote.get("TDD_CLSPRC")) if quote else None
        if mktcap is None:
            s.miss("시가총액", "KRX 일별 시세 없음")
        val_rows = []
        if annual is not None:
            v = self._valuation(mktcap, annual[0], annual[1], (quarter[1], quarter[3]) if quarter else None)
            val_rows.append(self._val_row(t, mktcap, v))
            s.add(f"최근 4분기 지배순이익 {_won(v['ttm'])}원 ({v['how']}), 지배자본 {_won(v['equity'])}원, "
                  f"시총 기준일 {self.latest_quote_day}")
            # Dividend yield.
            dv, _ = self._get(f"dart.dividends {corp} {annual[0]} 11011",
                              lambda: self.src.dart.dividends(corp, annual[0], "11011", as_of=self.d),
                              secs=[s], what="배당")
            dps = None
            for r in dv or []:
                if "주당 현금배당금" in str(r.get("se", "")) and "보통" in str(r.get("stock_knd", "")):
                    dps = _int_or_none(r.get("thstrm"))
                    break
            if dps is not None and close:
                s.add(f"배당수익률 {_pct(dps / close * 100, signed=False)} "
                      f"({annual[0]} 주당 현금배당금 {_won(dps)}원 / 종가 {_won(close)}원)")
            elif dv is not None:
                s.miss("배당수익률", "주당 현금배당금(보통주) 또는 종가 없음")
        # Peers, using the same report periods as the target.
        if annual is not None:
            for p in self.peers:
                pq = self.latest_quote.get(p.code)
                pmc = _int_or_none(pq.get("MKTCAP")) if pq else None
                if not p.corp_code:
                    val_rows.append({"종목": p.name, "비고": "DART 고유번호 없음"})
                    continue
                pa, err = self._get(f"dart.financials {p.corp_code} {annual[0]} 11011",
                                    lambda p=p: self.src.dart.financials(p.corp_code, annual[0], "11011", as_of=self.d))
                if err is not None or not pa:
                    note = f"조회 실패({err})" if err is not None else f"{annual[0]} 사업보고서 없음"
                    val_rows.append({"종목": p.name, "시가총액": _won(pmc), "비고": note})
                    continue
                pqr, qerr = None, None
                if quarter:
                    pqr, qerr = self._get(f"dart.financials {p.corp_code} {quarter[0]} {quarter[1]}",
                                          lambda p=p: self.src.dart.financials(p.corp_code, quarter[0], quarter[1], as_of=self.d))
                row = self._val_row(p, pmc, self._valuation(pmc, annual[0], pa,
                                                            (quarter[1], pqr) if quarter and pqr else None))
                if quarter and not pqr:
                    gap = f"{quarter[0]} {quarter[2]} " + (f"조회 실패({qerr})" if qerr else "없음")
                    row["비고"] = "; ".join(x for x in (row["비고"], f"{gap}: 최근 연간") if x)
                val_rows.append(row)
        if val_rows:
            s.add("동료 종목도 같은 방식·같은 보고 기간으로 계산:" if self.peers else "")
            s.table(val_rows, ["종목", "시가총액", "최근 4분기 지배순이익", "PER", "PBR", "비고"])

        s.h("지배구조")
        if annual is not None:
            year, code, label = (quarter[0], quarter[1], quarter[2]) if quarter else (annual[0], "11011", "사업보고서")
            hs, _ = self._get(f"dart.largest_holders {corp} {year} {code}",
                              lambda: self.src.dart.largest_holders(corp, year, code, as_of=self.d),
                              secs=[s], what="최대주주 현황")
            if hs:
                body = [r for r in hs if str(r.get("nm", "")).strip() not in ("계", "합계")]
                common = [r for r in body if "보통" in str(r.get("stock_knd", ""))]
                rates = [_float_or_none(r.get("trmend_posesn_stock_qota_rt")) for r in common]
                known = [x for x in rates if x is not None]
                no_rate = len(rates) - len(known)
                if known:
                    s.add(f"최대주주·특수관계인 보통주 지분 합: {sum(known):.2f}% ({year} {label} 기준"
                          + (f", {no_rate}건 비율 없음: 합에서 제외" if no_rate else "") + ")")
                else:
                    s.miss("최대주주·특수관계인 지분 합", f"{no_rate}건 비율 없음")
                s.table([{"이름": r.get("nm", ""), "관계": r.get("relate", ""), "주식 종류": r.get("stock_knd", ""),
                          "기말 주식수": r.get("trmend_posesn_stock_co", ""),
                          "지분율(%)": r.get("trmend_posesn_stock_qota_rt", "")} for r in body],
                        ["이름", "관계", "주식 종류", "기말 주식수", "지분율(%)"])
            elif hs is not None:
                s.miss("최대주주 현황", f"{year} {label} 데이터 없음")
        if self.filings is None:
            s.miss("지배구조 공시 이력", self.filings_error or "조회 실패")
        else:
            gov = [f for f in self._filings_since(GOVERNANCE_DAYS)
                   if _matches(str(f.get("report_nm", "")), ("분할", "합병", "지주"))]
            s.add("최근 3년 분할·합병·지주 관련 공시:")
            if gov:
                s.table(self._filing_rows(gov), ["접수일", "보고서명", "제출인", "정정", "접수번호"])
            else:
                s.add("- 없음")
            vu = [f for f in self._filings_since(GOVERNANCE_DAYS)
                  if _matches(str(f.get("report_nm", "")), ("기업가치 제고", "기업가치제고", "밸류업"))]
            s.add("밸류업(기업가치 제고 계획) 공시: " + (
                ", ".join(f"{_filing_date(f)} {f.get('report_nm')}" for f in vu) if vu else "최근 3년 없음"))
        buyback = {k: len(v) for k, v in self.major.items() if k.startswith("자기주식")}
        s.add("주주환원 참고: 공시 조회 구간 자기주식 주요사항보고서 " +
              (", ".join(f"{k} {n}건" for k, n in buyback.items()) if buyback else "데이터 없음"))

    def _val_row(self, p: Target, mktcap: int | None, v: dict) -> dict:
        note = ""
        if v["ttm"] is not None and v["ttm"] <= 0:
            note = "적자: PER 산출 안 함"
        return {"종목": p.name, "시가총액": _won(mktcap), "최근 4분기 지배순이익": _won(v["ttm"]),
                "PER": "" if v["per"] is None else f"{v['per']:.2f}",
                "PBR": "" if v["pbr"] is None else f"{v['pbr']:.2f}", "비고": note}

    # ------------------------------------------------------------------ calendar

    def _calendar(self) -> str:
        end = self.cal.next_sessions(self.d, max(self.h.trading_days, MIN_SESSIONS))[-1]

        def in_range(x: date) -> bool:
            return self.d <= x <= end

        # Dividend filings only go to the notes: a receipt date is not a record date.
        hints = sorted({_plus_year(fd) for f in self.filings or []
                        if "잠정" in str(f.get("report_nm", ""))
                        and (fd := _filing_date(f)) and in_range(_plus_year(fd))})
        events = build_calendar(self.d, self.h, self.cal, hints)
        recent = sorted(fd for f in self._filings_since(365)
                        if "배당" in str(f.get("report_nm", "")) and (fd := _filing_date(f)))
        if recent:
            events.notes.append(f"최근 1년 배당 관련 공시 {len(recent)}건(최근 {recent[-1]})"
                                " — 기준일은 공시 섹션·본문 확인")
        if self.filings is None:
            events.notes.append(f"배당·실적 예상일: 공시 조회 실패({self.filings_error})")
        return render_calendar(events)
