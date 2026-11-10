"""KRX 정보데이터시스템(data.krx.co.kr) client behind a member login.

Investor-group net buying, short selling and foreign holdings are not in the KRX Open
API, and data.krx.co.kr answers these queries only for a logged-in session. This is the
only module that logs in to KRX.

Protocol (learned from pykrx 1.2.9 sdist, PyPI, read 2026-11-10; no pykrx code is used):
- Login, pykrx/website/comm/auth.py (`warmup_krx_session`, `login_krx`):
  1. GET  /contents/MDC/COMS/client/MDCCOMS001.cmd          (issues JSESSIONID)
  2. GET  /contents/MDC/COMS/client/view/login.jsp?site=mdc (login iframe)
  3. POST /contents/MDC/COMS/client/MDCCOMS001D1.cmd, form
     mbrNm="", telNo="", di="", certType="", mbrId=<id>, pw=<pw>
     -> JSON `_error_code`: CD001 success, CD011 already logged in elsewhere (resend
     with skipDup=Y), CD010 password change required. The site's own login.jsp
     (fetched 2026-11-10) also handles CD005, CD006 (wrong password, with
     loginErrCnt/loginErrMaxCnt: the account locks at the max), CD007, and treats an
     empty code as success (mdc.util.js `isErrorCode`).
- Queries, pykrx/website/krx/krxio.py (`KrxWebIo`): POST
  /comm/bldAttendant/getJsonData.cmd, form `bld=<code>&...`, headers User-Agent,
  Referer .../outerLoader/index.cmd, X-Requested-With. Dates are YYYYMMDD.
  bld codes, pykrx/website/krx/market/core.py:
  - dbms/comm/finder/finder_stkisu (`상장종목검색`): locale, mktsel=ALL, searchText,
    typeNo=0 -> block1[{full_code (ISIN), short_code, ...}]. Works without login.
  - dbms/MDC/STAT/standard/MDCSTAT02303 (`투자자별_거래실적_개별종목_일별추이_상세`):
    strtDd, endDd, isuCd=<ISIN>, inqTpCd=2, trdVolVal=2 (거래대금), askBid=3 (순매수),
    detailView=1 -> output[{TRD_DD "YYYY/MM/DD", TRDVAL1..TRDVAL11, TRDVAL_TOT}];
    column order per pykrx wrap.py: 금융투자 보험 투신 사모 은행 기타금융 연기금
    기타법인 개인 외국인 기타외국인.
  - dbms/MDC/STAT/standard/MDCSTAT02203 (`투자자별_거래실적_전체시장_일별추이_상세`):
    strtDd, endDd, mktId=STK|KSQ, etf/etn/elw="", trdVolVal, askBid, detailView=1
    -> output, same columns.
  - dbms/MDC/STAT/srt/MDCSTAT30001 (`개별종목_공매도_종합정보`): isuCd, strtDd, endDd
    -> OutBlock_1[{TRD_DD, CVSRTSELL_TRDVOL, CVSRTSELL_TRDVAL, STR_CONST_VAL1 (잔고수량),
    STR_CONST_VAL2 (잔고금액)}]; "-" = balance not published yet.
  - dbms/MDC/STAT/standard/MDCSTAT03702 (`외국인보유량_개별추이`): searchType=2, strtDd,
    endDd, isuCd -> output[{TRD_DD, LIST_SHRS, FORN_HD_QTY, FORN_SHR_RT,
    FORN_ORD_LMT_QTY, FORN_LMT_EXHST_RT}].
  pykrx 1.2.9 has no program-trading or credit-balance bld, so neither is offered here.
- Session expiry: pykrx only re-logs in on a 1-hour timer. Observed 2026-11-10: the
  query endpoints answer a session that is not logged in with HTTP 400, body "LOGOUT".

Credentials are kept in this object only: the login POST is sent with log_body=False and
max_retries=0, at most MAX_LOGIN_POSTS times per object, and no reason string ever carries an id, password, cookie or form body. Session cookies
stay in the HttpClient's in-memory requests.Session. The JSESSIONID cookie is scoped to
`.krx.co.kr`, so give KrxPortal its own HttpClient rather than one shared with other
KRX hosts (e.g. the Open API on data-dbg.krx.co.kr).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from krta.data.krx_openapi import to_float, to_int
from krta.net import FetchError, HttpClient

SOURCE = "krx_portal"
BASE_URL = "https://data.krx.co.kr"
LOGIN_PAGE = BASE_URL + "/contents/MDC/COMS/client/MDCCOMS001.cmd"
LOGIN_JSP = BASE_URL + "/contents/MDC/COMS/client/view/login.jsp?site=mdc"
LOGIN_URL = BASE_URL + "/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
JSON_URL = BASE_URL + "/comm/bldAttendant/getJsonData.cmd"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_QUERY_HEADERS = {
    "User-Agent": USER_AGENT,
    "Referer": BASE_URL + "/contents/MDC/MDI/outerLoader/index.cmd",
    "X-Requested-With": "XMLHttpRequest",
}

_BLD_FINDER = "dbms/comm/finder/finder_stkisu"
_BLD_FLOW = "dbms/MDC/STAT/standard/MDCSTAT02303"
_BLD_FLOW_MARKET = "dbms/MDC/STAT/standard/MDCSTAT02203"
_BLD_SHORTING = "dbms/MDC/STAT/srt/MDCSTAT30001"
_BLD_FOREIGN = "dbms/MDC/STAT/standard/MDCSTAT03702"

Market = Literal["KS", "KQ"]
_MKT_ID = {"KS": "STK", "KQ": "KSQ"}

# TRDVALn columns of the detailed investor views, in order.
_FLOW_COLUMNS = (
    "금융투자", "보험", "투신", "사모", "은행", "기타금융", "연기금",
    "기타법인", "개인", "외국인", "기타외국인",
)
_INSTITUTIONS = ("금융투자", "보험", "투신", "사모", "은행", "기타금융", "연기금")
FLOW_KEYS = (
    "date", "외국인", "기관합계", "금융투자", "보험", "투신", "사모", "은행",
    "기타금융", "연기금", "기타법인", "개인",
)

# Login POSTs (each carries credentials) per KrxPortal: the first login plus one re-login
# after session expiry. A CD011 skipDup resend counts too. Wrong passwords lock the account.
MAX_LOGIN_POSTS = 2
_LOGIN_CAP = f"KRX 로그인 횟수 한도({MAX_LOGIN_POSTS}회) 초과: 계정 잠금 방지를 위해 중단"

_NO_CREDENTIALS = "KRX 정보데이터시스템 자격증명 없음(KRX_ID, KRX_PW)"
_LOGIN_CODES = {
    "CD005": "로그인 거부",
    "CD006": "아이디 또는 비밀번호 불일치",
    "CD007": "로그인 거부",
    "CD010": "비밀번호 변경 필요(krx.co.kr에서 변경)",
    "CD011": "중복 로그인 해제 실패",
}


class PortalUnavailable(Exception):
    """KRX portal data cannot be had: no credentials, login failure, format change, ...

    `reason` is safe to show and log: it never contains credentials or cookies.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _LoggedOut(Exception):
    """Internal: the query endpoint answered LOGOUT (session not logged in)."""


class KrxPortal:
    def __init__(self, http: HttpClient, user_id: str | None, password: str | None):
        self._http = http
        self._user_id = user_id or None
        self._password = password or None
        self._logged_in = False
        # Set once a login fails; later calls fail fast without contacting KRX, since
        # each wrong-password attempt counts toward the account lock.
        self._login_error: str | None = None
        self._isin: dict[str, str] = {}
        self._login_posts = 0

    def available(self) -> bool:
        return self._user_id is not None and self._password is not None

    def investor_flow(self, code: str, start: date, end: date) -> list[dict]:
        """Daily net buying (원) per investor group for one issue, oldest first.

        외국인 includes 기타외국인; 기관합계 is 금융투자 through 연기금.
        """
        _check_range(start, end)
        isin = self._isin_of(code)
        rows = self._query(
            _BLD_FLOW, "output",
            strtDd=_ymd(start), endDd=_ymd(end), isuCd=isin,
            inqTpCd=2, trdVolVal=2, askBid=3, detailView=1,
        )
        return self._normalize(_BLD_FLOW, rows, start, end, _flow_row)

    def investor_flow_market(self, market: Market, start: date, end: date) -> list[dict]:
        """Daily net buying (원) per investor group for the whole KOSPI or KOSDAQ market."""
        try:
            mkt_id = _MKT_ID[market]
        except KeyError:
            raise ValueError(f"지원하지 않는 시장: {market!r} (KS 또는 KQ)") from None
        _check_range(start, end)
        rows = self._query(
            _BLD_FLOW_MARKET, "output",
            strtDd=_ymd(start), endDd=_ymd(end), mktId=mkt_id, etf="", etn="", elw="",
            trdVolVal=2, askBid=3, detailView=1,
        )
        return self._normalize(_BLD_FLOW_MARKET, rows, start, end, _flow_row)

    def shorting(self, code: str, start: date, end: date) -> list[dict]:
        """Daily short-sale volume/value and balance; balance is None until KRX publishes it."""
        _check_range(start, end)
        isin = self._isin_of(code)
        rows = self._query(
            _BLD_SHORTING, "OutBlock_1", isuCd=isin, strtDd=_ymd(start), endDd=_ymd(end),
        )
        return self._normalize(_BLD_SHORTING, rows, start, end, _shorting_row)

    def foreign_holding(self, code: str, start: date, end: date) -> list[dict]:
        """Daily foreign holding quantity, share of listed shares (%) and limit usage (%)."""
        _check_range(start, end)
        isin = self._isin_of(code)
        rows = self._query(
            _BLD_FOREIGN, "output",
            searchType=2, strtDd=_ymd(start), endDd=_ymd(end), isuCd=isin,
        )
        return self._normalize(_BLD_FOREIGN, rows, start, end, _foreign_row)

    # --- session -------------------------------------------------------------------

    def _ensure_login(self) -> None:
        if not self.available():
            raise PortalUnavailable(_NO_CREDENTIALS) from None
        if not self._logged_in:
            self._login()

    def _login(self) -> None:
        self._logged_in = False
        if self._login_error is None:
            self._login_error = self._attempt_login()
        if self._login_error is not None:
            raise PortalUnavailable(self._login_error) from None
        self._logged_in = True

    def _attempt_login(self) -> str | None:
        """None on success, else a reason without credentials."""
        if self._login_posts >= MAX_LOGIN_POSTS:
            return _LOGIN_CAP
        try:
            self._http.get(
                LOGIN_PAGE, headers={"User-Agent": USER_AGENT}, source=SOURCE,
                follow_same_host_redirects=True,
            )
            self._http.get(
                LOGIN_JSP, headers={"User-Agent": USER_AGENT, "Referer": LOGIN_PAGE},
                source=SOURCE, follow_same_host_redirects=True,
            )
        except FetchError as exc:
            return f"KRX 로그인 페이지 요청 실패: {exc.message}"

        form = {
            "mbrNm": "", "telNo": "", "di": "", "certType": "",
            "mbrId": self._user_id, "pw": self._password,
        }
        code, body = self._post_login(form)
        if code == "CD011":
            code, body = self._post_login({**form, "skipDup": "Y"})
        if code in ("", "CD001"):
            return None
        if code is None:
            return body  # transport or format failure reason
        what = _LOGIN_CODES.get(code, "로그인 거부")
        tries = ""
        if code == "CD006":
            cnt, mx = body.get("loginErrCnt"), body.get("loginErrMaxCnt")
            if isinstance(cnt, int) and isinstance(mx, int):
                tries = f", 실패 {cnt}/{mx}회"
        # The server's _error_message is not echoed: it is free text we do not control.
        return f"KRX 로그인 실패({code}: {what}{tries})"

    def _post_login(self, form: dict[str, Any]) -> tuple[str | None, Any]:
        """(error code, JSON body) or (None, failure reason)."""
        if self._login_posts >= MAX_LOGIN_POSTS:
            return None, _LOGIN_CAP
        self._login_posts += 1
        try:
            # max_retries=0: a transport retry would resend the credentials (lock risk).
            resp = self._http.post(
                LOGIN_URL, data=form,
                headers={"User-Agent": USER_AGENT, "Referer": LOGIN_PAGE},
                source=SOURCE, log_body=False, max_retries=0,
            )
        except FetchError as exc:
            # HttpClient messages carry only method, URL and status, never the form body.
            return None, f"KRX 로그인 요청 실패: {exc.message}"
        body = _json_or_none(resp)
        if not isinstance(body, dict):
            return None, "KRX 로그인 응답 형식 변경(JSON 아님)"
        code = body.get("_error_code", "")
        if not isinstance(code, str):
            return None, "KRX 로그인 응답 형식 변경(_error_code)"
        return code, body

    # --- queries -------------------------------------------------------------------

    def _isin_of(self, code: str) -> str:
        if code in self._isin:
            return self._isin[code]
        rows = self._query(
            _BLD_FINDER, "block1", locale="ko_KR", mktsel="ALL", searchText=code, typeNo=0,
        )
        for r in rows:
            if isinstance(r, dict) and r.get("short_code") == code and r.get("full_code"):
                self._isin[code] = str(r["full_code"])
                return self._isin[code]
        raise PortalUnavailable(f"KRX 종목 검색에 없는 종목코드: {code}") from None

    def _query(self, bld: str, block: str, **params: Any) -> list[Any]:
        """POST getJsonData.cmd; on a LOGOUT answer re-login once and retry."""
        self._ensure_login()
        for attempt in range(2):
            outcome = self._post_query(bld, params)
            if outcome is _LoggedOut:
                if attempt:
                    break
                self._login()
                continue
            if isinstance(outcome, str):
                raise PortalUnavailable(outcome) from None
            rows = outcome.get(block) if isinstance(outcome, dict) else None
            if not isinstance(rows, list):
                raise PortalUnavailable(f"KRX 응답 형식 변경({block} 없음): {bld}") from None
            return rows
        self._logged_in = False
        raise PortalUnavailable(f"KRX 세션 만료: 재로그인 후에도 로그아웃 응답: {bld}") from None

    def _post_query(self, bld: str, params: dict[str, Any]) -> Any:
        """Parsed JSON body, `_LoggedOut`, or a failure reason (str)."""
        try:
            resp = self._http.post(
                JSON_URL, data={"bld": bld, **params}, headers=_QUERY_HEADERS, source=SOURCE,
            )
        except FetchError as exc:
            if exc.status == 400:  # the endpoint's LOGOUT answer
                return _LoggedOut
            return f"KRX 정보데이터시스템 요청 실패: {exc.message}"
        body = _json_or_none(resp)
        if body is None:
            if (getattr(resp, "text", "") or "").strip() == "LOGOUT":
                return _LoggedOut
            return f"KRX 응답 형식 변경(JSON 아님): {bld}"
        return body

    def _normalize(self, bld, rows: list[Any], start: date, end: date, convert) -> list[dict]:
        out: list[dict] = []
        bad: str | None = None
        for r in rows:
            try:
                row = convert(r)
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                bad = type(exc).__name__
                break
            if start <= row["date"] <= end:
                out.append(row)
        if bad is not None:
            raise PortalUnavailable(f"KRX 응답 형식 변경(행 변환 실패 {bad}): {bld}") from None
        out.sort(key=lambda row: row["date"])
        return out


# --- row conversion (raise KeyError/ValueError/TypeError on unexpected shapes) ---------


def _flow_row(r: dict) -> dict:
    # A blank amount means no net trade that day (pykrx also reads "" as 0).
    v = {name: to_int(r[f"TRDVAL{i}"]) or 0 for i, name in enumerate(_FLOW_COLUMNS, 1)}
    row: dict[str, Any] = {
        "date": _trd_dd(r),
        "외국인": v["외국인"] + v["기타외국인"],
        "기관합계": sum(v[k] for k in _INSTITUTIONS),
    }
    for k in _INSTITUTIONS + ("기타법인", "개인"):
        row[k] = v[k]
    return row


def _shorting_row(r: dict) -> dict:
    return {
        "date": _trd_dd(r),
        "공매도거래량": to_int(r["CVSRTSELL_TRDVOL"]),
        "공매도거래대금": to_int(r["CVSRTSELL_TRDVAL"]),
        "공매도잔고수량": to_int(r["STR_CONST_VAL1"]),
        "공매도잔고금액": to_int(r["STR_CONST_VAL2"]),
    }


def _foreign_row(r: dict) -> dict:
    return {
        "date": _trd_dd(r),
        "외국인보유수량": to_int(r["FORN_HD_QTY"]),
        "외국인지분율": to_float(r["FORN_SHR_RT"]),
        "한도소진율": to_float(r["FORN_LMT_EXHST_RT"]),
    }


def _trd_dd(r: dict) -> date:
    return datetime.strptime(r["TRD_DD"], "%Y/%m/%d").date()


def _json_or_none(resp: Any) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _check_range(start: date, end: date) -> None:
    if start > end:
        raise ValueError(f"시작일이 종료일보다 늦음: {start} > {end}")
