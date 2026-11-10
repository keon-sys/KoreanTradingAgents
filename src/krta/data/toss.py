"""토스증권 Open API client: investor flows, short selling, program trading, credit, lending.

Source of truth: the official OpenAPI 3.0 document
https://openapi.tossinvest.com/openapi-docs/latest/openapi.json (info.version 1.2.28) and
https://openapi.tossinvest.com/openapi-docs/overview.md, both fetched 2026-11-10 (linked
from https://developers.tossinvest.com/llms.txt). No third-party client code is used.

SECURITY. The client id/secret this module receives can also place orders. This module
therefore only ever sends the requests in `_ALLOWED` (a token POST and read-only GETs of
the "trading trend" and market-indicator endpoints); any other method/path raises
`TossPathRejected` before anything is sent, and so does any `X-Tossinvest-Account` header
(the header that order/account/asset endpoints require). Order, account, holdings,
transaction and buying-power endpoints are not reachable through this module.

Protocol (per the documents above):
- Token: POST /oauth2/token, `application/x-www-form-urlencoded` body
  grant_type=client_credentials, client_id, client_secret (credentials in the form body,
  not HTTP Basic). 200 -> {access_token (JWT), token_type "Bearer", expires_in (s)}.
  Errors use the OAuth2 shape {error, error_description}: 400 invalid_request,
  401 invalid_client, 403 access_denied ("IP address not allowed": the caller's public IP
  is not registered under 토스증권 WTS 설정 > Open API > 허용 IP 관리). No refresh token.
  Only ONE access token is valid per client: issuing a new one revokes the previous one
  at once (`401 token-revoked`), including a token another program holds.
- Data: `Authorization: Bearer <token>`. Envelope {"result": {...}}; errors
  {"error": {requestId, code, message}}: 401 invalid-token / expired-token /
  token-revoked, 403 ip-not-allowed / forbidden, 404 stock-not-found, 429.
- Paging (all endpoints below): query count (1..100), until (YYYY-MM-DD, inclusive);
  `result.records` newest first, `result.nextUntil` is the next page's `until` or null.
- Rate limits: client x group. STOCK_TRADING_TREND (the five /stocks/{symbol}/... series)
  and MARKET_INDICATOR are 10 requests/s each; AUTH 5/s. Give TossApi an HttpClient
  with `min_interval >= MIN_INTERVAL`.
- Numbers are decimal strings. Ratios are fractions (0.5089 = 50.89%); this module
  returns them in percent.

Endpoints and normalized rows (every method returns rows oldest first, start..end
inclusive; a value Toss does not provide is None, never 0):
- investor_flow: GET /api/v1/stocks/{symbol}/investor-trading.
  Keys FLOW_KEYS (same as KrxPortal.investor_flow), values are net-buy VOLUMES in shares
  (주) = buy - sell. Toss has no amount (원) axis for single stocks. 외국인 is
  registered foreigners only (등록외국인; Toss excludes unregistered ones). On the
  still-open day Toss gives provisional 외국인/기관합계 only, so 개인, 기타법인 and the
  seven institution sub-groups are None there.
- foreign_holding (same endpoint, fetched once): 외국인보유수량 (주), 외국인지분율 (%,
  of listed shares), 한도소진율 (%, = 보유수량 / 한도수량 x 100, computed here);
  all None until Toss publishes the day (evening, re-confirmed next morning).
- investor_flow_market: GET /api/v1/market-indicators/{KOSPI|KOSDAQ}/investor-trading,
  interval=1d. Keys FLOW_KEYS; net-buy AMOUNTS in 원 = buyAmount - sellAmount. 외국인 here
  includes unregistered foreigners. Today's record is provisional until the close.
- shorting: GET /api/v1/stocks/{symbol}/short-selling. 공매도거래량 (주),
  공매도거래대금 (원), 공매도거래량비중 / 공매도거래대금비중 (% of the day's total incl.
  off-hours sessions; None when Toss lacks the denominator). 공매도잔고수량 /
  공매도잔고금액 are kept for KrxPortal.shorting compatibility and are always None (Toss
  publishes no short-sale balance).
- program_trading: GET /api/v1/stocks/{symbol}/program-trades (KRX only, no NXT).
  차익매수, 차익매도, 차익순매수, 비차익매수, 비차익매도, 비차익순매수 (주).
- credit_balance: GET /api/v1/stocks/{symbol}/credit-trades (published T+1 early
  morning: the latest row is the previous session). 융자신규, 융자상환, 융자잔고 (주),
  융자잔고율 (% of listed shares), 융자공여율 (% of the day's volume); 대주신규, 대주상환,
  대주잔고, 대주잔고율, 대주공여율 likewise for 신용대주. A side Toss lacks that day is None.
- securities_lending: GET /api/v1/stocks/{symbol}/securities-lending (기관 간 대차).
  대차체결, 대차상환, 대차잔고수량 (주), 대차잔고금액 (원).

Secrets: the token is kept in memory only and registered with the Redactor as soon as it
is received; the token POST is sent with log_body=False and max_retries=0; no response
body is logged; reasons never carry the id, secret, token or headers.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from krta.data.krx_portal import FLOW_KEYS
from krta.net import FetchError, HttpClient
from krta.redact import Redactor

__all__ = [
    "BASE_URL", "FLOW_KEYS", "MIN_INTERVAL", "REASON_403", "SOURCE", "TOKEN_URL",
    "TossApi", "TossPathRejected", "TossUnavailable",
]

SOURCE = "toss"
HOST = "openapi.tossinvest.com"
BASE_URL = f"https://{HOST}"
TOKEN_PATH = "/oauth2/token"
TOKEN_URL = BASE_URL + TOKEN_PATH

# STOCK_TRADING_TREND / MARKET_INDICATOR: 10 requests/s per client.
MIN_INTERVAL = 0.15
PAGE_SIZE = 100
# 10 pages = 1,000 trading days, well beyond the longest look-back the collector uses.
MAX_PAGES = 10
# Refresh the token this many seconds before it expires.
TOKEN_REFRESH_MARGIN = 300.0
# Token POSTs per TossApi: first issue, one 401 re-issue, one expiry refresh. Every
# issue revokes the client's previous token, so never loop on it.
MAX_TOKEN_POSTS = 3

_SYMBOL = r"[0-9A-Za-z]{6}"
_STOCK_SERIES = ("investor-trading", "program-trades", "short-selling", "credit-trades",
                 "securities-lending")
# (method, full path) pairs this module may send. Matched with fullmatch on the path of
# the URL actually requested; the URL must also be https://openapi.tossinvest.com with no
# query string other than the params we pass.
_ALLOWED: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("POST", re.compile(re.escape(TOKEN_PATH))),
    ("GET", re.compile(rf"/api/v1/stocks/{_SYMBOL}/(?:{'|'.join(_STOCK_SERIES)})")),
    ("GET", re.compile(r"/api/v1/market-indicators/(?:KOSPI|KOSDAQ)/investor-trading")),
)
_FORBIDDEN_HEADER = "x-tossinvest-account"
# urlsplit silently drops \t\r\n, so the checked URL could differ from the sent one.
_CONTROL_OR_SPACE = re.compile(r"[\x00-\x20\x7f]")

Market = Literal["KS", "KQ"]
_MARKET_SYMBOL = {"KS": "KOSPI", "KQ": "KOSDAQ"}

# Toss institution breakdown keys -> FLOW_KEYS names.
_BREAKDOWN = (
    ("financialInvestment", "금융투자"), ("insurance", "보험"), ("trust", "투신"),
    ("privateEquityFund", "사모"), ("bank", "은행"), ("otherFinancialInstitution", "기타금융"),
    ("pensionFund", "연기금"),
)

REASON_403 = "토스 API 403: 허용 IP 미등록 또는 권한 문제"
_NO_CREDENTIALS = "토스증권 API 자격증명 없음(TOSS_CLIENT_ID, TOSS_CLIENT_SECRET)"
_TOKEN_401 = "토스 API 401: 클라이언트 인증 실패(TOSS_CLIENT_ID, TOSS_CLIENT_SECRET 값 확인)"
_TOKEN_400 = "토스 API 토큰 요청 거부(400: 요청 형식)"
_TOKEN_FORMAT = "토스 API 토큰 응답 형식 변경"
_TOKEN_CAP = f"토스 API 토큰 발급 횟수 한도({MAX_TOKEN_POSTS}회) 초과"
_DATA_401 = "토스 API 401: 토큰 재발급 후에도 인증 실패"


class TossUnavailable(Exception):
    """Toss data cannot be had: no credentials, auth/IP failure, format change, ...

    `reason` is safe to show and log: it never contains the client id/secret, the access
    token or request headers.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class TossPathRejected(Exception):
    """A request outside the read-only allowlist was attempted. A programming error, not
    a data gap: deliberately not a TossUnavailable, so it is never swallowed as one."""


class TossApi:
    def __init__(
        self,
        http: HttpClient,
        redactor: Redactor,
        client_id: str | None,
        client_secret: str | None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._http = http
        self._redactor = redactor
        self._client_id = client_id or None
        self._client_secret = client_secret or None
        # Belt and braces: Redactor.from_env already covers *_ID / *_SECRET env names.
        for s in (self._client_id, self._client_secret):
            if s:
                redactor.add(s)
        self._clock = clock
        self._token: str | None = None
        self._expires_at = 0.0
        self._token_posts = 0
        # Set once the token endpoint refuses us (400/401/403): later calls fail fast,
        # since retrying cannot help and each issue would revoke the previous token.
        self._token_error: str | None = None
        self._memo: dict[tuple, list[dict]] = {}

    def available(self) -> bool:
        return self._client_id is not None and self._client_secret is not None

    # --- public series ---------------------------------------------------------------

    def investor_flow(self, code: str, start: date, end: date) -> list[dict]:
        """Daily net-buy volume (주) per investor group for one issue, oldest first."""
        return self._series(f"/api/v1/stocks/{code}/investor-trading", start, end, _flow_row)

    def foreign_holding(self, code: str, start: date, end: date) -> list[dict]:
        """Daily foreign holding quantity, share of listed shares (%) and limit usage (%)."""
        return self._series(f"/api/v1/stocks/{code}/investor-trading", start, end, _foreign_row)

    def investor_flow_market(self, market: Market, start: date, end: date) -> list[dict]:
        """Daily net-buy amount (원) per investor group for the whole KOSPI or KOSDAQ."""
        try:
            symbol = _MARKET_SYMBOL[market]
        except KeyError:
            raise ValueError(f"지원하지 않는 시장: {market!r} (KS 또는 KQ)") from None
        return self._series(f"/api/v1/market-indicators/{symbol}/investor-trading", start, end,
                            _market_flow_row, {"interval": "1d"})

    def shorting(self, code: str, start: date, end: date) -> list[dict]:
        return self._series(f"/api/v1/stocks/{code}/short-selling", start, end, _shorting_row)

    def program_trading(self, code: str, start: date, end: date) -> list[dict]:
        return self._series(f"/api/v1/stocks/{code}/program-trades", start, end, _program_row)

    def credit_balance(self, code: str, start: date, end: date) -> list[dict]:
        return self._series(f"/api/v1/stocks/{code}/credit-trades", start, end, _credit_row)

    def securities_lending(self, code: str, start: date, end: date) -> list[dict]:
        return self._series(f"/api/v1/stocks/{code}/securities-lending", start, end, _lending_row)

    # --- transport -------------------------------------------------------------------

    def _send(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        data: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """The only place that sends anything. Enforces the allowlist and header ban."""
        url = _checked_url(method, path, headers)
        if method == "POST":
            # max_retries=0: never resend credentials; log_body=False: never log them.
            return self._http.post(url, data=data, headers=headers, source=SOURCE,
                                   log_body=False, max_retries=0)
        return self._http.get(url, params=params, headers=headers, source=SOURCE)

    def _access_token(self) -> str:
        if not self.available():
            raise TossUnavailable(_NO_CREDENTIALS) from None
        if self._token_error is not None:
            raise TossUnavailable(self._token_error) from None
        if self._token is not None and self._clock() < self._expires_at:
            return self._token
        self._token = None
        error = self._issue_token()
        if error is not None:
            raise TossUnavailable(self._redactor.redact(error)) from None
        assert self._token is not None
        return self._token

    def _issue_token(self) -> str | None:
        """None on success (token stored), else a reason without secrets."""
        if self._token_posts >= MAX_TOKEN_POSTS:
            return _TOKEN_CAP
        self._token_posts += 1
        form = {"grant_type": "client_credentials", "client_id": self._client_id,
                "client_secret": self._client_secret}
        try:
            resp = self._send("POST", TOKEN_PATH, data=form,
                              headers={"Accept": "application/json"})
        except FetchError as exc:
            sticky = {400: _TOKEN_400, 401: _TOKEN_401, 403: REASON_403}.get(exc.status or 0)
            if sticky is not None:
                self._token_error = sticky
                return sticky
            # HttpClient messages carry only method, URL and status, never the form body.
            return f"토스 API 토큰 요청 실패: {exc.message}"
        body = _json_or_none(resp)
        token = body.get("access_token") if isinstance(body, dict) else None
        if isinstance(token, str) and token:
            self._redactor.add(token)  # before anything else can echo it
        expires = body.get("expires_in") if isinstance(body, dict) else None
        token_type = body.get("token_type") if isinstance(body, dict) else None
        if (not isinstance(token, str) or not token or type(expires) is not int or expires <= 0
                or not isinstance(token_type, str) or token_type.lower() != "bearer"):
            return _TOKEN_FORMAT
        margin = min(TOKEN_REFRESH_MARGIN, expires / 2)
        self._token = token
        self._expires_at = self._clock() + expires - margin
        return None

    def _get_json(self, path: str, params: Mapping[str, Any]) -> Any:
        """GET with the bearer token; on 401 re-issue the token once and retry."""
        _checked_url("GET", path, None)  # reject before a token is even issued
        for attempt in range(2):
            token = self._access_token()
            try:
                resp = self._send("GET", path, params=params, headers={
                    "Authorization": f"Bearer {token}", "Accept": "application/json"})
            except FetchError as exc:
                if exc.status == 401 and attempt == 0:
                    self._token = None  # expired or revoked by a newer token
                    continue
                if exc.status == 401:
                    raise TossUnavailable(_DATA_401) from None
                if exc.status == 403:
                    raise TossUnavailable(REASON_403) from None
                raise TossUnavailable(
                    self._redactor.redact(f"토스 API 요청 실패: {exc.message}")) from None
            body = _json_or_none(resp)
            if body is None:
                raise TossUnavailable(f"토스 API 응답 형식 변경(JSON 아님): {_label(path)}") from None
            return body
        raise TossUnavailable(_DATA_401) from None  # not reached: the loop returns or raises

    def _records(self, path: str, start: date, end: date,
                 extra: Mapping[str, str]) -> list[Any]:
        """All records with start <= date <= end (raw, newest first), following nextUntil."""
        _check_range(start, end)
        key = (path, start, end, tuple(sorted(extra.items())))
        if key in self._memo:
            return self._memo[key]
        out: list[Any] = []
        until = end
        for _ in range(MAX_PAGES):
            body = self._get_json(path, {**extra, "count": PAGE_SIZE, "until": until.isoformat()})
            result = body.get("result") if isinstance(body, dict) else None
            records = result.get("records") if isinstance(result, dict) else None
            if not isinstance(records, list):
                raise TossUnavailable(f"토스 API 응답 형식 변경(records 없음): {_label(path)}") from None
            out.extend(records)
            nxt = result.get("nextUntil")
            if not records or nxt is None:
                break
            try:
                nxt_day = date.fromisoformat(nxt)
            except (TypeError, ValueError):
                raise TossUnavailable(
                    f"토스 API 응답 형식 변경(nextUntil): {_label(path)}") from None
            if nxt_day >= until:
                raise TossUnavailable(
                    f"토스 API 응답 형식 변경(nextUntil 진행 없음): {_label(path)}") from None
            if nxt_day < start:
                break
            until = nxt_day
        else:
            raise TossUnavailable(f"토스 API 페이지 한도({MAX_PAGES}) 초과: {_label(path)}") from None
        self._memo[key] = out
        return out

    def _series(self, path: str, start: date, end: date, convert: Callable[[Any], dict],
                extra: Mapping[str, str] | None = None) -> list[dict]:
        _check_range(start, end)
        records = self._records(path, start, end, extra or {})
        rows: dict[date, dict] = {}
        try:
            for r in records:
                row = convert(r)
                if start <= row["date"] <= end:
                    rows.setdefault(row["date"], row)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise TossUnavailable(
                f"토스 API 응답 형식 변경(행 변환 실패 {type(exc).__name__}): {_label(path)}"
            ) from None
        return [rows[d] for d in sorted(rows)]


def _checked_url(method: str, path: str, headers: Mapping[str, str] | None) -> str:
    """BASE_URL + path if (method, path) is allowlisted and no account header is set,
    else raise TossPathRejected."""
    url = BASE_URL + path
    parts = urlsplit(url)
    allowed = (
        not _CONTROL_OR_SPACE.search(url) and urlunsplit(parts) == url
        and parts.scheme == "https" and parts.netloc == HOST and not parts.query
        and not parts.fragment
        and any(m == method and p.fullmatch(parts.path) for m, p in _ALLOWED)
    )
    if not allowed:
        raise TossPathRejected(f"토스 API 허용 목록 밖 요청 거부: {method} {path!r}")
    if any(k.lower() == _FORBIDDEN_HEADER for k in (headers or {})):
        raise TossPathRejected("토스 API 계좌 헤더(X-Tossinvest-Account) 사용 금지")
    return url


# --- row conversion (raise KeyError/TypeError/ValueError on unexpected shapes) ---------


def _flow_row(r: dict) -> dict:
    inst = r["institution"]
    bd = inst.get("breakdown")
    row: dict[str, Any] = {
        "date": _day(r),
        "외국인": _int(r["foreigner"]["netBuyVolume"]),
        "기관합계": _int(inst["netBuyVolume"]),
    }
    for key, name in _BREAKDOWN:
        row[name] = None if bd is None else _int(bd[key]["netBuyVolume"])
    row["기타법인"] = _net_volume(r.get("otherCorporation"))
    row["개인"] = _net_volume(r.get("individual"))
    return {k: row[k] for k in FLOW_KEYS}


def _foreign_row(r: dict) -> dict:
    fh = r.get("foreignerHolding")
    if fh is None:
        return {"date": _day(r), "외국인보유수량": None, "외국인지분율": None, "한도소진율": None}
    held, limit = _int(fh["holdingQuantity"]), _int(fh["limitQuantity"])
    return {
        "date": _day(r),
        "외국인보유수량": held,
        "외국인지분율": _pct(fh["holdingRate"]),
        "한도소진율": held / limit * 100 if held is not None and limit else None,
    }


def _market_flow_row(r: dict) -> dict:
    """A group whose amounts are missing that day is None (the day is kept)."""
    inst = r.get("institution")
    bd = inst.get("breakdown") if isinstance(inst, dict) else None
    row: dict[str, Any] = {
        "date": _day(r),
        "외국인": _net_amount(r.get("foreigner")),
        "기관합계": _net_amount(inst),
        "기타법인": _net_amount(r.get("otherCorporation")),
        "개인": _net_amount(r.get("individual")),
    }
    for key, name in _BREAKDOWN:
        row[name] = _net_amount(bd.get(key)) if isinstance(bd, dict) else None
    return {k: row[k] for k in FLOW_KEYS}


def _shorting_row(r: dict) -> dict:
    return {
        "date": _day(r),
        "공매도거래량": _int(r["shortSellingVolume"]),
        "공매도거래대금": _int(r["shortSellingAmount"]),
        "공매도잔고수량": None,
        "공매도잔고금액": None,
        "공매도거래량비중": _pct(r.get("shortSellingVolumeRate")),
        "공매도거래대금비중": _pct(r.get("shortSellingAmountRate")),
    }


def _program_row(r: dict) -> dict:
    row: dict[str, Any] = {"date": _day(r)}
    for key, name in (("arbitrage", "차익"), ("nonArbitrage", "비차익")):
        v = r[key]
        row[f"{name}매수"] = _int(v["buyVolume"])
        row[f"{name}매도"] = _int(v["sellVolume"])
        row[f"{name}순매수"] = _int(v["netBuyVolume"])
    return row


def _credit_row(r: dict) -> dict:
    row: dict[str, Any] = {"date": _day(r)}
    for key, name in (("marginLoan", "융자"), ("stockLoan", "대주")):
        v = r.get(key)
        row[f"{name}신규"] = None if v is None else _int(v["newQuantity"])
        row[f"{name}상환"] = None if v is None else _int(v["returnQuantity"])
        row[f"{name}잔고"] = None if v is None else _int(v["balanceQuantity"])
        row[f"{name}잔고율"] = None if v is None else _pct(v["balanceRate"])
        row[f"{name}공여율"] = None if v is None else _pct(v["tradingRate"])
    return row


def _lending_row(r: dict) -> dict:
    return {
        "date": _day(r),
        "대차체결": _int(r["executionQuantity"]),
        "대차상환": _int(r["repaymentQuantity"]),
        "대차잔고수량": _int(r["balanceQuantity"]),
        "대차잔고금액": _int(r["balanceAmount"]),
    }


def _net_volume(v: dict | None) -> int | None:
    return None if v is None else _int(v["netBuyVolume"])


def _net_amount(v: dict | None) -> int | None:
    """buyAmount - sellAmount, or None when the group or either amount is missing.
    A present but malformed amount still raises (format change)."""
    if v is None:
        return None
    if not isinstance(v, dict):
        raise TypeError("investor group is not an object")
    buy, sell = _int(v.get("buyAmount")), _int(v.get("sellAmount"))
    return None if buy is None or sell is None else buy - sell


def _decimal(v: Any) -> Decimal:
    if isinstance(v, bool) or not isinstance(v, (str, int)):
        raise TypeError(f"not a decimal string: {type(v).__name__}")
    try:
        d = Decimal(v)
    except InvalidOperation:
        raise ValueError("not a decimal string") from None
    if not d.is_finite():
        raise ValueError("not a finite number")
    return d


def _int(v: Any) -> int | None:
    if v is None:
        return None
    d = _decimal(v)
    if d != d.to_integral_value():
        raise ValueError("not an integer")
    return int(d)


def _pct(v: Any) -> float | None:
    """Fraction string ('0.5089') -> percent (50.89)."""
    return None if v is None else float(_decimal(v) * 100)


def _day(r: dict) -> date:
    return date.fromisoformat(r["date"])


def _label(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _json_or_none(resp: Any) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


def _check_range(start: date, end: date) -> None:
    if start > end:
        raise ValueError(f"시작일이 종료일보다 늦음: {start} > {end}")
