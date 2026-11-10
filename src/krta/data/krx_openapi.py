"""KRX Open API client (data-dbg.krx.co.kr): daily quotes, listing info, indices, futures.

Rows are returned as the API sends them (numeric fields stay strings, possibly with
commas); convert with `to_int` / `to_float`. Responses are cached per endpoint and
trading day, except for `unconfirmed_date` (a day whose data may still change) and
empty results (possibly not published yet).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from krta.cache import FileCache
from krta.net import FetchError, HttpClient

SOURCE = "krx_openapi"
BASE_URL = "https://data-dbg.krx.co.kr/svc/apis/"

Market = Literal["KS", "KQ"]

_DAILY_TRADES = {"KS": "sto/stk_bydd_trd", "KQ": "sto/ksq_bydd_trd"}
_BASE_INFO = {"KS": "sto/stk_isu_base_info", "KQ": "sto/ksq_isu_base_info"}
_INDEX_DAILY = {"KS": "idx/kospi_dd_trd", "KQ": "idx/kosdaq_dd_trd"}
_FUTURES_DAILY = "drv/fut_bydd_trd"

_AUTH_MESSAGE = "KRX Open API 인증 실패: 서비스 이용신청 승인 또는 KRX_API_KEY를 확인"


class KrxAuthError(Exception):
    """HTTP 401 from KRX Open API: service not approved yet or wrong key."""


def _number_text(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().replace(",", "")
    return None if s in ("", "-") else s


def to_int(s: str | None) -> int | None:
    """'1,234' -> 1234; '', '-', None -> None. Other non-numeric text raises ValueError."""
    t = _number_text(s)
    return None if t is None else int(t)


def to_float(s: str | None) -> float | None:
    """'1,234.5' -> 1234.5; '', '-', None -> None. Other non-numeric text raises ValueError."""
    t = _number_text(s)
    return None if t is None else float(t)


class KrxOpenApi:
    def __init__(
        self,
        http: HttpClient,
        cache: FileCache,
        api_key: str,
        unconfirmed_date: date | None = None,
    ):
        self._http = http
        self._cache = cache
        self._api_key = api_key
        self._unconfirmed_date = unconfirmed_date

    def daily_trades(self, market: Market, d: date) -> list[dict]:
        """All issues' daily quotes (ISU_CD, TDD_CLSPRC, ACC_TRDVAL, MKTCAP, ...)."""
        return self._fetch(_by_market(_DAILY_TRADES, market), d)

    def base_info(self, market: Market, d: date) -> list[dict]:
        """All issues' listing info (ISU_SRT_CD, ISU_ABBRV, LIST_DD, SECUGRP_NM, ...)."""
        return self._fetch(_by_market(_BASE_INFO, market), d)

    def index_daily(self, market: Market, d: date) -> list[dict]:
        """KOSPI- or KOSDAQ-family index rows (IDX_NM, CLSPRC_IDX, FLUC_RT, ...)."""
        return self._fetch(_by_market(_INDEX_DAILY, market), d)

    def futures_daily(self, d: date) -> list[dict]:
        """Exchange futures daily rows (PROD_NM, ISU_NM, SETL_PRC, ACC_OPNINT_QTY, ...)."""
        return self._fetch(_FUTURES_DAILY, d)

    def _fetch(self, ep: str, d: date) -> list[dict]:
        bas_dd = d.strftime("%Y%m%d")
        key = f"krx/{ep}/{bas_dd}.json"
        cached = self._cache.get_json(key)
        if isinstance(cached, list):
            return cached

        rows = self._request(ep, bas_dd)
        # Never cache []: the day may simply not be published yet (callers only ask
        # for XKRX sessions, so a genuinely empty day costs just a repeat request).
        if rows and d != self._unconfirmed_date:
            self._cache.put_json(key, rows)
        return rows

    def _request(self, ep: str, bas_dd: str) -> list[dict]:
        try:
            resp = self._http.get(
                BASE_URL + ep,
                params={"basDd": bas_dd},
                headers={"AUTH_KEY": self._api_key},
                source=SOURCE,
            )
        except FetchError as exc:
            if exc.status != 401:
                raise
            auth_error = KrxAuthError(_AUTH_MESSAGE)
        else:
            auth_error = None
        if auth_error is not None:
            # Raised outside the except block so no exception chain is attached.
            raise auth_error from None

        # Messages name the endpoint and date only: never the key or request headers.
        where = f"{ep} basDd={bas_dd}"
        try:
            body: Any = resp.json()
        except ValueError:
            body = None
        rows = body.get("OutBlock_1") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            raise FetchError(SOURCE, f"응답 형식 오류(OutBlock_1 없음): {where}") from None
        return rows


def _by_market(table: dict[str, str], market: str) -> str:
    try:
        return table[market]
    except KeyError:
        raise ValueError(f"지원하지 않는 시장: {market!r} (KS 또는 KQ)") from None
