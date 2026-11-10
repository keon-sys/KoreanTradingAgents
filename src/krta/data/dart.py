"""OpenDART client (opendart.fss.or.kr): filings, company info, financials, ownership, dividends.

Rows are returned as the API sends them (amounts stay strings). Every dated result is
clipped to items received on or before `as_of` (the analysis date). Only the
stock-code -> corp-code map is cached (7 days); date-scoped queries are not cached, so a
filing published later is never hidden behind a stale entry.
"""

from __future__ import annotations

import io
import time
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterable
from datetime import date
from typing import Any, Literal

from krta.cache import FileCache
from krta.net import FetchError, HttpClient

SOURCE = "dart"
BASE_URL = "https://opendart.fss.or.kr/api/"

CORPCODE_CACHE_KEY = "dart/corpcode.json"
CORPCODE_TTL_SECONDS = 7 * 86400
# Real CORPCODE.xml is tens of MB; refuse anything far larger (zip bomb guard).
CORPCODE_MAX_BYTES = 256 * 1024 * 1024

PAGE_COUNT = 100  # list.json maximum
STATUS_OK = "000"
STATUS_NO_DATA = "013"
STATUS_RATE_LIMIT = "020"
KEY_STATUSES = frozenset({"010", "011", "012", "901"})
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BASE_DELAY = 2.0

# 11013 1Q, 11012 half-year, 11014 3Q, 11011 annual.
REPORT_CODES = frozenset({"11011", "11012", "11013", "11014"})

MajorKind = Literal["유상증자", "CB", "BW", "자기주식취득", "자기주식처분", "자기주식소각"]

_MAJOR_ENDPOINTS = {
    "유상증자": "piicDecsn.json",
    "CB": "cvbdIsDecsn.json",
    "BW": "bdwtIsDecsn.json",
    "자기주식취득": "tsstkAqDecsn.json",
    "자기주식처분": "tsstkDpDecsn.json",
}
_CANCELLATION_KIND = "자기주식소각"
_CANCELLATION_REPORT = "주식소각결정"


class DartKeyError(Exception):
    """OpenDART rejected the key (status 010/011/012/901). Never carries the key value."""


class Dart:
    def __init__(
        self,
        http: HttpClient,
        cache: FileCache,
        api_key: str,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ):
        self._http = http
        self._cache = cache
        self._api_key = api_key
        self._sleep = sleep
        self._clock = clock
        self._corp_map: dict[str, str] | None = None
        self._corp_map_fresh = False  # downloaded by this instance

    # -- corp code ---------------------------------------------------------------------

    def corp_code(self, stock_code: str) -> str:
        """6-digit stock code -> 8-digit DART corp code. Raises LookupError if unlisted."""
        if self._corp_map is None:
            self._corp_map = self._cached_corp_map()
        if self._corp_map is None or (stock_code not in self._corp_map and not self._corp_map_fresh):
            # No cache, or a cached map that predates the listing: download once.
            self._corp_map = self._download_corp_map()
            self._corp_map_fresh = True
        try:
            return self._corp_map[stock_code]
        except KeyError:
            raise LookupError(f"DART 고유번호 없음: 종목코드 {stock_code}") from None

    def _cached_corp_map(self) -> dict[str, str] | None:
        entry = self._cache.get_json(CORPCODE_CACHE_KEY)
        if not isinstance(entry, dict):
            return None
        fetched, mapping = entry.get("fetched_at"), entry.get("map")
        if not isinstance(fetched, (int, float)) or not isinstance(mapping, dict):
            return None
        age = self._clock() - fetched
        return mapping if 0 <= age < CORPCODE_TTL_SECONDS else None

    def _download_corp_map(self) -> dict[str, str]:
        resp = self._http.get(BASE_URL + "corpCode.xml", params=self._params(), source=SOURCE)
        xml_bytes = _unzip_corpcode(resp.content)
        if xml_bytes is None:
            # Errors come back as a bare XML <result><status>..</status></result>, not a zip.
            status, message = _xml_status(resp.content)
            self._check_status(status, message, "corpCode.xml")
            raise FetchError(SOURCE, "응답 형식 오류(zip 아님): corpCode.xml") from None
        mapping = _parse_corpcode(xml_bytes)
        if not mapping:
            raise FetchError(SOURCE, "응답 형식 오류(상장사 없음): corpCode.xml") from None
        self._cache.put_json(CORPCODE_CACHE_KEY, {"fetched_at": self._clock(), "map": mapping})
        return mapping

    # -- filings -----------------------------------------------------------------------

    def filings(self, corp: str, start: date, end: date, *, as_of: date) -> list[dict]:
        """All filings (list.json) received in [start, min(end, as_of)], newest first.

        Each item gains `is_correction` (정정 공시 여부).
        """
        cutoff = min(end, as_of)
        if start > cutoff:
            return []
        rows: list[dict] = []
        page = 1
        while True:
            body = self._call("list.json", {
                "corp_code": corp, "bgn_de": _ymd(start), "end_de": _ymd(cutoff),
                "page_no": page, "page_count": PAGE_COUNT,
            })
            if body is None:
                break
            items = _list_field(body, "list.json")
            rows.extend(items)
            total_page = body.get("total_page")
            if not isinstance(total_page, int):
                total_page = page if len(items) < PAGE_COUNT else page + 1
            if page >= total_page or not items:
                break
            page += 1
        out = []
        for row in _clip(rows, cutoff):
            name = str(row.get("report_nm", ""))
            out.append({**row, "is_correction": name.startswith("[기재정정]") or "정정" in name})
        return out

    def major_report(
        self, kind: MajorKind, corp: str, start: date, end: date, *, as_of: date
    ) -> list[dict]:
        """주요사항보고서 rows of one kind; 자기주식소각 comes from list.json (주식소각결정)."""
        if kind == _CANCELLATION_KIND:
            rows = self.filings(corp, start, end, as_of=as_of)
            return [r for r in rows if _CANCELLATION_REPORT in str(r.get("report_nm", ""))]
        try:
            ep = _MAJOR_ENDPOINTS[kind]
        except KeyError:
            raise ValueError(f"지원하지 않는 주요사항 종류: {kind!r}") from None
        cutoff = min(end, as_of)
        if start > cutoff:
            return []
        return self._rows(ep, {"corp_code": corp, "bgn_de": _ymd(start), "end_de": _ymd(cutoff)}, as_of)

    # -- company / financials ----------------------------------------------------------

    def company(self, corp: str) -> dict:
        """기업개황: corp_name, stock_code, induty_code (KSIC), acc_mt (결산월), ...

        Current snapshot only (the API has no point-in-time view). {} when not found.
        """
        body = self._call("company.json", {"corp_code": corp})
        if body is None:
            return {}
        return {k: v for k, v in body.items() if k not in ("status", "message")}

    def financials(self, corp: str, year: int, reprt_code: str, *, as_of: date) -> list[dict]:
        """단일회사 전체 재무제표: consolidated (CFS) first, separate (OFS) when CFS is 013."""
        params = {"corp_code": corp, **_report_params(year, reprt_code)}
        for fs_div in ("CFS", "OFS"):
            body = self._call("fnlttSinglAcntAll.json", {**params, "fs_div": fs_div})
            if body is not None:
                return _clip(_list_field(body, "fnlttSinglAcntAll.json"), as_of)
        return []

    # -- ownership / dividends ---------------------------------------------------------

    def largest_holders(self, corp: str, year: int, reprt_code: str, *, as_of: date) -> list[dict]:
        """최대주주 현황 (hyslrSttus.json)."""
        return self._rows("hyslrSttus.json", {"corp_code": corp, **_report_params(year, reprt_code)}, as_of)

    def major_holders(self, corp: str, *, as_of: date) -> list[dict]:
        """대량보유 상황보고 (majorstock.json)."""
        return self._rows("majorstock.json", {"corp_code": corp}, as_of)

    def exec_holdings(self, corp: str, *, as_of: date) -> list[dict]:
        """임원·주요주주 소유보고 (elestock.json)."""
        return self._rows("elestock.json", {"corp_code": corp}, as_of)

    def dividends(self, corp: str, year: int, reprt_code: str, *, as_of: date) -> list[dict]:
        """배당에 관한 사항 (alotMatter.json)."""
        return self._rows("alotMatter.json", {"corp_code": corp, **_report_params(year, reprt_code)}, as_of)

    # -- transport ---------------------------------------------------------------------

    def _rows(self, ep: str, params: dict[str, Any], as_of: date) -> list[dict]:
        body = self._call(ep, params)
        return [] if body is None else _clip(_list_field(body, ep), as_of)

    def _params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"crtfc_key": self._api_key, **(extra or {})}

    def _call(self, ep: str, params: dict[str, Any]) -> dict | None:
        """JSON body for status 000, None for 013. 020 is retried with backoff."""
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            if attempt:
                self._sleep(RATE_LIMIT_BASE_DELAY * 2 ** (attempt - 1))
            resp = self._http.get(BASE_URL + ep, params=self._params(params), source=SOURCE)
            try:
                body: Any = resp.json()
            except ValueError:
                body = None
            if not isinstance(body, dict):
                raise FetchError(SOURCE, f"응답 형식 오류(JSON 아님): {ep}") from None
            status = str(body.get("status", ""))
            if status == STATUS_OK:
                return body
            if status == STATUS_NO_DATA:
                return None
            if status != STATUS_RATE_LIMIT:
                self._check_status(status, body.get("message"), ep)
        raise FetchError(
            SOURCE, f"요청 제한(상태 {STATUS_RATE_LIMIT}) {RATE_LIMIT_RETRIES}회 재시도 후에도 지속: {ep}"
        ) from None

    def _check_status(self, status: str, message: Any, ep: str) -> None:
        """Raise for any non-000 status (callers handle 013/020 before calling this)."""
        if status in KEY_STATUSES:
            raise DartKeyError(f"DART 키 문제(상태 {status})") from None
        text = str(message or "")[:200]
        if self._api_key:
            text = text.replace(self._api_key, "***")
        raise FetchError(SOURCE, f"DART 오류(상태 {status or '없음'}) {text}: {ep}".strip()) from None


# -- helpers ---------------------------------------------------------------------------


def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _report_params(year: int, reprt_code: str) -> dict[str, str]:
    if reprt_code not in REPORT_CODES:
        raise ValueError(f"잘못된 보고서 코드: {reprt_code!r} (11011/11012/11013/11014)")
    return {"bsns_year": str(year), "reprt_code": reprt_code}


def _list_field(body: dict, ep: str) -> list[dict]:
    rows = body.get("list")
    if not isinstance(rows, list):
        raise FetchError(SOURCE, f"응답 형식 오류(list 없음): {ep}") from None
    return [r for r in rows if isinstance(r, dict)]


def _receipt_date(row: dict) -> date | None:
    """rcept_dt (YYYYMMDD or YYYY-MM-DD), else the first 8 digits of rcept_no."""
    for text in (str(row.get("rcept_dt") or "").replace("-", "").replace(".", "").strip(),
                 str(row.get("rcept_no") or "")[:8]):
        if len(text) == 8 and text.isascii() and text.isdigit():
            try:
                return date(int(text[:4]), int(text[4:6]), int(text[6:]))
            except ValueError:
                continue
    return None


def _clip(rows: Iterable[dict], cutoff: date) -> list[dict]:
    """Keep rows received on or before `cutoff`; rows with no readable date are dropped."""
    out = []
    for row in rows:
        d = _receipt_date(row)
        if d is not None and d <= cutoff:
            out.append(row)
    return out


def _unzip_corpcode(content: bytes) -> bytes | None:
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        return None
    with zf:
        members = [i for i in zf.infolist() if i.filename.lower().endswith(".xml")]
        if not members:
            return None
        info = members[0]
        if info.file_size > CORPCODE_MAX_BYTES:
            raise FetchError(SOURCE, "응답 형식 오류(corpCode.xml 과대)") from None
        return zf.read(info)


def _xml_status(content: bytes) -> tuple[str, str]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return "", ""
    return (root.findtext("status") or "").strip(), (root.findtext("message") or "").strip()


def _parse_corpcode(xml_bytes: bytes) -> dict[str, str]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        raise FetchError(SOURCE, "응답 형식 오류(XML 파싱 실패): corpCode.xml") from None
    best: dict[str, tuple[str, str]] = {}  # stock_code -> (modify_date, corp_code)
    for item in root.iter("list"):
        stock = (item.findtext("stock_code") or "").strip()
        corp = (item.findtext("corp_code") or "").strip()
        modified = (item.findtext("modify_date") or "").strip()
        if not stock or not corp:
            continue  # unlisted company
        # A reused stock code keeps the most recently modified entry.
        if stock not in best or modified > best[stock][0]:
            best[stock] = (modified, corp)
    return {stock: corp for stock, (_, corp) in best.items()}
