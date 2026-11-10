import io
import json
import zipfile
from datetime import date
from pathlib import Path

import pytest

from krta.cache import FileCache
from krta.data.dart import Dart, DartKeyError
from krta.net import FetchError, HttpClient
from krta.redact import Redactor

FIXTURES = Path(__file__).parent / "fixtures" / "dart"
BASE = "https://opendart.fss.or.kr/api/"
KEY = "DUMMYDARTKEY0000"
AS_OF = date(2026, 11, 6)
SAMSUNG = "00126380"


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def corpcode_zip(xml_text=None):
    xml_text = xml_text if xml_text is not None else (FIXTURES / "CORPCODE.xml").read_text(encoding="utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("CORPCODE.xml", xml_text.encode("utf-8"))
    return buf.getvalue()


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.headers = {}
        self._body = body
        if isinstance(body, bytes):
            self.content = body
        else:
            self.content = json.dumps(body, ensure_ascii=False).encode("utf-8")

    def json(self):
        if isinstance(self._body, bytes):
            raise ValueError("not json")
        return self._body


class FakeSession:
    """Replays (status, body) responses in order; body is a JSON object or raw bytes."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        status, body = self.responses.pop(0)
        return FakeResponse(status, body)


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def make_dart(tmp_path, *responses, clock=None):
    session = FakeSession(*responses)
    http = HttpClient(Redactor([KEY]), None, session=session, sleep=lambda s: None)
    sleeps = []
    dart = Dart(
        http, FileCache(tmp_path), KEY,
        sleep=sleeps.append, clock=clock if clock is not None else Clock(),
    )
    return dart, session, sleeps


def endpoint(call):
    return call[1].removeprefix(BASE)


def test_corp_code_mapping_cached(tmp_path):
    clock = Clock()
    dart, session, _ = make_dart(tmp_path, (200, corpcode_zip()), clock=clock)
    assert dart.corp_code("005930") == SAMSUNG
    assert dart.corp_code("035720") == "00258801"
    assert len(session.calls) == 1
    method, url, kw = session.calls[0]
    assert (method, url) == ("GET", BASE + "corpCode.xml")
    assert kw["params"] == {"crtfc_key": KEY}

    cached = FileCache(tmp_path).get_json("dart/corpcode.json")
    assert cached["map"] == {"005930": SAMSUNG, "035720": "00258801"}  # unlisted (blank) skipped

    # A new instance within 7 days reads the cache without any request.
    clock.t += 6 * 86400
    dart2, session2, _ = make_dart(tmp_path, clock=clock)
    assert dart2.corp_code("005930") == SAMSUNG
    assert session2.calls == []


def test_corp_code_cache_expires_after_7_days(tmp_path):
    clock = Clock()
    dart, _, _ = make_dart(tmp_path, (200, corpcode_zip()), clock=clock)
    dart.corp_code("005930")
    clock.t += 7 * 86400 + 1
    dart2, session2, _ = make_dart(tmp_path, (200, corpcode_zip()), clock=clock)
    assert dart2.corp_code("005930") == SAMSUNG
    assert len(session2.calls) == 1


def test_corp_code_unknown_refetches_stale_map_once_then_raises(tmp_path):
    dart, _, _ = make_dart(tmp_path, (200, corpcode_zip()))
    dart.corp_code("005930")
    # Fresh cache lacks the code (e.g. newly listed): one refetch, then a clear error.
    dart2, session2, _ = make_dart(tmp_path, (200, corpcode_zip()))
    with pytest.raises(LookupError, match="999999"):
        dart2.corp_code("999999")
    with pytest.raises(LookupError):
        dart2.corp_code("888888")
    assert len(session2.calls) == 1


def test_corp_code_error_xml_maps_status(tmp_path):
    err = '<?xml version="1.0" encoding="UTF-8"?><result><status>011</status><message>사용할 수 없는 키입니다.</message></result>'
    dart, _, _ = make_dart(tmp_path, (200, err.encode("utf-8")))
    with pytest.raises(DartKeyError, match=r"^DART 키 문제\(상태 011\)$"):
        dart.corp_code("005930")
    assert FileCache(tmp_path).get_json("dart/corpcode.json") is None


def test_filings_paginate_and_clip_after_as_of(tmp_path):
    dart, session, _ = make_dart(tmp_path, (200, load("list_p1.json")), (200, load("list_p2.json")))
    rows = dart.filings(SAMSUNG, date(2026, 9, 1), date(2026, 11, 30), as_of=AS_OF)

    assert [endpoint(c) for c in session.calls] == ["list.json", "list.json"]
    p1, p2 = (c[2]["params"] for c in session.calls)
    assert p1 == {
        "crtfc_key": KEY, "corp_code": SAMSUNG, "bgn_de": "20260901",
        "end_de": "20261106", "page_no": 1, "page_count": 100,
    }
    assert p2["page_no"] == 2

    assert [r["rcept_no"] for r in rows] == [
        "20261105000456", "20261020800111", "20261002000789", "20260915000321",
    ]  # 20261109 (after as_of) dropped
    assert all(r["rcept_dt"] <= "20261106" for r in rows)
    flagged = [r["report_nm"] for r in rows if r["is_correction"]]
    assert flagged == ["[기재정정]반기보고서 (2026.06)"]


def test_filings_clip_to_end_before_as_of(tmp_path):
    dart, session, _ = make_dart(tmp_path, (200, load("list_p1.json")), (200, load("list_p2.json")))
    rows = dart.filings(SAMSUNG, date(2026, 9, 1), date(2026, 10, 20), as_of=AS_OF)
    assert session.calls[0][2]["params"]["end_de"] == "20261020"
    assert [r["rcept_dt"] for r in rows] == ["20261020", "20261002", "20260915"]


def test_filings_start_after_cutoff_makes_no_request(tmp_path):
    dart, session, _ = make_dart(tmp_path)
    assert dart.filings(SAMSUNG, date(2026, 11, 7), date(2026, 11, 30), as_of=AS_OF) == []
    assert session.calls == []


def test_status_013_is_empty(tmp_path):
    dart, session, _ = make_dart(tmp_path, (200, load("status_013.json")))
    assert dart.filings(SAMSUNG, date(2026, 9, 1), AS_OF, as_of=AS_OF) == []
    assert len(session.calls) == 1


def test_status_020_retries_then_ok(tmp_path):
    dart, session, sleeps = make_dart(
        tmp_path,
        (200, load("status_020.json")),
        (200, load("status_020.json")),
        (200, load("cvbdIsDecsn.json")),
    )
    rows = dart.major_report("CB", "00258801", date(2026, 10, 1), date(2026, 11, 30), as_of=AS_OF)
    assert len(session.calls) == 3
    assert len(sleeps) == 2 and sleeps[0] > 0 and sleeps[1] > sleeps[0]
    assert [r["rcept_no"] for r in rows] == ["20261015000555"]


def test_status_020_exhausted_raises_fetch_error(tmp_path):
    dart, session, sleeps = make_dart(tmp_path, *[(200, load("status_020.json"))] * 4)
    with pytest.raises(FetchError, match="020") as ei:
        dart.company(SAMSUNG)
    assert len(session.calls) == 4  # 1 + 3 retries
    assert len(sleeps) == 3
    assert KEY not in str(ei.value)


def test_status_010_raises_key_error_without_key_value(tmp_path):
    dart, session, _ = make_dart(tmp_path, (200, load("status_010.json")))
    with pytest.raises(DartKeyError) as ei:
        dart.filings(SAMSUNG, date(2026, 9, 1), AS_OF, as_of=AS_OF)
    assert str(ei.value) == "DART 키 문제(상태 010)"
    assert KEY not in str(ei.value)
    assert ei.value.__cause__ is None and ei.value.__context__ is None
    assert len(session.calls) == 1


@pytest.mark.parametrize("code", ["011", "012", "901"])
def test_other_key_statuses_raise_key_error(tmp_path, code):
    dart, _, _ = make_dart(tmp_path, (200, {"status": code, "message": "x"}))
    with pytest.raises(DartKeyError, match=rf"^DART 키 문제\(상태 {code}\)$"):
        dart.company(SAMSUNG)


def test_unknown_status_raises_fetch_error(tmp_path):
    dart, _, _ = make_dart(tmp_path, (200, {"status": "100", "message": "필드의 부적절한 값입니다."}))
    with pytest.raises(FetchError, match="100") as ei:
        dart.company(SAMSUNG)
    assert ei.value.source == "dart"
    assert KEY not in str(ei.value)


def test_non_json_response_raises_fetch_error(tmp_path):
    dart, _, _ = make_dart(tmp_path, (200, b"<html>maintenance</html>"))
    with pytest.raises(FetchError):
        dart.company(SAMSUNG)


def test_financials_falls_back_to_ofs_on_013(tmp_path):
    ofs = {"status": "000", "message": "정상", "list": [
        {"rcept_no": "20260814000111", "reprt_code": "11012", "bsns_year": "2026", "sj_div": "IS",
         "account_id": "ifrs-full_Revenue", "account_nm": "매출액", "thstrm_amount": "1000"},
    ]}
    dart, session, _ = make_dart(tmp_path, (200, load("status_013.json")), (200, ofs))
    rows = dart.financials(SAMSUNG, 2026, "11012", as_of=AS_OF)
    assert [endpoint(c) for c in session.calls] == ["fnlttSinglAcntAll.json"] * 2
    divs = [c[2]["params"]["fs_div"] for c in session.calls]
    assert divs == ["CFS", "OFS"]
    assert session.calls[0][2]["params"] == {
        "crtfc_key": KEY, "corp_code": SAMSUNG, "bsns_year": "2026",
        "reprt_code": "11012", "fs_div": "CFS",
    }
    assert rows == ofs["list"]


def test_financials_cfs_used_and_report_after_as_of_dropped(tmp_path):
    cfs = {"status": "000", "message": "정상", "list": [
        {"rcept_no": "20261114000111", "reprt_code": "11014", "account_nm": "매출액"},
    ]}
    dart, session, _ = make_dart(tmp_path, (200, cfs))
    assert dart.financials(SAMSUNG, 2026, "11014", as_of=AS_OF) == []
    assert len(session.calls) == 1  # CFS exists: no OFS request


def test_financials_rejects_bad_reprt_code(tmp_path):
    dart, session, _ = make_dart(tmp_path)
    with pytest.raises(ValueError):
        dart.financials(SAMSUNG, 2026, "99999", as_of=AS_OF)
    assert session.calls == []


def test_company_fields(tmp_path):
    body = {"status": "000", "message": "정상", "corp_code": SAMSUNG, "corp_name": "삼성전자",
            "stock_code": "005930", "induty_code": "264", "acc_mt": "12"}
    dart, session, _ = make_dart(tmp_path, (200, body))
    info = dart.company(SAMSUNG)
    assert endpoint(session.calls[0]) == "company.json"
    assert session.calls[0][2]["params"] == {"crtfc_key": KEY, "corp_code": SAMSUNG}
    assert (info["induty_code"], info["acc_mt"]) == ("264", "12")
    assert "status" not in info and "message" not in info


@pytest.mark.parametrize(
    ("kind", "ep"),
    [
        ("유상증자", "piicDecsn.json"),
        ("CB", "cvbdIsDecsn.json"),
        ("BW", "bdwtIsDecsn.json"),
        ("자기주식취득", "tsstkAqDecsn.json"),
        ("자기주식처분", "tsstkDpDecsn.json"),
    ],
)
def test_major_report_endpoints(tmp_path, kind, ep):
    dart, session, _ = make_dart(tmp_path, (200, load("cvbdIsDecsn.json")))
    rows = dart.major_report(kind, "00258801", date(2026, 10, 1), date(2026, 11, 30), as_of=AS_OF)
    assert endpoint(session.calls[0]) == ep
    assert session.calls[0][2]["params"] == {
        "crtfc_key": KEY, "corp_code": "00258801", "bgn_de": "20261001", "end_de": "20261106",
    }
    assert [r["rcept_no"] for r in rows] == ["20261015000555"]  # 20261110 (after as_of) dropped


def test_major_report_buyback_cancellation_filters_list(tmp_path):
    dart, session, _ = make_dart(tmp_path, (200, load("list_p1.json")), (200, load("list_p2.json")))
    rows = dart.major_report("자기주식소각", SAMSUNG, date(2026, 9, 1), date(2026, 11, 30), as_of=AS_OF)
    assert [endpoint(c) for c in session.calls] == ["list.json", "list.json"]
    assert [r["report_nm"] for r in rows] == ["주식소각결정"]


def test_major_report_rejects_unknown_kind(tmp_path):
    dart, _, _ = make_dart(tmp_path)
    with pytest.raises(ValueError):
        dart.major_report("감자", SAMSUNG, date(2026, 1, 1), AS_OF, as_of=AS_OF)


def test_holders_and_dividends_endpoints_and_clip(tmp_path):
    def body(*rows):
        return {"status": "000", "message": "정상", "list": list(rows)}

    dart, session, _ = make_dart(
        tmp_path,
        (200, body({"rcept_no": "20260814000001", "nm": "삼성생명보험"},
                   {"rcept_no": "20261114000001", "nm": "late"})),
        (200, body({"rcept_no": "20261001000002", "rcept_dt": "2026-10-01", "repror": "국민연금공단"},
                   {"rcept_no": "20261107000002", "rcept_dt": "2026-11-07", "repror": "late"})),
        (200, body({"rcept_no": "20261105000003", "rcept_dt": "20261105", "repror": "임원A"},
                   {"rcept_no": "20261109000003", "rcept_dt": "20261109", "repror": "late"})),
        (200, body({"rcept_no": "20260314000004", "se": "주당 현금배당금(원)", "thstrm": "1,446"},
                   {"rcept_no": "bad", "se": "undated"})),
    )
    holders = dart.largest_holders(SAMSUNG, 2026, "11012", as_of=AS_OF)
    major = dart.major_holders(SAMSUNG, as_of=AS_OF)
    execs = dart.exec_holdings(SAMSUNG, as_of=AS_OF)
    divs = dart.dividends(SAMSUNG, 2025, "11011", as_of=AS_OF)

    assert [endpoint(c) for c in session.calls] == [
        "hyslrSttus.json", "majorstock.json", "elestock.json", "alotMatter.json",
    ]
    assert session.calls[0][2]["params"] == {
        "crtfc_key": KEY, "corp_code": SAMSUNG, "bsns_year": "2026", "reprt_code": "11012",
    }
    assert session.calls[1][2]["params"] == {"crtfc_key": KEY, "corp_code": SAMSUNG}
    assert [r["nm"] for r in holders] == ["삼성생명보험"]
    assert [r["repror"] for r in major] == ["국민연금공단"]
    assert [r["repror"] for r in execs] == ["임원A"]
    assert [r["se"] for r in divs] == ["주당 현금배당금(원)"]  # undated row excluded


def test_http_error_propagates_redacted(tmp_path):
    dart, _, _ = make_dart(tmp_path, (403, {}))
    with pytest.raises(FetchError) as ei:
        dart.company(SAMSUNG)
    assert ei.value.status == 403
    assert KEY not in str(ei.value)
