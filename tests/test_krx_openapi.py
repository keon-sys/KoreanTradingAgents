import json
from datetime import date
from pathlib import Path

import pytest

from krta.cache import FileCache
from krta.data.krx_openapi import KrxAuthError, KrxOpenApi, to_float, to_int
from krta.net import FetchError, HttpClient
from krta.redact import Redactor

FIXTURES = Path(__file__).parent / "fixtures" / "krx"
BASE = "https://data-dbg.krx.co.kr/svc/apis/"
KEY = "DUMMYKRXKEY0000"
D = date(2026, 11, 6)


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.headers = {}
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeSession:
    """Replays (status, json body) responses in order; records each request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        status, body = self.responses.pop(0)
        return FakeResponse(status, body)


def make_api(tmp_path, *responses, unconfirmed_date=None):
    session = FakeSession(*responses)
    http = HttpClient(Redactor([KEY]), None, session=session, sleep=lambda s: None)
    api = KrxOpenApi(http, FileCache(tmp_path), KEY, unconfirmed_date=unconfirmed_date)
    return api, session


def test_daily_trades_uses_endpoint_and_header(tmp_path):
    api, session = make_api(tmp_path, (200, load("stk_bydd_trd.json")))
    rows = api.daily_trades("KS", D)
    method, url, kw = session.calls[0]
    assert (method, url) == ("GET", BASE + "sto/stk_bydd_trd")
    assert kw["params"] == {"basDd": "20261106"}
    assert kw["headers"]["AUTH_KEY"] == KEY
    assert len(rows) == 3
    sam = next(r for r in rows if r["ISU_CD"] == "005930")
    assert sam["TDD_CLSPRC"] == "98,500"  # numeric strings are left as returned


@pytest.mark.parametrize(
    ("call", "ep", "fixture"),
    [
        (lambda a: a.daily_trades("KQ", D), "sto/ksq_bydd_trd", "ksq_bydd_trd.json"),
        (lambda a: a.base_info("KS", D), "sto/stk_isu_base_info", None),
        (lambda a: a.base_info("KQ", D), "sto/ksq_isu_base_info", None),
        (lambda a: a.index_daily("KS", D), "idx/kospi_dd_trd", "kospi_dd_trd.json"),
        (lambda a: a.index_daily("KQ", D), "idx/kosdaq_dd_trd", None),
        (lambda a: a.futures_daily(D), "drv/fut_bydd_trd", None),
    ],
)
def test_endpoint_per_method_and_market(tmp_path, call, ep, fixture):
    body = load(fixture) if fixture else {"OutBlock_1": []}
    api, session = make_api(tmp_path, (200, body))
    rows = call(api)
    assert session.calls[0][1] == BASE + ep
    assert rows == body["OutBlock_1"]


def test_index_row_values(tmp_path):
    api, _ = make_api(tmp_path, (200, load("kospi_dd_trd.json")))
    kospi = next(r for r in api.index_daily("KS", D) if r["IDX_NM"] == "코스피")
    assert to_float(kospi["CLSPRC_IDX"]) == 5059.69


def test_cached_second_call_makes_no_request(tmp_path):
    api, session = make_api(tmp_path, (200, load("stk_bydd_trd.json")))
    first = api.daily_trades("KS", D)
    second = api.daily_trades("KS", D)
    assert second == first and len(session.calls) == 1
    assert (tmp_path / "krx/sto/stk_bydd_trd/20261106.json").is_file()
    # A fresh client on the same cache root is served from disk as well.
    api2, session2 = make_api(tmp_path)
    assert api2.daily_trades("KS", D) == first and session2.calls == []


def test_empty_result_is_not_cached(tmp_path):
    # An empty block may mean "not published yet" (day T requested on T+1 morning);
    # caching [] would hide that day's data forever.
    full = load("stk_bydd_trd.json")
    api, session = make_api(tmp_path, (200, {"OutBlock_1": []}), (200, full))
    assert api.daily_trades("KS", D) == []
    assert not (tmp_path / "krx/sto/stk_bydd_trd/20261106.json").exists()
    assert api.daily_trades("KS", D) == full["OutBlock_1"]
    assert len(session.calls) == 2
    assert (tmp_path / "krx/sto/stk_bydd_trd/20261106.json").is_file()


def test_unconfirmed_date_not_cached(tmp_path):
    body = load("stk_bydd_trd.json")
    api, session = make_api(tmp_path, (200, body), (200, body), unconfirmed_date=D)
    api.daily_trades("KS", D)
    api.daily_trades("KS", D)
    assert len(session.calls) == 2
    assert not (tmp_path / "krx").exists()


def test_401_raises_auth_error(tmp_path):
    api, _ = make_api(tmp_path, (401, load("unauthorized.json")))
    with pytest.raises(KrxAuthError) as e:
        api.daily_trades("KS", D)
    assert str(e.value) == "KRX Open API 인증 실패: 서비스 이용신청 승인 또는 KRX_API_KEY를 확인"
    assert e.value.__cause__ is None and e.value.__context__ is None
    assert KEY not in str(e.value)


def test_other_http_error_propagates_as_fetch_error(tmp_path):
    api, _ = make_api(tmp_path, (403, {}))
    with pytest.raises(FetchError) as e:
        api.daily_trades("KS", D)
    assert not isinstance(e.value, KrxAuthError) and e.value.status == 403


@pytest.mark.parametrize("body", [{"respMsg": "x"}, {"OutBlock_1": None}, ValueError("not json")])
def test_malformed_response_raises_fetch_error_without_key(tmp_path, body):
    api, _ = make_api(tmp_path, (200, body))
    with pytest.raises(FetchError) as e:
        api.daily_trades("KS", D)
    assert e.value.source == "krx_openapi"
    assert "sto/stk_bydd_trd" in str(e.value) and KEY not in str(e.value)
    assert e.value.__context__ is None
    assert not (tmp_path / "krx").exists()


def test_rejects_unknown_market(tmp_path):
    api, session = make_api(tmp_path)
    with pytest.raises(ValueError):
        api.daily_trades("XX", D)
    assert session.calls == []


def test_to_int_handles_commas_and_blanks():
    assert to_int("1,234") == 1234 and to_int("") is None and to_int("-") is None
    assert to_int("-8,000") == -8000 and to_int(None) is None and to_int(" 12 ") == 12


def test_to_float_handles_commas_and_blanks():
    assert to_float("5059.69") == 5059.69 and to_float("1,234.5") == 1234.5
    assert to_float("-1.54") == -1.54 and to_float("") is None and to_float("-") is None
    assert to_float(None) is None


def test_number_helpers_reject_non_numeric_text():
    # A format change must surface, not silently become "no data".
    with pytest.raises(ValueError):
        to_int("abc")
    with pytest.raises(ValueError):
        to_float("n/a")
