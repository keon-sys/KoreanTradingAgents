import json
import traceback
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from krta.data import toss
from krta.data.krx_portal import FLOW_KEYS as PORTAL_FLOW_KEYS
from krta.data.toss import (
    BASE_URL,
    FLOW_KEYS,
    TOKEN_URL,
    TossApi,
    TossPathRejected,
    TossUnavailable,
)
from krta.net import HttpClient
from krta.redact import Redactor
from krta.toollog import ToolLog

FIXTURES = Path(__file__).parent / "fixtures" / "toss"
CID = "dummy-client-id-0000"
CSECRET = "dummy-client-secret-0000"
TOKEN = "dummy-access-token-0001"
D0 = date(2026, 11, 4)
D1 = date(2026, 11, 6)


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.headers = {}
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)

    def json(self):
        if isinstance(self._body, str):
            return json.loads(self._body)
        return self._body


class FakeSession:
    """Replays (status, body) responses in order; records each request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError(f"unexpected request {method} {url}")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        status, body = item
        return FakeResponse(status, body)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def token_ok(token=TOKEN, expires_in=86400):
    return (200, {**load("token_ok.json"), "access_token": token, "expires_in": expires_in})


def data(name):
    return (200, load(name))


def make(*responses, cid=CID, secret=CSECRET, toollog=None, redactor=None, clock=None):
    session = FakeSession(*responses)
    red = redactor if redactor is not None else Redactor([])
    http = HttpClient(red, toollog, session=session, sleep=lambda s: None, max_retries=0)
    api = TossApi(http, red, cid, secret, clock=clock or Clock())
    return api, session


def data_calls(session):
    return [c for c in session.calls if c[1] != TOKEN_URL]


def token_calls(session):
    return [c for c in session.calls if c[1] == TOKEN_URL]


# --------------------------------------------------------------------------- availability


@pytest.mark.parametrize(("cid", "secret"), [(None, None), ("", CSECRET), (CID, None)])
def test_unavailable_without_credentials(cid, secret):
    api, session = make(cid=cid, secret=secret)
    assert api.available() is False
    with pytest.raises(TossUnavailable) as e:
        api.investor_flow("005930", D0, D1)
    assert "TOSS_CLIENT_ID" in e.value.reason
    assert session.calls == []


# --------------------------------------------------------------------------- token


def test_token_request_is_form_post_without_logging_or_retry(tmp_path):
    log = ToolLog(tmp_path / "t.log", Redactor([]))
    api, session = make(token_ok(), data("investor_trading.json"), toollog=log)
    api.investor_flow("005930", D0, D1)
    method, url, kw = session.calls[0]
    assert (method, url) == ("POST", "https://openapi.tossinvest.com/oauth2/token")
    assert kw["data"] == {"grant_type": "client_credentials", "client_id": CID,
                          "client_secret": CSECRET}
    text = (tmp_path / "t.log").read_text(encoding="utf-8")
    assert "[request] toss ===\nPOST https://openapi.tossinvest.com/oauth2/token -> 200\n" in text
    assert CSECRET not in text and CID not in text and "client_secret" not in text


def test_token_not_resent_on_transport_error():
    import requests

    api, session = make(requests.ConnectionError("boom"))
    with pytest.raises(TossUnavailable):
        api.investor_flow("005930", D0, D1)
    assert len(session.calls) == 1


def test_token_cached_and_refreshed_five_minutes_before_expiry():
    clock = Clock()
    api, session = make(
        token_ok(expires_in=3600), data("investor_trading.json"), data("program_trades.json"),
        token_ok(token="dummy-access-token-0002"), data("short_selling.json"),
        clock=clock,
    )
    api.investor_flow("005930", D0, D1)
    clock.now += 3600 - 301
    api.program_trading("005930", D0, D1)
    assert len(token_calls(session)) == 1
    clock.now += 2  # now within 5 minutes of expiry
    api.shorting("005930", D0, D1)
    assert len(token_calls(session)) == 2
    auths = [c[2]["headers"]["Authorization"] for c in data_calls(session)]
    assert auths == [f"Bearer {TOKEN}", f"Bearer {TOKEN}", "Bearer dummy-access-token-0002"]


def test_token_value_masked_everywhere_after_success(tmp_path):
    red = Redactor([])
    log = ToolLog(tmp_path / "t.log", red)
    api, session = make(token_ok(), data("investor_trading.json"), (500, "oops"),
                        toollog=log, redactor=red)
    api.investor_flow("005930", D0, D1)
    with pytest.raises(TossUnavailable) as e:
        api.shorting("005930", D0, D1)
    # The token reached the wire only in the Authorization header ...
    assert data_calls(session)[0][2]["headers"]["Authorization"] == f"Bearer {TOKEN}"
    # ... and nowhere a user can see.
    text = (tmp_path / "t.log").read_text(encoding="utf-8")
    shown = "".join(traceback.format_exception(e.value)) + e.value.reason + str(e.value)
    for blob in (text, shown):
        assert TOKEN not in blob
    assert red.redact(f"x {TOKEN} y") == "x *** y"


def test_client_secret_never_appears_anywhere(tmp_path):
    red = Redactor([])
    log = ToolLog(tmp_path / "t.log", red)
    errors = []
    for script in ([(403, load("token_403.json"))], [(401, load("token_401.json"))],
                   [(400, {"error": "invalid_request"})], [token_ok(), (404, "nf")]):
        api, _ = make(*script, toollog=log, redactor=red)
        with pytest.raises(TossUnavailable) as e:
            api.investor_flow("005930", D0, D1)
        errors.append(e.value)
    text = (tmp_path / "t.log").read_text(encoding="utf-8")
    for err in errors:
        blob = "".join(traceback.format_exception(err)) + err.reason
        assert CSECRET not in blob and CID not in blob
    assert CSECRET not in text and CID not in text
    # Registered with the redactor too, in case anything else echoes it.
    assert red.redact(CSECRET) == "***"


def test_token_403_reason_and_no_further_token_requests():
    api, session = make((403, load("token_403.json")))
    with pytest.raises(TossUnavailable) as e:
        api.investor_flow("005930", D0, D1)
    assert e.value.reason == "토스 API 403: 허용 IP 미등록 또는 권한 문제"
    with pytest.raises(TossUnavailable) as e2:
        api.shorting("005930", D0, D1)
    assert e2.value.reason == e.value.reason
    assert len(session.calls) == 1  # failed token is not retried by later calls


def test_token_401_reason_names_keys_not_values():
    api, session = make((401, load("token_401.json")))
    with pytest.raises(TossUnavailable) as e:
        api.investor_flow("005930", D0, D1)
    assert "401" in e.value.reason and "TOSS_CLIENT_SECRET" in e.value.reason
    assert len(session.calls) == 1


def test_token_response_without_access_token_is_format_error():
    api, _ = make((200, {"token_type": "Bearer", "expires_in": 10}))
    with pytest.raises(TossUnavailable) as e:
        api.investor_flow("005930", D0, D1)
    assert "형식" in e.value.reason


# --------------------------------------------------------------------------- data errors


def test_data_401_retokens_once_then_succeeds():
    api, session = make(
        token_ok(), (401, load("error_401_expired.json")),
        token_ok(token="dummy-access-token-0002"), data("investor_trading.json"),
    )
    rows = api.investor_flow("005930", D0, D1)
    assert len(rows) == 3
    assert len(token_calls(session)) == 2
    assert data_calls(session)[-1][2]["headers"]["Authorization"] == "Bearer dummy-access-token-0002"


def test_data_401_twice_gives_up():
    api, session = make(
        token_ok(), (401, load("error_401_expired.json")),
        token_ok(token="dummy-access-token-0002"), (401, load("error_401_expired.json")),
    )
    with pytest.raises(TossUnavailable) as e:
        api.investor_flow("005930", D0, D1)
    assert "401" in e.value.reason
    assert len(session.calls) == 4


def test_data_403_reason():
    api, _ = make(token_ok(), (403, load("error_403_ip.json")))
    with pytest.raises(TossUnavailable) as e:
        api.program_trading("005930", D0, D1)
    assert e.value.reason == "토스 API 403: 허용 IP 미등록 또는 권한 문제"


def test_non_json_or_shape_change_is_unavailable():
    api, _ = make(token_ok(), (200, "<html>"), (200, {"result": {"rows": []}}),
                  (200, {"result": {"records": [{"date": "2026-11-05"}]}}))
    for _ in range(3):
        with pytest.raises(TossUnavailable) as e:
            api.securities_lending("005930", D0, D1)
        assert "형식" in e.value.reason


# --------------------------------------------------------------------------- security


@pytest.mark.parametrize(("method", "path"), [
    ("POST", "/api/v1/orders"),
    ("GET", "/api/v1/orders"),
    ("POST", "/api/v1/orders/1/cancel"),
    ("POST", "/api/v1/conditional-orders"),
    ("DELETE", "/api/v1/conditional-orders/1"),
    ("GET", "/api/v1/accounts"),
    ("GET", "/api/v1/holdings"),
    ("GET", "/api/v1/transactions"),
    ("GET", "/api/v1/buying-power"),
    ("GET", "/api/v1/stocks/005930/../../orders"),
    ("GET", "/api/v1/stocks/005930/investor-trading/../../../orders"),
    ("GET", "/api/v1/stocks/005930/investor-trading?x=1"),
    ("GET", "/api/v1/stocks/005930%2F..%2Forders/investor-trading"),
    ("GET", "//evil.example/api/v1/stocks/005930/investor-trading"),
    ("GET", "@evil.example/api/v1/stocks/005930/investor-trading"),
    ("GET", ".evil.example/api/v1/stocks/005930/investor-trading"),
    ("GET", "/api/v1/stocks/005930/investor-trading#frag"),
    ("POST", "/api/v1/stocks/005930/investor-trading"),
    ("GET", "/oauth2/token"),
    ("GET", "/api/v1/market-indicators/KOSPI/candles"),
    ("GET", "/api/v1/stocks/005930/investor-trading\n"),
    ("GET", "/api/v1/stocks/005930/investor-trading\t"),
    ("GET", "/api/v1/stocks/005930/invest\r\nor-trading"),
    ("GET", "/api/v1/stocks/005930/investor-trading\x00"),
    ("GET", "/api/v1/stocks/005930/investor-trading "),
    ("GET", "/api/v1/stocks/005930/investor-trading\x7f"),
])
def test_non_allowlisted_path_raises_before_sending(method, path):
    api, session = make(token_ok())
    with pytest.raises(TossPathRejected):
        api._send(method, path)
    assert session.calls == []


def test_account_header_is_refused_before_sending():
    api, session = make(token_ok())
    for name in ("X-Tossinvest-Account", "x-tossinvest-account", "X-TOSSINVEST-ACCOUNT"):
        with pytest.raises(TossPathRejected):
            api._send("GET", "/api/v1/stocks/005930/short-selling", headers={name: "1"})
    assert session.calls == []


def test_bad_symbol_rejected_by_allowlist():
    api, session = make(token_ok())
    with pytest.raises(TossPathRejected):
        api.investor_flow("../orders", D0, D1)
    assert session.calls == []


def test_every_public_method_uses_only_allowlisted_get_without_account_header():
    api, session = make(
        token_ok(), data("investor_trading.json"), data("market_investor_trading.json"),
        data("market_investor_trading.json"), data("short_selling.json"),
        data("program_trades.json"), data("credit_trades.json"), data("securities_lending.json"),
    )
    api.investor_flow("005930", D0, D1)
    api.foreign_holding("005930", D0, D1)  # same endpoint, served from memory
    api.investor_flow_market("KS", D0, D1)
    api.investor_flow_market("KQ", D0, D1)
    api.shorting("005930", D0, D1)
    api.program_trading("005930", D0, D1)
    api.credit_balance("005930", D0, D1)
    api.securities_lending("005930", D0, D1)
    paths = []
    for method, url, kw in session.calls:
        parts = urlsplit(url)
        assert (parts.scheme, parts.netloc) == ("https", "openapi.tossinvest.com")
        headers = {k.lower() for k in (kw.get("headers") or {})}
        assert "x-tossinvest-account" not in headers
        paths.append((method, parts.path))
    assert paths == [
        ("POST", "/oauth2/token"),
        ("GET", "/api/v1/stocks/005930/investor-trading"),
        ("GET", "/api/v1/market-indicators/KOSPI/investor-trading"),
        ("GET", "/api/v1/market-indicators/KOSDAQ/investor-trading"),
        ("GET", "/api/v1/stocks/005930/short-selling"),
        ("GET", "/api/v1/stocks/005930/program-trades"),
        ("GET", "/api/v1/stocks/005930/credit-trades"),
        ("GET", "/api/v1/stocks/005930/securities-lending"),
    ]


def test_module_never_mentions_order_or_account_paths_outside_docs():
    src = Path(toss.__file__).read_text(encoding="utf-8")
    code = src.split('"""', 2)[2]  # skip the module docstring
    for needle in ("/orders", "/accounts", "/holdings", "/buying-power", "conditional-orders"):
        assert needle not in code


# --------------------------------------------------------------------------- normalization


def test_investor_flow_normalized_shares_with_none_for_missing():
    api, session = make(token_ok(), data("investor_trading.json"))
    rows = api.investor_flow("005930", D0, D1)
    assert FLOW_KEYS == PORTAL_FLOW_KEYS
    assert [r["date"] for r in rows] == [date(2026, 11, 4), date(2026, 11, 5), date(2026, 11, 6)]
    assert all(list(r) == list(FLOW_KEYS) for r in rows)
    assert rows[0] == {
        "date": date(2026, 11, 4), "외국인": 200_001, "기관합계": 2_800,
        "금융투자": 100, "보험": 200, "투신": 300, "사모": 400, "은행": 500, "기타금융": 600,
        "연기금": 700, "기타법인": -14_200, "개인": -100_001,
    }
    # Provisional day: only foreigner/institution totals; the rest is None, never 0.
    assert rows[2] == {"date": date(2026, 11, 6), "외국인": 119_900, "기관합계": -113_200,
                       **{k: None for k in FLOW_KEYS if k not in ("date", "외국인", "기관합계")}}
    assert data_calls(session)[0][2]["params"] == {"count": 100, "until": "2026-11-06"}


def test_foreign_holding_from_investor_trading():
    api, session = make(token_ok(), data("investor_trading.json"))
    rows = api.foreign_holding("005930", D0, D1)
    assert rows[0] == {"date": date(2026, 11, 4), "외국인보유수량": 3_012_456_789,
                       "외국인지분율": 50.89, "한도소진율": pytest.approx(3_012_456_789 / 5_919_637_922 * 100)}
    assert rows[2] == {"date": date(2026, 11, 6), "외국인보유수량": None, "외국인지분율": None,
                       "한도소진율": None}
    api.investor_flow("005930", D0, D1)
    assert len(data_calls(session)) == 1


def test_investor_flow_market_amounts_in_won():
    api, session = make(token_ok(), data("market_investor_trading.json"))
    rows = api.investor_flow_market("KS", D0, D1)
    assert [r["date"] for r in rows] == [date(2026, 11, 4), date(2026, 11, 5), date(2026, 11, 6)]
    r = rows[0]
    assert list(r) == list(FLOW_KEYS)
    assert r["외국인"] == 199_999_999_999
    assert r["금융투자"] == 999_999_999 and r["연기금"] == 6_999_999_999
    assert r["기관합계"] == 27_999_999_993
    assert r["개인"] == -150_000_000_000 and r["기타법인"] == 30_000_000_000
    assert data_calls(session)[0][2]["params"] == {"interval": "1d", "count": 100, "until": "2026-11-06"}


def test_investor_flow_market_rejects_unknown_market():
    api, session = make(token_ok())
    with pytest.raises(ValueError):
        api.investor_flow_market("XX", D0, D1)
    assert session.calls == []


def test_shorting_keys_and_rates():
    api, _ = make(token_ok(), data("short_selling.json"))
    rows = api.shorting("005930", D0, D1)
    assert rows[-1] == {
        "date": date(2026, 11, 6), "공매도거래량": 512_300, "공매도거래대금": 41_250_000_000,
        "공매도잔고수량": None, "공매도잔고금액": None,
        "공매도거래량비중": pytest.approx(3.215), "공매도거래대금비중": pytest.approx(3.18),
    }
    assert rows[1]["공매도거래량비중"] is None and rows[1]["공매도거래대금비중"] is None
    assert rows[0]["공매도거래량"] == 0 and rows[0]["공매도거래량비중"] == 0


def test_program_trading_keys():
    api, _ = make(token_ok(), data("program_trades.json"))
    rows = api.program_trading("005930", D0, D1)
    assert rows[-1] == {
        "date": date(2026, 11, 6), "차익매수": 152_300, "차익매도": 183_400, "차익순매수": -31_100,
        "비차익매수": 1_210_500, "비차익매도": 1_105_200, "비차익순매수": 105_300,
    }


def test_credit_balance_keys_with_missing_side_none():
    api, _ = make(token_ok(), data("credit_trades.json"))
    rows = api.credit_balance("005930", D0, D1)
    assert rows[1] == {
        "date": date(2026, 11, 5), "융자신규": 125_300, "융자상환": 98_200, "융자잔고": 2_513_400,
        "융자잔고율": pytest.approx(0.42), "융자공여율": pytest.approx(9.0),
        "대주신규": 5_200, "대주상환": 3_100, "대주잔고": 45_200,
        "대주잔고율": pytest.approx(0.01), "대주공여율": pytest.approx(0.04),
    }
    assert rows[0]["대주잔고"] is None and rows[0]["대주신규"] is None
    assert rows[0]["융자잔고"] == 2_486_300


def test_securities_lending_keys():
    api, _ = make(token_ok(), data("securities_lending.json"))
    rows = api.securities_lending("005930", D0, D1)
    assert rows[-1] == {"date": date(2026, 11, 6), "대차체결": 210_500, "대차상환": 185_300,
                        "대차잔고수량": 15_234_000, "대차잔고금액": 1_218_720_000_000}


def test_paginates_with_next_until_and_stops_before_start():
    def page(days, nxt):
        return (200, {"result": {"nextUntil": nxt, "records": [
            {"date": d, "updatedAt": f"{d}T19:00:00+09:00", "executionQuantity": "1",
             "repaymentQuantity": "1", "balanceQuantity": "1", "balanceAmount": "1"} for d in days]}})

    api, session = make(
        token_ok(),
        page(["2026-11-06", "2026-11-05"], "2026-11-04"),
        page(["2026-11-04", "2026-11-03"], "2026-11-02"),
    )
    rows = api.securities_lending("005930", date(2026, 11, 3), D1)
    assert [r["date"].day for r in rows] == [3, 4, 5, 6]
    untils = [c[2]["params"]["until"] for c in data_calls(session)]
    assert untils == ["2026-11-06", "2026-11-04"]


def test_pagination_without_progress_is_format_error():
    stuck = (200, {"result": {"nextUntil": "2026-11-06", "records": [
        {"date": "2026-11-06", "updatedAt": "x", "executionQuantity": "1",
         "repaymentQuantity": "1", "balanceQuantity": "1", "balanceAmount": "1"}]}})
    api, _ = make(token_ok(), stuck)
    with pytest.raises(TossUnavailable):
        api.securities_lending("005930", date(2026, 1, 1), D1)


def test_start_after_end_is_value_error():
    api, session = make(token_ok())
    with pytest.raises(ValueError):
        api.shorting("005930", D1, D0)
    assert session.calls == []


def test_base_url_is_https_toss_host():
    assert BASE_URL == "https://openapi.tossinvest.com"
    assert toss.MIN_INTERVAL >= 0.15


def test_market_flow_missing_amount_makes_only_that_value_none():
    body = load("market_investor_trading.json")
    recs = body["result"]["records"]
    del recs[0]["foreigner"]["sellAmount"]  # 2026-11-06
    recs[1]["individual"] = None  # 2026-11-05
    recs[1]["institution"]["breakdown"] = None
    api, _ = make(token_ok(), (200, body))
    rows = api.investor_flow_market("KS", D0, D1)
    assert len(rows) == 3
    assert rows[2]["외국인"] is None and rows[2]["개인"] is not None
    assert rows[1]["개인"] is None and rows[1]["연기금"] is None
    assert rows[1]["기관합계"] is not None and rows[1]["외국인"] is not None
    assert rows[0]["외국인"] == 199_999_999_999
