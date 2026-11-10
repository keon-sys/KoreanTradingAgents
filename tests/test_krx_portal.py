import json
import os
from datetime import date
from pathlib import Path

import pytest
import requests

from krta.data.krx_portal import (
    FLOW_KEYS,
    JSON_URL,
    LOGIN_URL,
    KrxPortal,
    PortalUnavailable,
)
from krta.net import HttpClient
from krta.redact import Redactor
from krta.toollog import ToolLog

FIXTURES = Path(__file__).parent / "fixtures" / "krx_portal"
D0 = date(2026, 11, 4)
D1 = date(2026, 11, 6)
UID = "dummy-user"
UPW = "dummy-pass-0000"


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
            return json.loads(self._body)  # raises ValueError for non-JSON text
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


PAGE = (200, "<html></html>")
EXPIRED = (load("session_expired.json")["status"], load("session_expired.json")["body"])


def login_ok():
    return [PAGE, PAGE, (200, load("login_ok.json"))]


def finder():
    return (200, load("finder.json"))


def data(name):
    return (200, load(name))


def make(*responses, user=UID, pw=UPW, toollog=None, redactor=None, max_retries=0):
    session = FakeSession(*responses)
    kw = {} if max_retries is None else {"max_retries": max_retries}
    http = HttpClient(
        redactor if redactor is not None else Redactor([]),
        toollog,
        session=session,
        sleep=lambda s: None,
        **kw,
    )
    return KrxPortal(http, user, pw), session


def login_posts(session):
    return [c for c in session.calls if c[0] == "POST" and c[1] == LOGIN_URL]


def json_posts(session):
    return [c for c in session.calls if c[0] == "POST" and c[1] == JSON_URL]


@pytest.mark.parametrize(("user", "pw"), [(None, None), ("", UPW), (UID, None)])
def test_unavailable_without_credentials(user, pw):
    p, session = make(user=user, pw=pw)
    assert p.available() is False
    with pytest.raises(PortalUnavailable) as e:
        p.investor_flow("005930", D0, D1)
    assert "KRX_ID" in e.value.reason
    assert session.calls == []  # nothing is sent without credentials


def test_available_with_credentials():
    p, _ = make()
    assert p.available() is True


def test_login_request_shape():
    p, session = make(*login_ok(), finder(), data("investor_flow.json"))
    p.investor_flow("005930", D0, D1)
    ((_, _, kw),) = login_posts(session)
    assert kw["data"]["mbrId"] == UID and kw["data"]["pw"] == UPW
    assert "skipDup" not in kw["data"]
    # Warm-up pages precede the login POST.
    assert [c[0] for c in session.calls[:3]] == ["GET", "GET", "POST"]


def test_investor_flow_normalized():
    p, session = make(*login_ok(), finder(), data("investor_flow.json"))
    rows = p.investor_flow("005930", D0, D1)

    finder_call, flow_call = json_posts(session)
    assert finder_call[2]["data"]["bld"] == "dbms/comm/finder/finder_stkisu"
    sent = flow_call[2]["data"]
    assert sent["bld"] == "dbms/MDC/STAT/standard/MDCSTAT02303"
    assert sent["isuCd"] == "KR7005930003"
    assert (sent["strtDd"], sent["endDd"]) == ("20261104", "20261106")
    assert (sent["trdVolVal"], sent["askBid"], sent["detailView"]) == (2, 3, 1)

    assert [r["date"] for r in rows] == [date(2026, 11, 4), date(2026, 11, 5), date(2026, 11, 6)]
    for r in rows:
        assert set(r) == set(FLOW_KEYS)
        assert all(type(r[k]) is int for k in FLOW_KEYS if k != "date")
    last = rows[-1]
    assert last["금융투자"] == -35_210_000_000
    assert last["개인"] == 61_338_000_000
    # 외국인 = 외국인 + 기타외국인; 기관합계 = 금융투자..연기금.
    assert last["외국인"] == -32_900_000_000 - 100_000_000
    inst = ["금융투자", "보험", "투신", "사모", "은행", "기타금융", "연기금"]
    assert last["기관합계"] == sum(last[k] for k in inst)
    # Net buying over all investor groups sums to zero.
    for r in rows:
        assert r["외국인"] + r["기관합계"] + r["기타법인"] + r["개인"] == 0


def test_isin_lookup_cached_and_login_once():
    p, session = make(
        *login_ok(), finder(), data("investor_flow.json"), data("shorting.json"),
        data("foreign_holding.json"),
    )
    p.investor_flow("005930", D0, D1)
    p.shorting("005930", D0, D1)
    p.foreign_holding("005930", D0, D1)
    assert len(login_posts(session)) == 1
    assert len(json_posts(session)) == 4  # one finder lookup + three data queries


def test_unknown_code_is_unavailable():
    p, _ = make(*login_ok(), finder())
    with pytest.raises(PortalUnavailable) as e:
        p.investor_flow("999999", D0, D1)
    assert "999999" in e.value.reason


def test_shorting_balance_missing_is_none():
    p, session = make(*login_ok(), finder(), data("shorting.json"))
    rows = p.shorting("005930", D0, D1)
    sent = json_posts(session)[-1][2]["data"]
    assert sent["bld"] == "dbms/MDC/STAT/srt/MDCSTAT30001"
    assert sent["isuCd"] == "KR7005930003"
    assert [r["date"] for r in rows] == [date(2026, 11, 4), date(2026, 11, 5), date(2026, 11, 6)]
    for r in rows:
        assert set(r) == {"date", "공매도거래량", "공매도거래대금", "공매도잔고수량", "공매도잔고금액"}
    assert rows[0]["공매도잔고수량"] == 6_120_455
    assert rows[0]["공매도잔고금액"] == 602_864_817_500
    assert rows[-1]["공매도거래량"] == 512_340
    assert rows[-1]["공매도거래대금"] == 50_465_490_000
    assert rows[-1]["공매도잔고수량"] is None and rows[-1]["공매도잔고금액"] is None


def test_foreign_holding_normalized():
    p, session = make(*login_ok(), finder(), data("foreign_holding.json"))
    rows = p.foreign_holding("005930", D0, D1)
    sent = json_posts(session)[-1][2]["data"]
    assert sent["bld"] == "dbms/MDC/STAT/standard/MDCSTAT03702"
    assert sent["searchType"] == 2
    assert [set(r) for r in rows] == [{"date", "외국인보유수량", "외국인지분율", "한도소진율"}] * 3
    last = rows[-1]
    assert last["date"] == date(2026, 11, 6)
    assert last["외국인보유수량"] == 2_987_654_321
    assert last["외국인지분율"] == pytest.approx(50.47)
    assert last["한도소진율"] == pytest.approx(50.47)


@pytest.mark.parametrize(("market", "mkt_id"), [("KS", "STK"), ("KQ", "KSQ")])
def test_investor_flow_market(market, mkt_id):
    p, session = make(*login_ok(), data("investor_flow_market.json"))
    rows = p.investor_flow_market(market, D0, D1)
    (call,) = json_posts(session)  # no ISIN lookup for the market-wide query
    sent = call[2]["data"]
    assert sent["bld"] == "dbms/MDC/STAT/standard/MDCSTAT02203"
    assert sent["mktId"] == mkt_id
    assert (sent["trdVolVal"], sent["askBid"], sent["detailView"]) == (2, 3, 1)
    assert len(rows) == 3
    assert all(set(r) == set(FLOW_KEYS) for r in rows)
    assert rows[-1]["개인"] == 612_200_000_000


def test_invalid_market_rejected():
    p, session = make()
    with pytest.raises(ValueError):
        p.investor_flow_market("KN", D0, D1)
    assert session.calls == []


def test_rows_outside_range_are_dropped():
    p, _ = make(*login_ok(), finder(), data("investor_flow.json"))
    rows = p.investor_flow("005930", D0, date(2026, 11, 5))
    assert [r["date"] for r in rows] == [date(2026, 11, 4), date(2026, 11, 5)]


def test_relogin_once_on_session_expiry():
    p, session = make(
        *login_ok(), finder(), EXPIRED, *login_ok(), data("investor_flow.json"),
    )
    rows = p.investor_flow("005930", D0, D1)
    assert len(rows) == 3
    assert len(login_posts(session)) == 2


def test_relogin_on_200_logout_text():
    p, session = make(
        *login_ok(), finder(), (200, "LOGOUT"), *login_ok(), data("investor_flow.json"),
    )
    assert len(p.investor_flow("005930", D0, D1)) == 3
    assert len(login_posts(session)) == 2


def test_still_expired_after_relogin_is_unavailable():
    p, session = make(*login_ok(), finder(), EXPIRED, *login_ok(), EXPIRED)
    with pytest.raises(PortalUnavailable) as e:
        p.investor_flow("005930", D0, D1)
    assert "세션" in e.value.reason
    assert len(login_posts(session)) == 2


def test_duplicate_login_retries_with_skipdup():
    p, session = make(
        PAGE, PAGE, (200, load("login_dup.json")), (200, load("login_ok.json")),
        finder(), data("investor_flow.json"),
    )
    assert len(p.investor_flow("005930", D0, D1)) == 3
    first, second = login_posts(session)
    assert "skipDup" not in first[2]["data"]
    assert second[2]["data"]["skipDup"] == "Y"


def test_login_failure_is_sticky():
    """A rejected login is never retried: repeated wrong passwords can lock the account."""
    p, session = make(PAGE, PAGE, (200, load("login_fail.json")))
    with pytest.raises(PortalUnavailable) as e1:
        p.investor_flow("005930", D0, D1)
    assert "CD006" in e1.value.reason and "1/5" in e1.value.reason
    with pytest.raises(PortalUnavailable):
        p.shorting("005930", D0, D1)
    assert len(login_posts(session)) == 1


def test_login_response_not_json_is_unavailable():
    p, _ = make(PAGE, PAGE, (200, "<html>점검중</html>"))
    with pytest.raises(PortalUnavailable) as e:
        p.investor_flow("005930", D0, D1)
    assert "로그인" in e.value.reason


def test_login_http_error_is_unavailable():
    p, _ = make(PAGE, PAGE, (503, "busy"))
    with pytest.raises(PortalUnavailable) as e:
        p.investor_flow("005930", D0, D1)
    assert "503" in e.value.reason


def test_data_http_error_is_unavailable():
    p, _ = make(*login_ok(), finder(), (500, "error"))
    with pytest.raises(PortalUnavailable) as e:
        p.investor_flow("005930", D0, D1)
    assert "500" in e.value.reason


@pytest.mark.parametrize(
    "body",
    [
        {"OutBlock_1": []},  # block renamed
        {"output": [{"TRD_DD": "2026/11/06", "TRDVAL1": "1"}]},  # columns missing
        {"output": [{**load("investor_flow.json")["output"][0], "TRDVAL1": "abc"}]},
        {"output": [{**load("investor_flow.json")["output"][0], "TRD_DD": "20261106"}]},
        "not json at all",
    ],
)
def test_format_change_is_unavailable(body):
    p, _ = make(*login_ok(), finder(), (200, body))
    with pytest.raises(PortalUnavailable) as e:
        p.investor_flow("005930", D0, D1)
    assert "형식" in e.value.reason
    assert e.value.__cause__ is None and e.value.__context__ is None


def test_empty_output_is_empty_list():
    p, _ = make(*login_ok(), finder(), (200, {"output": []}))
    assert p.investor_flow("005930", D0, D1) == []


def test_login_failure_reason_has_no_secrets(tmp_path):
    secret_id, secret_pw = "myid99", "mypw-SECRET"
    # Empty redactor: the module itself must keep credentials out, not rely on masking.
    log = ToolLog(tmp_path / "message_tool.log", Redactor([]))
    p, _ = make(
        PAGE, PAGE, (200, load("login_fail.json")),
        user=secret_id, pw=secret_pw, toollog=log,
    )
    with pytest.raises(PortalUnavailable) as e:
        p.investor_flow("005930", D0, D1)
    text = (tmp_path / "message_tool.log").read_text(encoding="utf-8")
    assert "[request] krx_portal" in text
    for s in [secret_id, secret_pw]:
        assert s not in str(e.value) and s not in e.value.reason and s not in text
        assert s not in repr(e.value)
    assert e.value.__cause__ is None and e.value.__context__ is None


def test_credentials_never_logged_on_success(tmp_path):
    log = ToolLog(tmp_path / "message_tool.log", Redactor([]))
    p, _ = make(
        *login_ok(), finder(), EXPIRED, *login_ok(), data("investor_flow.json"),
        toollog=log,
    )
    p.investor_flow("005930", D0, D1)
    text = (tmp_path / "message_tool.log").read_text(encoding="utf-8")
    assert UID not in text and UPW not in text
    assert "MDCSTAT02303" in text  # data queries are logged with their parameters


def test_no_cookie_file_written(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    log = ToolLog(tmp_path / "message_tool.log", Redactor([]))
    p, _ = make(
        *login_ok(), finder(), data("investor_flow.json"), data("shorting.json"),
        data("foreign_holding.json"), data("investor_flow_market.json"), toollog=log,
    )
    p.investor_flow("005930", D0, D1)
    p.shorting("005930", D0, D1)
    p.foreign_holding("005930", D0, D1)
    p.investor_flow_market("KS", D0, D1)
    created = sorted(
        str(Path(root, f).relative_to(tmp_path))
        for root, _, files in os.walk(tmp_path)
        for f in files
    )
    assert created == ["message_tool.log"]


def test_start_after_end_rejected():
    p, session = make()
    with pytest.raises(ValueError):
        p.investor_flow("005930", D1, D0)
    assert session.calls == []


# --- account-lock protection with the client's DEFAULT retry settings ----------------


@pytest.mark.parametrize("failure", [(503, "busy"), (429, "slow down"), requests.Timeout("t")])
def test_login_post_never_resent_by_transport_retries(failure):
    # max_retries=None: HttpClient default (4 retries) for everything except the login POST.
    p, session = make(PAGE, PAGE, failure, max_retries=None)
    with pytest.raises(PortalUnavailable):
        p.investor_flow("005930", D0, D1)
    assert len(login_posts(session)) == 1
    with pytest.raises(PortalUnavailable):
        p.shorting("005930", D0, D1)  # sticky: nothing more is sent
    assert len(login_posts(session)) == 1


def test_login_posts_capped_across_repeated_session_expiry():
    p, session = make(
        *login_ok(), finder(), EXPIRED, *login_ok(), data("investor_flow.json"),
        # A third login would succeed if it were sent; the cap must stop it anyway.
        EXPIRED, *login_ok(), data("shorting.json"),
        max_retries=None,
    )
    assert len(p.investor_flow("005930", D0, D1)) == 3
    with pytest.raises(PortalUnavailable) as e:
        p.shorting("005930", D0, D1)
    assert "로그인 횟수" in e.value.reason
    with pytest.raises(PortalUnavailable):
        p.foreign_holding("005930", D0, D1)
    assert len(login_posts(session)) == 2
    assert session.responses[0] == PAGE  # the third login's warm-up was never sent


def test_skipdup_resend_counts_toward_login_cap():
    p, session = make(
        PAGE, PAGE, (200, load("login_dup.json")), (200, load("login_ok.json")),
        finder(), EXPIRED, max_retries=None,
    )
    with pytest.raises(PortalUnavailable):
        p.investor_flow("005930", D0, D1)
    assert len(login_posts(session)) == 2
