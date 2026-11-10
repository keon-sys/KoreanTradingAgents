import pytest
import requests

from krta.net import FetchError, HttpClient
from krta.redact import Redactor
from krta.toollog import ToolLog


class FakeResponse:
    def __init__(self, status_code: int, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class FakeSession:
    """Stands in for requests.Session: replays a script of statuses or exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item if isinstance(item, FakeResponse) else FakeResponse(item)


def make_client(script, *, toollog=None, redactor=None, **kw):
    sleeps = []
    session = FakeSession(script)
    client = HttpClient(redactor or Redactor([]), toollog, session=session, sleep=sleeps.append, **kw)
    return client, session, sleeps


def test_retries_429_then_succeeds():
    client, session, sleeps = make_client([429, 429, 200])
    resp = client.get("https://x/", source="s")
    assert resp.status_code == 200
    assert sleeps == [1.0, 2.0]
    assert len(session.calls) == 3


def test_gives_up_after_max_retries():
    client, session, sleeps = make_client([503], max_retries=3)
    with pytest.raises(FetchError) as e:
        client.get("https://x/", source="s")
    assert len(session.calls) == 3 + 1
    assert sleeps == [1.0, 2.0, 4.0]
    assert e.value.source == "s" and e.value.status == 503


def test_retries_timeout_and_connection_error():
    client, session, sleeps = make_client(
        [requests.Timeout("t"), requests.ConnectionError("c"), ConnectionError("b"), 200]
    )
    assert client.get("https://x/", source="s").status_code == 200
    assert len(session.calls) == 4


def test_client_error_fails_immediately():
    client, session, sleeps = make_client([404, 200])
    with pytest.raises(FetchError) as e:
        client.get("https://x/", source="s")
    assert len(session.calls) == 1 and sleeps == []
    assert "404" in str(e.value) and e.value.status == 404


def test_fetch_error_message_is_redacted():
    leak = "https://x/?crtfc_key=SECRET1234"
    client, _, _ = make_client([requests.ConnectionError(f"Max retries exceeded with url: {leak}")], max_retries=1)
    with pytest.raises(FetchError) as e:
        client.get("https://x/", params={"crtfc_key": "SECRET1234"}, source="dart")
    assert "SECRET1234" not in str(e.value) and e.value.__cause__ is None
    # The original exception must not be reachable through the implicit chain either.
    assert e.value.__context__ is None


def test_fetch_error_redacts_known_secret_in_status_error():
    client, _, _ = make_client([401], redactor=Redactor(["PATHSECRET9"]))
    with pytest.raises(FetchError) as e:
        client.get("https://x/a/PATHSECRET9/b", source="s")
    assert "PATHSECRET9" not in str(e.value)


def test_other_request_errors_are_wrapped_and_redacted():
    exc = requests.exceptions.InvalidURL("bad url https://x/?key=SECRETKEY77")
    client, session, _ = make_client([exc])
    with pytest.raises(FetchError) as e:
        client.get("https://x/", source="s")
    assert "SECRETKEY77" not in str(e.value) and e.value.__context__ is None
    assert len(session.calls) == 1


def test_rejects_non_https():
    client, session, _ = make_client([200])
    with pytest.raises(FetchError):
        client.get("http://x/", source="s")
    assert session.calls == []


def test_request_log_has_redacted_url_and_status_only(tmp_path):
    log = tmp_path / "message_tool.log"
    tl = ToolLog(log, Redactor([]))
    client, _, _ = make_client([200], toollog=tl)
    client.get(
        "https://x/api",
        params={"crtfc_key": "SECRET1234", "corp": "1"},
        headers={"AUTH_KEY": "HEADERSECRET"},
        source="dart",
    )
    text = log.read_text(encoding="utf-8")
    assert "[request] dart" in text
    assert "GET https://x/api?crtfc_key=***&corp=1 -> 200" in text
    assert "SECRET1234" not in text and "HEADERSECRET" not in text


def test_post_body_logged_when_enabled(tmp_path):
    log = tmp_path / "message_tool.log"
    client, session, _ = make_client([200], toollog=ToolLog(log, Redactor([])))
    client.post("https://x/p", data={"q": "hello"}, source="s")
    assert "q=hello" in log.read_text(encoding="utf-8")
    assert session.calls[0][2]["data"] == {"q": "hello"}


def test_post_body_not_logged_when_disabled(tmp_path):
    log = tmp_path / "message_tool.log"
    client, _, _ = make_client([200], toollog=ToolLog(log, Redactor([])))
    client.post("https://x/login", data={"pw": "PWVALUE"}, source="krx_portal", log_body=False)
    text = log.read_text(encoding="utf-8")
    assert "PWVALUE" not in text and "[request] krx_portal" in text


def test_min_interval_between_same_source_requests():
    now = [100.0]
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        now[0] += s

    client = HttpClient(Redactor([]), None, session=FakeSession([200]), sleep=fake_sleep, min_interval=0.5)
    client._clock = lambda: now[0]
    client.get("https://x/", source="a")
    client.get("https://x/", source="b")  # different source: no wait
    assert sleeps == []
    now[0] += 0.2
    client.get("https://x/", source="a")
    assert sleeps == [pytest.approx(0.3)]


def redirect(status, location):
    return FakeResponse(status, {"Location": location})


def test_redirects_never_delegated_to_requests():
    client, session, _ = make_client([200])
    client.get("https://x/", source="s")
    assert session.calls[0][2]["allow_redirects"] is False


def test_redirect_not_followed_by_default():
    client, session, _ = make_client([redirect(302, "/next"), 200])
    with pytest.raises(FetchError) as e:
        client.get("https://x/a", source="s")
    assert e.value.status == 302 and len(session.calls) == 1


def test_redirect_to_other_host_rejected_and_header_not_sent():
    client, session, _ = make_client([redirect(302, "https://evil.example/steal?key=LEAKKEY55"), 200])
    with pytest.raises(FetchError) as e:
        client.get("https://x/a", headers={"AUTH_KEY": "HDRSECRET1"}, source="krx", follow_same_host_redirects=True)
    assert "LEAKKEY55" not in str(e.value)
    assert [c[1] for c in session.calls] == ["https://x/a"]  # nothing sent to evil.example


def test_redirect_to_http_same_host_rejected():
    client, session, _ = make_client([redirect(301, "http://x/a"), 200])
    with pytest.raises(FetchError):
        client.get("https://x/a", source="s", follow_same_host_redirects=True)
    assert len(session.calls) == 1


def test_redirect_to_other_port_rejected():
    client, session, _ = make_client([redirect(302, "https://x:8443/a"), 200])
    with pytest.raises(FetchError):
        client.get("https://x/a", source="s", follow_same_host_redirects=True)
    assert len(session.calls) == 1


def test_same_host_https_redirect_followed_and_logged(tmp_path):
    log = tmp_path / "message_tool.log"
    client, session, _ = make_client(
        [redirect(302, "/b?api_key=QSECRET77"), 200], toollog=ToolLog(log, Redactor([]))
    )
    resp = client.post(
        "https://x/login", data={"pw": "PWVALUE"}, headers={"H": "v"}, source="krx_portal",
        log_body=False, follow_same_host_redirects=True,
    )
    assert resp.status_code == 200
    method, url, kw = session.calls[1]
    # 302 after POST becomes GET without body; the Location query is used as-is.
    assert (method, url, kw["data"], kw["params"]) == ("GET", "https://x/b?api_key=QSECRET77", None, None)
    text = log.read_text(encoding="utf-8")
    assert "POST https://x/login -> 302" in text and "GET https://x/b?api_key=*** -> 200" in text
    assert "QSECRET77" not in text and "PWVALUE" not in text


def test_307_redirect_keeps_method_and_body():
    client, session, _ = make_client([redirect(307, "https://x/c"), 200])
    client.post("https://x/p", data={"q": "1"}, source="s", follow_same_host_redirects=True)
    assert session.calls[1][:2] == ("POST", "https://x/c") and session.calls[1][2]["data"] == {"q": "1"}


def test_redirect_hops_bounded():
    client, session, _ = make_client([redirect(302, "/loop")])
    with pytest.raises(FetchError):
        client.get("https://x/a", source="s", follow_same_host_redirects=True)
    assert len(session.calls) == 6  # initial + MAX_REDIRECTS hops


@pytest.mark.parametrize("item", [503, 429, requests.Timeout("t"), requests.ConnectionError("c")])
def test_per_call_max_retries_zero_sends_once(item):
    client, session, sleeps = make_client([item])  # client default: 4 retries
    with pytest.raises(FetchError):
        client.post("https://x/login", data={"pw": "PWVALUE"}, source="s", log_body=False, max_retries=0)
    assert len(session.calls) == 1 and sleeps == []


def test_per_call_max_retries_overrides_both_ways():
    client, session, _ = make_client([503], max_retries=0)
    with pytest.raises(FetchError):
        client.get("https://x/", source="s", max_retries=2)
    assert len(session.calls) == 3
    # None keeps the client default.
    client, session, _ = make_client([503])
    with pytest.raises(FetchError):
        client.get("https://x/", source="s", max_retries=None)
    assert len(session.calls) == 5
