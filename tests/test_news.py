from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from krta.data.news import GoogleNews, NewsItem
from krta.net import FetchError, HttpClient
from krta.redact import Redactor

FIXTURES = Path(__file__).parent / "fixtures" / "news"
KST = timezone(timedelta(hours=9))
AS_OF = date(2026, 11, 10)


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def make_news(*responses):
    session = FakeSession(responses)
    http = HttpClient(Redactor([]), None, session=session, sleep=lambda s: None)
    return GoogleNews(http), session


def fixture(name: str) -> FakeResponse:
    return FakeResponse(200, (FIXTURES / name).read_bytes())


def sent_url(session, i=0) -> str:
    method, url, kwargs = session.calls[i]
    return requests.Request(method, url, params=kwargs["params"]).prepare().url


def test_news_filters_future_and_dedupes():
    news, _ = make_news(fixture("dedupe_future.xml"))
    items = news.search("삼성전자", AS_OF)
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, NewsItem)
    assert item.title == "삼성전자, HBM4 공급 확대"
    assert item.source == "한국경제"
    assert item.link == "https://news.google.com/rss/articles/AAA1?oc=5"
    assert item.published == datetime(2026, 11, 9, 10, 0, tzinfo=KST)


def test_news_query_is_korean_locale():
    news, session = make_news(fixture("empty.xml"))
    news.search("삼성전자 반도체", AS_OF, days=30)
    url = sent_url(session)
    parts = urlsplit(url)
    assert (parts.scheme, parts.netloc, parts.path) == ("https", "news.google.com", "/rss/search")
    query = parse_qs(parts.query)
    assert query == {
        "q": ["삼성전자 반도체 after:2026-10-11 before:2026-11-11"],
        "hl": ["ko"],
        "gl": ["KR"],
        "ceid": ["KR:ko"],
    }


def test_news_default_window_is_14_days():
    news, session = make_news(fixture("empty.xml"))
    assert news.search("삼성전자", AS_OF) == []
    assert parse_qs(urlsplit(sent_url(session)).query)["q"] == ["삼성전자 after:2026-10-27 before:2026-11-11"]


def test_news_past_as_of_uses_absolute_date_window():
    news, session = make_news(fixture("empty.xml"))
    news.search("삼성전자", date(2026, 11, 6), days=14)
    assert parse_qs(urlsplit(sent_url(session)).query)["q"] == ["삼성전자 after:2026-10-23 before:2026-11-07"]


def test_news_past_as_of_still_cuts_at_end_of_day_kst():
    # Google's before: is day-granular; the 23:59:59 KST cut stays client-side.
    news, _ = make_news(fixture("dedupe_future.xml"))
    items = news.search("삼성전자", date(2026, 11, 9))
    assert [i.title for i in items] == ["삼성전자, HBM4 공급 확대"]


@pytest.mark.parametrize("query, days", [("", 14), ("   ", 14), ("x", 0), ("x", -1)])
def test_news_rejects_empty_query_and_nonpositive_days(query, days):
    news, session = make_news(fixture("empty.xml"))
    with pytest.raises(ValueError):
        news.search(query, AS_OF, days=days)
    assert session.calls == []


def test_news_newest_first_dedupes_across_outlets_and_drops_undated():
    news, _ = make_news(fixture("search.xml"))
    items = news.search("삼성전자", AS_OF)
    # Same headline from two outlets kept once (newest); 11-11 KST item and undated item dropped.
    assert [(i.title, i.source) for i in items] == [
        ("삼성전자 3분기 실적 발표", "연합뉴스"),
        ("삼성전자, HBM4 공급 확대", "한국경제"),
    ]
    assert items[0].published == datetime(2026, 11, 10, 0, 30, tzinfo=KST)


def test_news_cutoff_is_end_of_day_kst():
    news, _ = make_news(fixture("boundary.xml"))
    items = news.search("x", AS_OF)
    assert [i.title for i in items] == ["마감 직전 기사"]


def test_news_malformed_xml_is_fetch_error():
    news, _ = make_news(FakeResponse(200, b"<rss><channel><item>"))
    with pytest.raises(FetchError) as e:
        news.search("x", AS_OF)
    assert e.value.source == "google_news"


def test_news_cross_host_redirect_refused():
    news, session = make_news(FakeResponse(302, headers={"Location": "https://consent.google.com/ml?continue=x"}))
    with pytest.raises(FetchError) as e:
        news.search("x", AS_OF)
    assert e.value.source == "google_news" and len(session.calls) == 1


def test_news_same_host_redirect_followed():
    news, session = make_news(
        FakeResponse(302, headers={"Location": "/rss/search?q=x&hl=ko&gl=KR&ceid=KR:ko"}),
        fixture("boundary.xml"),
    )
    assert [i.title for i in news.search("x", AS_OF)] == ["마감 직전 기사"]
    assert len(session.calls) == 2
