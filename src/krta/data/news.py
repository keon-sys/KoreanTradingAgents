"""Korean-language headlines from the Google News RSS search feed (ko/KR edition).

Headlines and outlet only (the feed carries no article body). Items published after the
analysis date's 23:59:59 KST are dropped, as are undated ones (they cannot be checked).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime

from krta.net import FetchError, HttpClient

SOURCE = "google_news"
SEARCH_URL = "https://news.google.com/rss/search"
KST = timezone(timedelta(hours=9))

_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class NewsItem:
    title: str
    source: str
    published: datetime  # KST, timezone-aware
    link: str


class GoogleNews:
    def __init__(self, http: HttpClient):
        self._http = http

    def search(self, query: str, as_of: date, days: int = 14) -> list[NewsItem]:
        """Newest first, one item per headline, from the `days` days up to `as_of`.

        The window is absolute (`after:`/`before:` dates), so a past `as_of` works too;
        the exact 23:59:59 KST cut is applied here since Google's date operators are
        day-granular.
        """
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if days <= 0:
            raise ValueError(f"days must be positive: {days}")
        after = (as_of - timedelta(days=days)).isoformat()
        before = (as_of + timedelta(days=1)).isoformat()
        resp = self._http.get(
            SEARCH_URL,
            params={
                "q": f"{query} after:{after} before:{before}",
                "hl": "ko",
                "gl": "KR",
                "ceid": "KR:ko",
            },
            source=SOURCE,
            # Same-host hops only (e.g. URL normalisation); a consent-page bounce to
            # another host is still refused by HttpClient.
            follow_same_host_redirects=True,
        )
        cutoff = datetime.combine(as_of, time(23, 59, 59), tzinfo=KST)
        items = [i for i in _parse(resp.content) if i.published <= cutoff]
        items.sort(key=lambda i: i.published, reverse=True)
        seen: set[str] = set()
        unique = []
        for item in items:
            if item.title not in seen:
                seen.add(item.title)
                unique.append(item)
        return unique


def _parse(content: bytes) -> list[NewsItem]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        error = f"RSS 파싱 실패: {exc}"
    else:
        error = None
    if error is not None:
        raise FetchError(SOURCE, error) from None
    items = []
    for node in root.iter("item"):
        published = _published(node.findtext("pubDate"))
        title = _clean(node.findtext("title"))
        if published is None or not title:
            continue
        source = _clean(node.findtext("source"))
        items.append(
            NewsItem(
                title=_strip_outlet(title, source),
                source=source,
                published=published,
                link=_clean(node.findtext("link")),
            )
        )
    return items


def _published(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        value = parsedate_to_datetime(text.strip())
    except (TypeError, ValueError):
        return None
    if value.tzinfo is None:  # RFC 822 "-0000": treat as UTC
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(KST)


def _clean(text: str | None) -> str:
    return _SPACES.sub(" ", text or "").strip()


def _strip_outlet(title: str, source: str) -> str:
    """Google appends " - <outlet>"; drop it so one story from many outlets dedupes."""
    suffix = f" - {source}"
    if source and title.endswith(suffix) and len(title) > len(suffix):
        return title[: -len(suffix)].strip()
    return title
