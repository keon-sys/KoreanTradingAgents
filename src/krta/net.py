"""HTTP client for every external request: retry with backoff, redacted logs and errors."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests

from krta.redact import Redactor
from krta.toollog import ToolLog

# (connect, read) seconds.
DEFAULT_TIMEOUT = (10.0, 60.0)
# Same-host https redirect hops followed on opt-in.
MAX_REDIRECTS = 5

_RETRYABLE_EXC = (requests.Timeout, requests.ConnectionError, ConnectionError)


class FetchError(Exception):
    """Request failed. `message` must already be redacted by the caller.

    `status` is the last HTTP status code when the failure was an HTTP response
    (e.g. callers map 401 to an auth error), else None.
    """

    def __init__(self, source: str, message: str, status: int | None = None):
        super().__init__(f"[{source}] {message}")
        self.source = source
        self.message = message
        self.status = status


class HttpClient:
    def __init__(
        self,
        redactor: Redactor,
        toollog: ToolLog | None,
        session: requests.Session | None = None,
        max_retries: int = 4,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        min_interval: float = 0.0,
    ):
        self._redactor = redactor
        self._toollog = toollog
        self._session = session if session is not None else requests.Session()
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._sleep = sleep
        self._min_interval = min_interval
        self._clock: Callable[[], float] = time.monotonic
        self._last_request: dict[str, float] = {}

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        source: str,
        follow_same_host_redirects: bool = False,
        max_retries: int | None = None,
    ) -> requests.Response:
        """`max_retries` overrides the client default for this call (None = default)."""
        return self._request(
            "GET", url, source=source, params=params, headers=headers, log_body=False,
            follow=follow_same_host_redirects, max_retries=max_retries,
        )

    def post(
        self,
        url: str,
        *,
        data: Any = None,
        headers: Mapping[str, str] | None = None,
        source: str,
        log_body: bool = True,
        follow_same_host_redirects: bool = False,
        max_retries: int | None = None,
    ) -> requests.Response:
        """`max_retries` overrides the client default for this call (None = default);
        pass 0 for requests that must never be resent, e.g. one carrying credentials."""
        return self._request(
            "POST", url, source=source, data=data, headers=headers, log_body=log_body,
            follow=follow_same_host_redirects, max_retries=max_retries,
        )

    def _request(
        self,
        method: str,
        url: str,
        *,
        source: str,
        headers: Mapping[str, str] | None,
        log_body: bool,
        follow: bool,
        params: Mapping[str, Any] | None = None,
        data: Any = None,
        max_retries: int | None = None,
    ) -> requests.Response:
        # Redirects are never followed by requests itself: it would resend custom headers
        # (AUTH_KEY) to other hosts or over http://, and 307/308 resend POST bodies.
        # Only same-host https hops are followed, on opt-in, each one checked and logged.
        origin: tuple[str, int] | None = None
        for hop in range(MAX_REDIRECTS + 1):
            raw_url, shown_url, shown_body, error = self._prepare(method, url, params, data)
            parts = urlsplit(raw_url) if error is None else None
            if parts is not None and parts.scheme != "https":
                error = f"HTTPS가 아닌 요청 거부: {shown_url}"
            elif parts is not None:
                try:
                    host = ((parts.hostname or "").lower(), parts.port or 443)
                except ValueError:  # malformed port
                    host = ("", -1)
                if origin is None:
                    origin = host
                elif host != origin:
                    error = f"다른 호스트로의 리다이렉트 거부: {shown_url}"
            if error is not None:
                raise FetchError(source, self._redactor.redact(error)) from None

            resp = self._send_with_retry(
                method, url, source=source, params=params, data=data, headers=headers,
                prefix=f"{method} {shown_url}",
                body_suffix=f"\n{shown_body}" if log_body and shown_body else "",
                max_retries=self._max_retries if max_retries is None else max_retries,
            )
            if not 300 <= resp.status_code < 400:
                return resp
            location = resp.headers.get("Location") if resp.headers is not None else None
            if not follow or not location:
                error = f"HTTP {resp.status_code} 리다이렉트 미허용: {shown_url}"
                raise FetchError(source, self._redactor.redact(error), status=resp.status_code) from None
            # Location already carries its own query; mirror browser semantics for the method.
            url, params = urljoin(raw_url, location), None
            if resp.status_code in (301, 302, 303) and method != "GET":
                method, data = "GET", None
        error = f"리다이렉트 {MAX_REDIRECTS}회 초과: {method} {self._redactor.redact(url)}"
        raise FetchError(source, self._redactor.redact(error)) from None

    def _send_with_retry(
        self,
        method: str,
        url: str,
        *,
        source: str,
        params: Mapping[str, Any] | None,
        data: Any,
        headers: Mapping[str, str] | None,
        prefix: str,
        body_suffix: str,
        max_retries: int,
    ) -> requests.Response:
        last_error = ""
        last_status: int | None = None
        for attempt in range(max_retries + 1):
            if attempt:
                self._sleep(self._base_delay * 2 ** (attempt - 1))
            self._wait_turn(source)
            try:
                resp = self._session.request(
                    method, url, params=params, data=data, headers=headers,
                    timeout=DEFAULT_TIMEOUT, allow_redirects=False,
                )
            except _RETRYABLE_EXC as exc:
                last_status = None
                last_error = self._exc_text(exc)
                self._log(source, f"{prefix} -> {type(exc).__name__}{body_suffix}")
                continue
            except requests.RequestException as exc:
                # Not retryable; still never let the raw exception (URL, body) propagate.
                last_status = None
                last_error = self._exc_text(exc)
                self._log(source, f"{prefix} -> {type(exc).__name__}{body_suffix}")
                break

            status = last_status = resp.status_code
            self._log(source, f"{prefix} -> {status}{body_suffix}")
            if status == 429 or status >= 500:
                last_error = f"HTTP {status}: {prefix}"
                continue
            if status >= 400:
                last_error = f"HTTP {status}: {prefix}"
                break
            return resp
        else:
            last_error = f"재시도 {max_retries}회 소진, 마지막 오류 {last_error}"

        # Raised outside any `except` block so neither __cause__ nor __context__
        # carries the original exception (and its unredacted URL/body).
        raise FetchError(source, self._redactor.redact(last_error), status=last_status) from None

    def _prepare(
        self, method: str, url: str, params: Mapping[str, Any] | None, data: Any
    ) -> tuple[str, str, str, str | None]:
        """(raw url, redacted url, redacted body, redacted error).

        The logged form is the prepared URL/body string, never repr() of the params dict:
        the redactor's URL-key rule only recognises `?k=v` / `&k=v`. Headers are never logged.
        """
        try:
            prepared = requests.Request(method, url, params=params, data=data).prepare()
        except requests.RequestException as exc:
            return "", "", "", self._exc_text(exc)
        body = prepared.body or ""
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        raw = prepared.url or ""
        return raw, self._redactor.redact(raw), self._redactor.redact(body), None

    def _exc_text(self, exc: BaseException) -> str:
        return self._redactor.redact(f"{type(exc).__name__}: {exc}")

    def _wait_turn(self, source: str) -> None:
        if self._min_interval > 0:
            last = self._last_request.get(source)
            if last is not None:
                wait = last + self._min_interval - self._clock()
                if wait > 0:
                    self._sleep(wait)
        self._last_request[source] = self._clock()

    def _log(self, source: str, payload: str) -> None:
        if self._toollog is not None:
            self._toollog.write("request", source, payload)
