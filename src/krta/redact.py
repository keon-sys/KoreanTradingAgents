"""Mask secret values before text reaches logs, reports, exceptions or prompts."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from urllib.parse import quote, quote_plus

MASK = "***"

# Env var names whose values are treated as secrets.
SECRET_ENV_PATTERN = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|_PW|_ID)$")

# Values shorter than this are not treated as secrets (too likely to collide with normal text).
MIN_SECRET_LEN = 4

# URL query parameters whose values are always masked, regardless of the known secret list.
_URL_KEY_PARAM = re.compile(
    r"(?P<prefix>[?&](?:crtfc_key|AUTH_KEY|api_key|key)=)[^&#\s\"'<>]*",
    re.IGNORECASE,
)


class Redactor:
    def __init__(self, secrets: Iterable[str]):
        self._variants: set[str] = set()
        self._pattern: re.Pattern[str] | None = None
        for s in secrets:
            self._register(s)
        self._compile()

    def add(self, secret: str) -> None:
        """Register a secret learned at run time (e.g. an issued access token).

        Same rules as the constructor: values shorter than MIN_SECRET_LEN are ignored.
        """
        if self._register(secret):
            self._compile()

    def _register(self, s: str | None) -> bool:
        if not s or len(s) < MIN_SECRET_LEN:
            return False
        before = len(self._variants)
        self._variants.add(s)
        # Secrets also appear percent-encoded in URLs and form bodies.
        self._variants.add(quote(s, safe=""))
        self._variants.add(quote_plus(s, safe=""))
        return len(self._variants) != before

    def _compile(self) -> None:
        # Longest first so a secret that contains another is replaced whole.
        ordered = sorted(self._variants, key=len, reverse=True)
        self._pattern = re.compile("|".join(map(re.escape, ordered))) if ordered else None

    def redact(self, text: str) -> str:
        if not text:
            return text
        if self._pattern is not None:
            text = self._pattern.sub(MASK, text)
        return _URL_KEY_PARAM.sub(lambda m: m.group("prefix") + MASK, text)

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Redactor:
        return cls(
            value
            for name, value in env.items()
            if SECRET_ENV_PATTERN.search(name) and value and len(value) >= MIN_SECRET_LEN
        )
