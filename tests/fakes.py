"""Test doubles shared across test modules."""

from __future__ import annotations

import threading
from collections.abc import Callable

from pydantic import BaseModel


class FakeLLM:
    """Answers with `responder(system, user)`; records every (system, user) pair.

    `quick_flags[i]` is the `quick` argument of call i (None for `structured`). Calls may come
    from parallel graph nodes, so both lists are appended together under a lock.
    """

    def __init__(self, responder: Callable[[str, str], str]):
        self.responder = responder
        self.calls: list[tuple[str, str]] = []
        self.quick_flags: list[bool | None] = []
        self._lock = threading.Lock()

    def _record(self, system: str, user: str, quick: bool | None) -> None:
        with self._lock:
            self.calls.append((system, user))
            self.quick_flags.append(quick)

    def complete(self, system: str, user: str, *, quick: bool) -> str:
        self._record(system, user, quick)
        return self.responder(system, user)

    def structured(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel | None:
        """`responder` returns JSON for `schema`, or "null" for no structured output."""
        self._record(system, user, None)
        text = self.responder(system, user)
        return None if text == "null" else schema.model_validate_json(text)
