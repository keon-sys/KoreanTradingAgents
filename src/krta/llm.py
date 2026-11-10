"""LLM interface used by data collection and the agents, and its Anthropic implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from krta.config import Settings
from krta.toollog import ToolLog

LOG_PROMPT_CHARS = 2000
# Seconds per Anthropic request. Without it a stalled connection hangs until the Job's
# activeDeadlineSeconds kills the whole run.
REQUEST_TIMEOUT = 300.0


class LLM(Protocol):
    def complete(self, system: str, user: str, *, quick: bool) -> str:
        """Plain-text answer; `quick` selects the fast model, else the deep one."""
        ...

    def structured(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel | None:
        """Answer parsed into `schema` (deep model); None when the model gave no structured output."""
        ...


def _chat_anthropic(model: str, max_retries: int) -> Any:
    # The API key is read by the library from ANTHROPIC_API_KEY; it is never passed here.
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, max_retries=max_retries, timeout=REQUEST_TIMEOUT)


class AnthropicLLM:
    """Claude via langchain-anthropic; every call is logged to `message_tool.log`."""

    def __init__(
        self,
        settings: Settings,
        toollog: ToolLog,
        *,
        chat_factory: Callable[[str, int], Any] = _chat_anthropic,
    ):
        self._toollog = toollog
        self._quick_model = settings.quick_model
        self._deep_model = settings.deep_model
        self._quick = chat_factory(settings.quick_model, settings.llm_max_retries)
        self._deep = chat_factory(settings.deep_model, settings.llm_max_retries)

    @staticmethod
    def _messages(system: str, user: str) -> list:
        return [SystemMessage(content=system), HumanMessage(content=user)]

    def _log(self, model: str, system: str, user: str, response: str) -> None:
        self._toollog.write(
            "agent",
            f"llm {model}",
            f"[system]\n{system[:LOG_PROMPT_CHARS]}\n[user]\n{user[:LOG_PROMPT_CHARS]}\n"
            f"[response]\n{response}",
        )

    def complete(self, system: str, user: str, *, quick: bool) -> str:
        chat, model = (self._quick, self._quick_model) if quick else (self._deep, self._deep_model)
        text = chat.invoke(self._messages(system, user)).text
        self._log(model, system, user, text)
        return text

    def structured(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel | None:
        result = self._deep.with_structured_output(schema).invoke(self._messages(system, user))
        self._log(self._deep_model, system, user,
                  "(구조화 출력 없음)" if result is None else result.model_dump_json())
        return result
