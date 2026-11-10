"""Runtime settings read from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = "~/.krta"


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ValueError(f"환경변수 {name}가 비어 있습니다")
    return value


def _int(env: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"환경변수 {name}는 정수여야 합니다") from None
    if value < minimum:
        raise ValueError(f"환경변수 {name}는 {minimum} 이상이어야 합니다")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    quick_model: str
    deep_model: str
    debate_rounds: int
    risk_rounds: int
    llm_max_retries: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        data_dir = env.get("KRTA_DATA_DIR", "").strip() or DEFAULT_DATA_DIR
        return cls(
            data_dir=Path(data_dir).expanduser(),
            quick_model=_required(env, "KRTA_QUICK_MODEL"),
            deep_model=_required(env, "KRTA_DEEP_MODEL"),
            debate_rounds=_int(env, "KRTA_DEBATE_ROUNDS", 1, minimum=1),
            risk_rounds=_int(env, "KRTA_RISK_ROUNDS", 1, minimum=1),
            llm_max_retries=_int(env, "KRTA_LLM_MAX_RETRIES", 6, minimum=0),
        )
