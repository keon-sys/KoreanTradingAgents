"""Risk debate: aggressive -> conservative -> neutral, `risk_rounds` times."""

from __future__ import annotations

from typing import TYPE_CHECKING

from krta.agents.debate import data_block, stage_context, stage_system
from krta.llm import LLM

if TYPE_CHECKING:
    from krta.graph import RunState

AGGRESSIVE = "공격적 리스크 분석가"
CONSERVATIVE = "보수적 리스크 분석가"
NEUTRAL = "중립적 리스크 분석가"

STANCES = {
    AGGRESSIVE: "기대수익이 위험을 감수할 만하면 더 적극적인 진입·비중을 옹호한다. "
                "트레이더 계획이 지나치게 보수적이어서 놓치는 기회를 짚는다.",
    CONSERVATIVE: "손실 방어를 우선한다. 트레이더 계획에서 과소평가된 하방 위험과 "
                  "손절·비중을 더 보수적으로 잡아야 하는 이유를 짚는다.",
    NEUTRAL: "공격적·보수적 의견을 저울질해 균형 잡힌 수정안을 낸다. "
             "어느 한쪽 주장이 데이터로 뒷받침되지 않으면 그 점을 짚는다.",
}

KR_RISKS = """\
- 반드시 한국장 리스크 관점에서 트레이더 계획을 검토한다:
  - 가격제한폭(±30%): 상·하한가 도달 시 체결 불가, 다음 날 갭, 손절가가 실제로 체결될 수 있는지.
  - 유동성: 거래대금 수준, 분할 체결 가능성, 호가 공백.
  - 만기일 변동성: 선물옵션 동시 만기 등 이벤트 캘린더의 만기일 전후 변동.
  - 환율: 원/달러 수준·변화와 외국인 수급에 주는 영향.
  - 희석: CB/BW 전환·리픽싱, 유상증자 등 공시상 잠재 물량.
- 다른 리스크 분석가의 직전 발언이 있으면 그 논점에 직접 답한다."""


def risk_transcript(history: list[tuple[str, str]]) -> str:
    if not history:
        return "## 지금까지의 리스크 토론\n- 아직 발언 없음"
    parts = ["## 지금까지의 리스크 토론"]
    for role, text in history:
        parts += [f"### {role}", data_block(text)]
    return "\n".join(parts)


def plans_and_risk(state: RunState) -> list[str]:
    """Research plan, trader plan and the risk debate so far, as prompt blocks."""
    return [
        f"## 리서치 매니저 투자 계획\n{data_block(state['investment_plan'])}",
        f"## 트레이더 실행 계획\n{data_block(state['trader_plan'])}",
        risk_transcript(state["risk_history"]),
    ]


def risk_turn(role: str, state: RunState, llm: LLM) -> dict:
    user = "\n\n".join([stage_context(state["bundle"], state["reports"]), *plans_and_risk(state)])
    system = stage_system(role, f"{STANCES[role]}\n{KR_RISKS}")
    text = llm.complete(system, user, quick=False)
    out: dict = {"risk_history": [*state["risk_history"], (role, text)]}
    if role == NEUTRAL:
        out["risk_round"] = state["risk_round"] + 1
    return out
