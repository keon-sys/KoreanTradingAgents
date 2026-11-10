"""Portfolio manager (spec §9): the final structured decision."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from krta.agents.debate import data_block, lessons_block, stage_context, stage_system
from krta.agents.risk import plans_and_risk
from krta.agents.schemas import Decision
from krta.collect import CRITICAL_HEADER
from krta.llm import LLM

if TYPE_CHECKING:
    from krta.graph import RunState

PORTFOLIO_MANAGER = "포트폴리오 매니저"
NONE_ITEM = "- 없음"
_RECEIPT = re.compile(r"접수번호 (\d+)")

PM_TASK = """\
리서치 매니저 계획, 트레이더 계획, 리스크 토론, 지난 판단 교훈을 종합해 최종 판단을 구조화 출력으로 낸다.
- rating: BUY(매수), OVERWEIGHT(비중확대), HOLD(보유), UNDERWEIGHT(비중축소), SELL(매도) 중 하나의 영문 키.
- target_price, stop_loss: 원 단위 정수. 요청 기간에 맞춘다. 데이터로 근거를 댈 수 없으면 null.
- review_triggers: 판단을 다시 볼 가격·이벤트·공시 조건.
- price_context, bull_points, bear_points: 핵심 근거 표의 가격 상황 / 강세 근거 / 약세 근거. 리포트의 숫자만 인용한다.
- upcoming_events: 이벤트 캘린더에서 요청 기간 안에 드는 일정.
- missing_data: 각 리포트의 빠진 데이터를 종합한다.
- critical_disclosures_addressed: 치명적 공시 후보를 기간 구간과 상관없이 하나씩, 접수번호를 적어 답한다(판단에 주는 영향과 이유). 후보가 없으면 "해당 없음"이라고 쓴다. 후보 목록을 확인할 수 없으면 그 사실을 쓰고 missing_data에도 넣는다.
- summary: 최종 판단 요약. 매수 비중·손절 등은 보유 현황을 모르는 일반 제안임을 전제로 쓴다.
- 지난 판단 교훈이 있으면 같은 실수를 되풀이하지 않도록 반영한다."""


def critical_candidates(disclosure_text: str) -> list[str] | None:
    """Bullet lines under the disclosure section's critical-candidate header.

    None when the header is absent (the candidate list could not be built)."""
    lines = disclosure_text.splitlines()
    try:
        start = [line.strip() for line in lines].index(CRITICAL_HEADER) + 1
    except ValueError:
        return None
    out = []
    for line in lines[start:]:
        item = line.strip()
        if not item.startswith("- "):
            break
        if item != NONE_ITEM:
            out.append(item)
    return out


def _critical_block(candidates: list[str] | None) -> str:
    title = "## 치명적 공시 후보 (공시 섹션, 하나씩 답할 것)"
    if candidates is None:
        return f"{title}\n- 치명적 공시 후보 목록을 확인할 수 없음 (공시 섹션에 목록 없음)"
    if not candidates:
        return f"{title}\n- 치명적 공시 후보 없음 → critical_disclosures_addressed는 \"해당 없음\""
    return f"{title}\n{data_block(chr(10).join(candidates))}"


def _unaddressed(candidates: list[str], answer: str) -> list[str]:
    """Receipt numbers of candidates the answer does not mention."""
    ids = [m.group(1) for c in candidates if (m := _RECEIPT.search(c))]
    return [i for i in ids if i not in answer]


def _decide(llm: LLM, system: str, user: str) -> Decision:
    """Structured decision; one retry when the model gives no structured output."""
    for _ in range(2):
        decision = llm.structured(system, user, Decision)
        if decision is not None:
            return decision  # type: ignore[return-value]
    raise RuntimeError("PM 구조화 출력 없음")


def pm(state: RunState, llm: LLM) -> dict:
    b = state["bundle"]
    candidates = critical_candidates(b.sections["disclosure"].text)
    user = "\n\n".join([
        stage_context(b, state["reports"]),
        *plans_and_risk(state),
        lessons_block(state["lessons"]),
        _critical_block(candidates),
    ])
    system = stage_system(PORTFOLIO_MANAGER, PM_TASK)
    decision = _decide(llm, system, user)
    missing = _unaddressed(candidates or [], decision.critical_disclosures_addressed)
    if missing:
        note = (
            "\n\n## 누락된 치명적 공시 후보\n직전 답변의 critical_disclosures_addressed가 "
            f"다음 접수번호를 다루지 않았다: {', '.join(missing)}. 후보마다 접수번호를 적고 하나씩 답할 것."
        )
        decision = _decide(llm, system + note, user)
        missing = _unaddressed(candidates or [], decision.critical_disclosures_addressed)
        if missing:
            patched = (
                f"{decision.critical_disclosures_addressed.rstrip()}\n"
                f"(PM이 다루지 않음: 접수번호 {', '.join(missing)})"
            )
            decision = decision.model_copy(update={"critical_disclosures_addressed": patched})
    return {"decision": decision}
