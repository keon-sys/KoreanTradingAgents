"""Trader (spec §9): entry, scaling and stop plan -> `trader_investment_plan.md`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from krta.agents.debate import data_block, lessons_block, stage_context, stage_system
from krta.llm import LLM

if TYPE_CHECKING:
    from krta.graph import RunState

TRADER = "트레이더"

TRADER_TASK = """\
리서치 매니저의 투자 계획을 실행 계획(`trader_investment_plan.md`)으로 바꾼다.
- 진입 방식(즉시/분할/조건부), 분할 횟수와 가격대, 손절가, 목표가, 재점검 시점을 적는다. 목표가·손절가·재점검 시점은 요청 기간에 맞춘다.
- 조건부 진입·증액 조건("OO선 회복 시", "지지 확인 후" 등)을 쓸 때는 조건마다 아래 차트 섹션의 최근 일봉(OHLC)으로 분석일 기준 판정을 붙인다: "이미 충족" 또는 "미충족", 그리고 근거가 된 날짜와 가격. 이미 충족된 조건을 앞으로의 조건처럼 쓰지 말 것. 일봉 데이터가 없으면 "판정 불가(데이터 없음)"라고 쓴다.
- 지난 판단 교훈이 있으면 손절 폭, 목표가 설정 등에 반영한다.
- 매수 비중·손절가는 사용자의 보유 현황을 모르고 쓰는 일반 제안이라고 밝힌다."""


def trader(state: RunState, llm: LLM) -> dict:
    b = state["bundle"]
    user = "\n\n".join([
        stage_context(b, state["reports"]),
        f"## 리서치 매니저 투자 계획\n{data_block(state['investment_plan'])}",
        "## 차트·거래량 섹션 원자료 (섹션: technical, 조건 판정용)\n"
        f"{data_block(b.sections['technical'].text)}",
        lessons_block(state["lessons"]),
    ])
    text = llm.complete(stage_system(TRADER, TRADER_TASK), user, quick=False)
    return {"trader_plan": text}
