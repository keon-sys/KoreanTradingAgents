"""LangGraph flow (spec §5): six analysts in parallel -> bull/bear debate -> research manager
-> trader -> risk debate -> portfolio manager. Memory scoring stays outside the graph."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from krta.agents import debate, pm, risk, trader
from krta.agents.analysts import ANALYSTS, run_analyst
from krta.agents.schemas import Decision
from krta.collect import DataBundle
from krta.config import Settings
from krta.llm import LLM

RECURSION_MARGIN = 10


def merge(a: dict[str, str], b: dict[str, str]) -> dict[str, str]:
    """Reducer for `reports`: parallel analysts each add their own key."""
    return {**a, **b}


class RunState(TypedDict):
    bundle: DataBundle
    reports: Annotated[dict[str, str], merge]
    bull_history: list[str]
    bear_history: list[str]
    debate_round: int
    investment_plan: str
    trader_plan: str
    risk_history: list[tuple[str, str]]
    risk_round: int
    decision: Decision | None
    lessons: str


def initial_state(bundle: DataBundle, lessons: str = "") -> RunState:
    return RunState(
        bundle=bundle, reports={}, bull_history=[], bear_history=[], debate_round=0,
        investment_plan="", trader_plan="", risk_history=[], risk_round=0, decision=None,
        lessons=lessons,
    )


def build_graph(llm: LLM, settings: Settings) -> CompiledStateGraph:
    g = StateGraph(RunState)
    analyst_nodes = [f"analyst_{key}" for key in ANALYSTS]
    for key, name in zip(ANALYSTS, analyst_nodes, strict=True):
        g.add_node(name, lambda s, key=key: {"reports": {key: run_analyst(key, s["bundle"], llm)}})
        g.add_edge(START, name)

    g.add_node("bull", lambda s: debate.bull(s, llm))
    g.add_node("bear", lambda s: debate.bear(s, llm))
    g.add_node("research_manager", lambda s: debate.research_manager(s, llm))
    g.add_node("trader", lambda s: trader.trader(s, llm))
    g.add_node("aggressive", lambda s: risk.risk_turn(risk.AGGRESSIVE, s, llm))
    g.add_node("conservative", lambda s: risk.risk_turn(risk.CONSERVATIVE, s, llm))
    g.add_node("neutral", lambda s: risk.risk_turn(risk.NEUTRAL, s, llm))
    g.add_node("pm", lambda s: pm.pm(s, llm))

    g.add_edge(analyst_nodes, "bull")  # waits for all six
    g.add_edge("bull", "bear")
    g.add_conditional_edges(
        "bear",
        lambda s: "bull" if s["debate_round"] < settings.debate_rounds else "research_manager",
        ["bull", "research_manager"],
    )
    g.add_edge("research_manager", "trader")
    g.add_edge("trader", "aggressive")
    g.add_edge("aggressive", "conservative")
    g.add_edge("conservative", "neutral")
    g.add_conditional_edges(
        "neutral",
        lambda s: "aggressive" if s["risk_round"] < settings.risk_rounds else "pm",
        ["aggressive", "pm"],
    )
    g.add_edge("pm", END)

    # One superstep per node visit. An explicit cap (instead of LangGraph's default, which
    # varies by version and env) stops a routing bug before it makes thousands of LLM calls.
    steps = 1 + 2 * settings.debate_rounds + 2 + 3 * settings.risk_rounds + 1
    return g.compile().with_config(recursion_limit=steps + RECURSION_MARGIN)
