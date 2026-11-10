import json
from dataclasses import replace
from datetime import date
from pathlib import Path

from krta.agents.analysts import ANALYSTS
from krta.agents.schemas import Decision, Rating
from krta.collect import UNCONFIRMED_BANNER, DataBundle, Target
from krta.graph import initial_state
from krta.inputs import parse_horizon
from krta.report import (
    DISCLAIMER,
    PAST_DATE_NOTE,
    REPORT_FILES,
    render_final_decision,
    run_dir,
    write_reports,
)
from krta.sections import Section

DAY = date(2026, 11, 10)
CALENDAR = "| 날짜 | 이벤트 | 비고 |\n|---|---|---|\n| 2026-11-12 | 선물옵션 만기 |  |\n"


def bundle(**over) -> DataBundle:
    b = DataBundle(
        target=Target("005930", "KS", "삼성전자", "00126380", date(1975, 6, 11)),
        date=DAY,
        window_end=date(2027, 2, 12),
        horizon=parse_horizon("3m"),
        unconfirmed=False,
        date_note=None,
        peers=[],
        calendar_md=CALENDAR,
        sections={key: Section(key, f"{key} 섹션\n") for key in ANALYSTS},
    )
    return replace(b, **over)


def decision(**over) -> Decision:
    d = dict(
        rating=Rating.OVERWEIGHT,
        target_price=80000,
        stop_loss=68000,
        review_triggers=["종가 68,000원 이탈", "3분기 실적 발표"],
        price_context="20일선 위, 종가 72,300원",
        bull_points=["HBM 출하 증가", "외국인 순매수 | 5일 연속"],
        bear_points=["희석 우려"],
        upcoming_events=["2026-11-12 선물옵션 만기"],
        missing_data=["투자자별 매매"],
        critical_disclosures_addressed="20261103000123: 규모 작음",
        summary="분할 매수",
    )
    d.update(over)
    return Decision(**d)


def state(b: DataBundle | None = None, d: Decision | None = None):
    s = initial_state(b or bundle())
    s["reports"] = {key: f"# {key} 리포트\n본문" for key in ANALYSTS}
    s["investment_plan"] = "# 투자 계획\n비중확대"
    s["trader_plan"] = "# 트레이더 계획\n분할 진입"
    s["decision"] = d or decision()
    return s


def test_run_dir_layout(tmp_path):
    assert run_dir(tmp_path, bundle()) == tmp_path / "logs-kr" / "005930.KS" / "2026-11-10" / "3m"


def test_writes_all_report_files(tmp_path):
    out = run_dir(tmp_path, bundle())
    written = write_reports(out, state(), default_date=DAY)

    names = sorted(p.name for p in (out / "reports").iterdir())
    assert names == sorted(REPORT_FILES)
    assert len(REPORT_FILES) == 10
    assert (out / "decision.json").is_file()
    assert set(written) == {out / "reports" / n for n in REPORT_FILES} | {out / "decision.json"}

    reports = out / "reports"
    assert (reports / "event_calendar.md").read_text(encoding="utf-8") == CALENDAR
    for key, spec in ANALYSTS.items():
        assert (reports / spec.report_file).read_text(encoding="utf-8") == f"# {key} 리포트\n본문"
    assert (reports / "investment_plan.md").read_text(encoding="utf-8") == "# 투자 계획\n비중확대"
    assert (reports / "trader_investment_plan.md").read_text(encoding="utf-8") == "# 트레이더 계획\n분할 진입"
    assert not list(out.rglob(".tmp-*"))

    data = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    assert data["code"] == "005930"
    assert data["market"] == "KS"
    assert data["date"] == "2026-11-10"
    assert data["horizon"] == "3m"
    assert data["trading_days"] == 63
    assert data["rating"] == "OVERWEIGHT"
    assert data["rating_ko"] == "비중확대"
    assert data["target_price"] == 80000
    assert data["unconfirmed"] is False


def test_write_reports_without_decision_fails(tmp_path):
    s = state()
    s["decision"] = None
    try:
        write_reports(tmp_path, s, default_date=DAY)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    assert not (tmp_path / "decision.json").exists()


def test_final_decision_contains_fixed_disclaimer_and_rating_ko():
    md = render_final_decision(bundle(), decision(), default_date=DAY)

    assert DISCLAIMER == "매수 비중·손절가 등은 보유 현황을 모르고 쓴 일반 제안입니다."
    assert md.rstrip().endswith(DISCLAIMER)
    assert "비중확대 (OVERWEIGHT)" in md
    assert "80,000원" in md and "68,000원" in md
    assert "3m (63거래일, 중기)" in md
    assert "종가 68,000원 이탈" in md
    # section order
    order = ["## 결론", "## 핵심 근거", "## 다가오는 이벤트", "## 빠진 데이터", "## 치명적 공시 대응"]
    pos = [md.index(h) for h in order]
    assert pos == sorted(pos)
    # evidence table: pipes in cells are escaped, one table row per category
    assert "| 가격 상황 |" in md and "| 강세 근거 |" in md and "| 약세 근거 |" in md
    assert "외국인 순매수 \\| 5일 연속" in md
    assert "2026-11-12 선물옵션 만기" in md
    assert "20261103000123: 규모 작음" in md
    assert UNCONFIRMED_BANNER not in md
    assert PAST_DATE_NOTE not in md


def test_final_decision_null_prices_and_empty_lists():
    md = render_final_decision(
        bundle(), decision(target_price=None, stop_loss=None, upcoming_events=[], missing_data=[]),
        default_date=DAY,
    )
    assert "목표가: 근거 없음" in md
    assert "손절가: 근거 없음" in md
    events = md[md.index("## 다가오는 이벤트"):md.index("## 빠진 데이터")]
    assert "- 없음" in events


def test_final_decision_unconfirmed_warning_on_top_and_notes_under_conclusion():
    b = bundle(unconfirmed=True, date_note="요청일 2026-11-08은 휴장일이라 직전 거래일 2026-11-06로 분석",
               date=date(2026, 11, 6))
    md = render_final_decision(b, decision(), default_date=DAY)

    first = md.splitlines()[0]
    assert UNCONFIRMED_BANNER in first
    conclusion = md[md.index("## 결론"):md.index("## 핵심 근거")]
    assert "휴장일이라 직전 거래일" in conclusion
    assert PAST_DATE_NOTE in conclusion
    assert PAST_DATE_NOTE == "과거 날짜 분석: 재무제표 정정 전 원본·당시 상장 목록은 재현하지 않음"


def test_past_date_note_only_when_before_default_date():
    assert PAST_DATE_NOTE in render_final_decision(bundle(), decision(), default_date=date(2026, 11, 11))
    assert PAST_DATE_NOTE not in render_final_decision(bundle(), decision(), default_date=DAY)


def test_reports_overwrite_on_rerun(tmp_path):
    write_reports(tmp_path, state(d=decision(summary="첫 실행")), default_date=DAY)
    write_reports(tmp_path, state(d=decision(rating=Rating.SELL)), default_date=DAY)
    data = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert data["rating"] == "SELL"
    assert "매도 (SELL)" in (tmp_path / "reports" / "final_trade_decision.md").read_text(encoding="utf-8")


def test_report_dir_is_created(tmp_path: Path):
    out = tmp_path / "a" / "b"
    write_reports(out, state(), default_date=DAY)
    assert (out / "reports" / "final_trade_decision.md").is_file()
