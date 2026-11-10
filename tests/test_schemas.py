import pytest
from pydantic import ValidationError

from krta.agents.schemas import RATING_KO, Decision, Rating

VALID = {
    "rating": "HOLD",
    "target_price": 80000,
    "stop_loss": 70000,
    "review_triggers": ["종가 70,000원 이탈"],
    "price_context": "20일선 위",
    "bull_points": ["외국인 순매수"],
    "bear_points": ["환율 부담"],
    "upcoming_events": ["2026-11-12 선물옵션 만기"],
    "missing_data": ["공매도 잔고"],
    "critical_disclosures_addressed": "해당 없음",
    "summary": "보유 유지",
}


def test_rating_keys_and_korean_labels():
    assert [r.value for r in Rating] == ["BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"]
    assert RATING_KO == {
        Rating.BUY: "매수",
        Rating.OVERWEIGHT: "비중확대",
        Rating.HOLD: "보유",
        Rating.UNDERWEIGHT: "비중축소",
        Rating.SELL: "매도",
    }


def test_decision_schema_accepts_valid_and_null_prices():
    d = Decision.model_validate({**VALID, "target_price": None, "stop_loss": None})
    assert d.rating is Rating.HOLD and d.target_price is None


def test_decision_schema_rejects_bad_rating():
    with pytest.raises(ValidationError):
        Decision.model_validate({**VALID, "rating": "STRONG_BUY"})


def test_decision_schema_requires_every_field():
    for key in VALID:
        partial = {k: v for k, v in VALID.items() if k != key}
        with pytest.raises(ValidationError):
            Decision.model_validate(partial)
