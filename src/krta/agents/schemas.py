"""Structured final decision of the portfolio manager (spec §9)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Rating(str, Enum):
    BUY = "BUY"
    OVERWEIGHT = "OVERWEIGHT"
    HOLD = "HOLD"
    UNDERWEIGHT = "UNDERWEIGHT"
    SELL = "SELL"


RATING_KO: dict[Rating, str] = {
    Rating.BUY: "매수",
    Rating.OVERWEIGHT: "비중확대",
    Rating.HOLD: "보유",
    Rating.UNDERWEIGHT: "비중축소",
    Rating.SELL: "매도",
}


class Decision(BaseModel):
    """PM output. Horizon and the unconfirmed flag are attached by code, not by the model."""

    rating: Rating = Field(description="5단계 판단 영문 키: BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL")
    target_price: int | None = Field(description="요청 기간 기준 목표가(원). 근거가 없으면 null")
    stop_loss: int | None = Field(description="손절가(원). 근거가 없으면 null")
    review_triggers: list[str] = Field(description="판단을 다시 볼 조건(가격·이벤트·공시)")
    price_context: str = Field(description="현재 가격 상황(데이터에 있는 숫자만 인용)")
    bull_points: list[str] = Field(description="강세 근거")
    bear_points: list[str] = Field(description="약세 근거")
    upcoming_events: list[str] = Field(description="이벤트 캘린더에서 요청 기간 안에 드는 일정")
    missing_data: list[str] = Field(description="리포트들의 빠진 데이터 종합")
    critical_disclosures_addressed: str = Field(
        description="치명적 공시 후보 각각(접수번호 포함)에 대한 판단. 후보가 없으면 '해당 없음'"
    )
    summary: str = Field(description="최종 판단 요약")
