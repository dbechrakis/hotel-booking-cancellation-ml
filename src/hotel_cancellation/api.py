"""REST scoring service over the validated cancellation artifact.

Run locally with ``uvicorn hotel_cancellation.api:app``. The service loads the same
hash-verified artifact as the Streamlit app and applies the same booking validation,
risk bands and decision rules, so an API score and an app score never diverge.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hotel_cancellation import __version__
from hotel_cancellation.contracts import CATEGORICAL_FEATURES, FEATURES
from hotel_cancellation.decision import (
    expected_intervention_value,
    recommendation,
    risk_band,
)
from hotel_cancellation.model import load_model_bundle, local_contributions


ARTIFACT_DIR = Path(
    os.environ.get(
        "HOTEL_ARTIFACT_DIR",
        Path(__file__).resolve().parents[2] / "artifacts",
    )
)
MAX_BATCH_SIZE = 1000

Month = Literal[
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


class Booking(BaseModel):
    """The explicit 21-feature booking contract, known before the outcome."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "hotel": "City Hotel",
                "lead_time": 120,
                "arrival_date_month": "May",
                "arrival_date_week_number": 20,
                "arrival_date_day_of_month": 15,
                "stays_in_weekend_nights": 1,
                "stays_in_week_nights": 3,
                "adults": 2,
                "children": 0,
                "babies": 0,
                "meal": "BB",
                "country": "PRT",
                "market_segment": "Online TA",
                "distribution_channel": "TA/TO",
                "is_repeated_guest": 0,
                "previous_cancellations": 0,
                "previous_bookings_not_canceled": 0,
                "reserved_room_type": "A",
                "deposit_type": "No Deposit",
                "customer_type": "Transient",
                "adr": 110.0,
            }
        },
    )

    hotel: str
    lead_time: int = Field(ge=0, le=800)
    arrival_date_month: Month
    arrival_date_week_number: int = Field(ge=1, le=53)
    arrival_date_day_of_month: int = Field(ge=1, le=31)
    stays_in_weekend_nights: int = Field(ge=0, le=30)
    stays_in_week_nights: int = Field(ge=0, le=60)
    adults: int = Field(ge=0, le=20)
    children: int = Field(ge=0, le=10)
    babies: int = Field(ge=0, le=10)
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: Literal[0, 1]
    previous_cancellations: int = Field(ge=0, le=100)
    previous_bookings_not_canceled: int = Field(ge=0, le=100)
    reserved_room_type: str
    deposit_type: str
    customer_type: str
    adr: float = Field(ge=0, le=5000)

    @model_validator(mode="after")
    def check_stay(self) -> "Booking":
        if self.adults + self.children + self.babies <= 0:
            raise ValueError("At least one guest is required")
        if self.stays_in_weekend_nights + self.stays_in_week_nights <= 0:
            raise ValueError("At least one stay night is required")
        return self


class Economics(BaseModel):
    """User-supplied intervention assumptions; none of these are measured."""

    recoverable_margin: float = Field(ge=0, description="Margin recovered if a cancellation is prevented")
    intervention_cost: float = Field(ge=0, description="Cost of one retention contact")
    intervention_success_rate: float = Field(ge=0, le=1, description="Assumed share of contacted cancellations prevented")
    policy_threshold: float | None = Field(
        default=None, ge=0, le=1,
        description="Review threshold on the risk score; defaults to the documented 0.50",
    )


class ScoreRequest(BaseModel):
    booking: Booking
    economics: Economics | None = None
    explain: bool = Field(default=True, description="Return the largest signed logistic contributions")


class BatchRequest(BaseModel):
    bookings: list[Booking] = Field(min_length=1, max_length=MAX_BATCH_SIZE)
    economics: Economics | None = None


class Contribution(BaseModel):
    feature: str
    contribution: float
    direction: str


class Decision(BaseModel):
    policy_threshold: float
    assumed_expected_value: float
    recommendation: str


class Score(BaseModel):
    risk_score: float = Field(description="Class-weighted logistic score used for ranking and thresholds")
    cancellation_probability: float = Field(description="Isotonic-calibrated probability used for risk bands and economics")
    risk_band: str
    flagged: bool = Field(description="Risk score is at or above the model's documented 0.50 threshold")
    unseen_categories: dict[str, str] = Field(
        default_factory=dict,
        description="Categorical values absent from training; they are encoded as all-zero",
    )
    decision: Decision | None = None
    contributions: list[Contribution] | None = None


class BatchResponse(BaseModel):
    model_version: str
    scores: list[Score]


class ScoreResponse(Score):
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_model_bundle(ARTIFACT_DIR)
    yield


app = FastAPI(
    title="Hotel Cancellation Risk API",
    version=__version__,
    description=(
        "Scores hotel bookings with the chronologically validated logistic model. "
        "The risk score ranks bookings and drives the review threshold; the isotonic-calibrated "
        "probability drives risk bands and expected value. Economic outputs depend on "
        "caller-supplied assumptions and are not measured uplift."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def timing_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    return response


def model_version(metadata: dict) -> str:
    return metadata["artifact_sha256"][:12]


def unseen_categories(booking: Booking, metadata: dict) -> dict[str, str]:
    options = metadata["category_options"]
    values = booking.model_dump()
    return {
        feature: values[feature]
        for feature in CATEGORICAL_FEATURES
        if values[feature] not in options[feature]
    }


def score_frame(request: Request, bookings: list[Booking]) -> tuple[pd.DataFrame, list[tuple[float, float]]]:
    frame = pd.DataFrame([booking.model_dump() for booking in bookings])[FEATURES]
    try:
        scores, probabilities = request.app.state.bundle.score(frame)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return frame, [(float(s), float(p)) for s, p in zip(scores, probabilities)]


def build_score(
    scored: tuple[float, float],
    booking: Booking,
    metadata: dict,
    economics: Economics | None,
) -> Score:
    score, probability = scored
    threshold = float(metadata["threshold"])
    decision = None
    if economics is not None:
        policy_threshold = (
            economics.policy_threshold
            if economics.policy_threshold is not None
            else threshold
        )
        value = expected_intervention_value(
            probability,
            economics.recoverable_margin,
            economics.intervention_cost,
            economics.intervention_success_rate,
        )
        decision = Decision(
            policy_threshold=policy_threshold,
            assumed_expected_value=value,
            recommendation=recommendation(score, policy_threshold, value),
        )
    return Score(
        risk_score=score,
        cancellation_probability=probability,
        risk_band=risk_band(probability),
        flagged=score >= threshold,
        unseen_categories=unseen_categories(booking, metadata),
        decision=decision,
    )


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_version": model_version(request.app.state.bundle.metadata)}


@app.get("/model")
def model_card(request: Request) -> dict:
    """Validation evidence and limits for the served artifact."""
    metadata = request.app.state.bundle.metadata
    return {
        "model": metadata["model"],
        "model_version": model_version(metadata),
        "purpose": metadata["model_purpose"],
        "scikit_learn_version": metadata["scikit_learn_version"],
        "threshold": metadata["threshold"],
        "booking_date_cutoff": metadata["booking_date_cutoff"],
        "train_rows": metadata["train_rows"],
        "test_rows": metadata["test_rows"],
        "holdout_metrics": metadata["holdout_metrics"],
        "threshold_scale": metadata["threshold_scale"],
        "calibration": metadata["calibration"],
        "features": metadata["features"],
        "category_options": metadata["category_options"],
        "limitations": metadata["limitations"],
    }


@app.post("/predict", response_model=ScoreResponse)
def predict(payload: ScoreRequest, request: Request) -> ScoreResponse:
    metadata = request.app.state.bundle.metadata
    frame, scored = score_frame(request, [payload.booking])
    score = build_score(scored[0], payload.booking, metadata, payload.economics)
    if payload.explain:
        contributions = local_contributions(request.app.state.bundle.pipeline, frame, limit=6)
        score.contributions = [
            Contribution(**row) for row in contributions.to_dict(orient="records")
        ]
    return ScoreResponse(model_version=model_version(metadata), **score.model_dump())


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest, request: Request) -> BatchResponse:
    metadata = request.app.state.bundle.metadata
    _, scored = score_frame(request, payload.bookings)
    return BatchResponse(
        model_version=model_version(metadata),
        scores=[
            build_score(pair, booking, metadata, payload.economics)
            for pair, booking in zip(scored, payload.bookings)
        ],
    )
