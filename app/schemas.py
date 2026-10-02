"""Request/response models with input validation."""
from pydantic import BaseModel, Field, field_validator

MAX_TEXT_LEN = 2000


def _clean(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("text must not be blank")
    if len(value) > MAX_TEXT_LEN:
        raise ValueError(f"text must be at most {MAX_TEXT_LEN} characters")
    return value


class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        return _clean(v)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        return [_clean(t) for t in v]


class Prediction(BaseModel):
    label: str
    confidence: float


class PredictResponse(Prediction):
    latency_ms: float


class BatchPredictResponse(BaseModel):
    predictions: list[Prediction]
    latency_ms: float


class HistoryItem(BaseModel):
    id: int
    text: str
    label: str
    confidence: float
    endpoint: str
    created_at: str
