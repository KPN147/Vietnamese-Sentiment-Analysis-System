"""
Pydantic schemas for request/response validation.
Used by FastAPI endpoints and service layer.
"""

from pydantic import BaseModel, Field


# ── Sentiment Schemas ────────────────────────────────────────────────

class SentimentRequest(BaseModel):
    """Request body for sentiment prediction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Vietnamese text to analyze",
        examples=["Sản phẩm này thật tuyệt vời!"],
    )


class SentimentResponse(BaseModel):
    """Response body for sentiment prediction."""

    text: str = Field(description="Original input text")
    label: str = Field(description="Predicted sentiment label")
    confidence: float = Field(
        ge=0, le=100, description="Confidence score in percentage"
    )
    probabilities: dict[str, float] = Field(
        description="Probability distribution over all classes"
    )


# ── Speech-to-Text Schemas ───────────────────────────────────────────

class SpeechToTextResponse(BaseModel):
    """Response body for speech-to-text transcription."""

    transcription: str = Field(description="Transcribed text from audio")


class AudioSentimentResponse(BaseModel):
    """Response body for combined audio → text → sentiment pipeline."""

    transcription: str = Field(description="Transcribed text from audio")
    label: str = Field(description="Predicted sentiment label")
    confidence: float = Field(ge=0, le=100)
    probabilities: dict[str, float]


# ── Health Check ─────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for health check endpoint."""

    status: str = "ok"
    sentiment_model_loaded: bool = False
    speech_model_loaded: bool = False
