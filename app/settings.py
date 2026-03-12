"""
Application settings using Pydantic BaseSettings.
Loads configuration from environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration with environment variable support."""

    # ── Model Configuration ──────────────────────────────────────────
    SENTIMENT_MODEL_NAME: str = Field(
        default="KPN14/phobert_sentiment",
        description="HuggingFace model ID for PhoBERT sentiment classification",
    )
    SPEECH_MODEL_NAME: str = Field(
        default="vinai/PhoWhisper-large",
        description="HuggingFace model ID for PhoWhisper speech recognition",
    )

    # ── Processing Configuration ─────────────────────────────────────
    MAX_TEXT_LENGTH: int = Field(
        default=256,
        description="Maximum number of tokens for text input",
    )

    # ── Label Configuration ──────────────────────────────────────────
    LABEL_0_IS_NEGATIVE: bool = Field(
        default=True,
        description="True if model label 0 = Negative, 1 = Positive",
    )
    POSITIVE_LABEL: str = "Tích cực ✅"
    NEGATIVE_LABEL: str = "Tiêu cực ❌"

    # ── Server Configuration ─────────────────────────────────────────
    SERVER_PORT: int = Field(default=7860, description="Port to run the application")
    SERVER_NAME: str = Field(
        default="0.0.0.0",
        description="Server host (0.0.0.0 for LAN access)",
    )
    SHARE_PUBLICLY: bool = Field(
        default=False,
        description="Create a public Gradio share link",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Singleton instance
settings = Settings()
