"""
Model registry – singleton that manages model lifecycle.
Ensures models are loaded only once and provides easy access.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.models.loaders import load_sentiment_model, load_speech_pipeline
from app.settings import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Centralised registry for all ML models used by the application.

    Usage::

        registry = ModelRegistry()
        registry.load_all()          # pre-load at startup
        model, tok, dev = registry.get_sentiment_model()
        asr = registry.get_speech_pipeline()
    """

    _instance: Optional["ModelRegistry"] = None

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        self._sentiment_model = None
        self._sentiment_tokenizer = None
        self._device = None
        self._speech_pipeline = None
        self._initialised = True

    # ── Loading ──────────────────────────────────────────────────────

    def load_sentiment(self) -> None:
        """Load the sentiment model (idempotent)."""
        if self._sentiment_model is not None:
            return
        logger.info("Loading sentiment model: %s", settings.SENTIMENT_MODEL_NAME)
        self._sentiment_model, self._sentiment_tokenizer, self._device = (
            load_sentiment_model(settings.SENTIMENT_MODEL_NAME)
        )

    def load_speech(self) -> None:
        """Load the speech pipeline (idempotent)."""
        if self._speech_pipeline is not None:
            return
        logger.info("Loading speech model: %s", settings.SPEECH_MODEL_NAME)
        self._speech_pipeline = load_speech_pipeline(settings.SPEECH_MODEL_NAME)

    def load_all(self) -> None:
        """Convenience: load every model the app needs."""
        self.load_sentiment()
        self.load_speech()

    # ── Accessors ────────────────────────────────────────────────────

    def get_sentiment_model(self):
        """Return *(model, tokenizer, device)*.  Loads on first call."""
        self.load_sentiment()
        return self._sentiment_model, self._sentiment_tokenizer, self._device

    def get_speech_pipeline(self):
        """Return the ASR pipeline.  Loads on first call."""
        self.load_speech()
        return self._speech_pipeline

    # ── Status ───────────────────────────────────────────────────────

    @property
    def sentiment_ready(self) -> bool:
        return self._sentiment_model is not None

    @property
    def speech_ready(self) -> bool:
        return self._speech_pipeline is not None


# Module-level singleton for convenience imports
registry = ModelRegistry()
