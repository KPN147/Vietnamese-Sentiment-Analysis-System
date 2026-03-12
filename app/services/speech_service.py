"""
Speech-to-text service.
Encapsulates PhoWhisper inference logic, independent of any UI framework.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

from app.models.registry import registry
from app.services.sentiment_service import sentiment_service

logger = logging.getLogger(__name__)


class SpeechService:
    """Handles audio → text and audio → text → sentiment pipelines."""

    def transcribe(self, audio_path: str) -> str:
        """
        Convert an audio file to Vietnamese text using PhoWhisper.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Transcribed text.

        Raises:
            ValueError: If *audio_path* is ``None`` or empty.
            RuntimeError: If transcription fails.
        """
        if not audio_path:
            raise ValueError("Audio path must not be empty")

        asr = registry.get_speech_pipeline()

        try:
            result = asr(audio_path)
            text = result.get("text", "").strip()
            if not text:
                return "⚠️ Không nhận dạng được văn bản từ audio. Vui lòng thử file khác."
            return text
        except Exception as exc:
            logger.exception("Speech-to-text failed for %s", audio_path)
            raise RuntimeError(f"Transcription error: {exc}") from exc

    # ── Gradio-friendly wrappers ─────────────────────────────────────

    def transcribe_for_ui(self, audio_path: str | None) -> str:
        """Wrapper that returns user-friendly messages instead of raising."""
        if audio_path is None:
            return "Vui lòng tải lên file audio!"
        try:
            return self.transcribe(audio_path)
        except Exception as exc:
            return f"❌ Lỗi khi chuyển đổi audio: {exc}"

    def analyze_audio_for_ui(
        self, audio_path: str | None
    ) -> Tuple[str, str, Dict[str, float]]:
        """
        Combined STT + Sentiment pipeline for the Gradio UI.

        Returns:
            (transcription, sentiment_markdown, prob_dict)
        """
        if audio_path is None:
            return "Vui lòng tải lên file audio!", "", {}

        text = self.transcribe_for_ui(audio_path)
        if text.startswith(("❌", "⚠️")):
            return text, "", {}

        sentiment_result, prob_dict = sentiment_service.predict_formatted(text)
        return text, sentiment_result, prob_dict


# Module-level convenience instance
speech_service = SpeechService()
