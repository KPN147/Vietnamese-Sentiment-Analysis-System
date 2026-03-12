"""
Unit tests for the sentiment analysis service.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestSentimentService:
    """Tests for SentimentService."""

    def test_predict_formatted_empty_text_returns_warning(self):
        """Empty input should return a user-friendly warning, not crash."""
        from app.services.sentiment_service import SentimentService

        service = SentimentService()
        result_text, prob_dict = service.predict_formatted("")
        assert "Vui lòng nhập văn bản" in result_text
        assert prob_dict == {}

    def test_predict_formatted_none_text_returns_warning(self):
        """None input should return a user-friendly warning."""
        from app.services.sentiment_service import SentimentService

        service = SentimentService()
        result_text, prob_dict = service.predict_formatted(None)
        assert "Vui lòng nhập văn bản" in result_text
        assert prob_dict == {}

    def test_predict_formatted_whitespace_returns_warning(self):
        """Whitespace-only input should return a user-friendly warning."""
        from app.services.sentiment_service import SentimentService

        service = SentimentService()
        result_text, prob_dict = service.predict_formatted("   ")
        assert "Vui lòng nhập văn bản" in result_text
        assert prob_dict == {}


class TestSpeechService:
    """Tests for SpeechService."""

    def test_transcribe_for_ui_none_input(self):
        """None audio path should return a user-friendly message."""
        from app.services.speech_service import SpeechService

        service = SpeechService()
        result = service.transcribe_for_ui(None)
        assert "Vui lòng tải lên file audio" in result

    def test_analyze_audio_for_ui_none_input(self):
        """None audio path should return a user-friendly tuple."""
        from app.services.speech_service import SpeechService

        service = SpeechService()
        text, sentiment, probs = service.analyze_audio_for_ui(None)
        assert "Vui lòng tải lên file audio" in text
        assert sentiment == ""
        assert probs == {}


class TestSettings:
    """Tests for application settings."""

    def test_default_settings_load(self):
        """Settings should load with sensible defaults."""
        from app.settings import Settings

        s = Settings()
        assert s.SENTIMENT_MODEL_NAME == "KPN14/phobert_sentiment"
        assert s.SPEECH_MODEL_NAME == "vinai/PhoWhisper-large"
        assert s.MAX_TEXT_LENGTH == 256
        assert s.SERVER_PORT == 7860

    def test_label_configuration(self):
        """Label config should default to 0=Negative, 1=Positive."""
        from app.settings import Settings

        s = Settings()
        assert s.LABEL_0_IS_NEGATIVE is True
        assert "Tích cực" in s.POSITIVE_LABEL
        assert "Tiêu cực" in s.NEGATIVE_LABEL


class TestSchemas:
    """Tests for Pydantic schemas."""

    def test_sentiment_request_valid(self):
        """Valid text should create a SentimentRequest."""
        from app.schemas import SentimentRequest

        req = SentimentRequest(text="Xin chào")
        assert req.text == "Xin chào"

    def test_sentiment_request_empty_raises(self):
        """Empty text should raise a validation error."""
        from app.schemas import SentimentRequest

        with pytest.raises(Exception):
            SentimentRequest(text="")

    def test_health_response_defaults(self):
        """HealthResponse should have sensible defaults."""
        from app.schemas import HealthResponse

        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.sentiment_model_loaded is False
        assert resp.speech_model_loaded is False
