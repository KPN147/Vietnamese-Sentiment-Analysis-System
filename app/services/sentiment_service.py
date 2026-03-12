"""
Sentiment analysis service.
Encapsulates PhoBERT inference logic, independent of any UI framework.
"""

from __future__ import annotations

import logging
from typing import Tuple, Dict

import torch

from app.models.registry import registry
from app.settings import settings

logger = logging.getLogger(__name__)


class SentimentService:
    """Stateless service — all state lives in :pyclass:`ModelRegistry`."""

    # ── Public API ───────────────────────────────────────────────────

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Run sentiment classification on *text*.

        Returns:
            (label, confidence_pct, probabilities) where *probabilities*
            maps display labels to their softmax scores.
        """
        self._validate(text)
        model, tokenizer, device = registry.get_sentiment_model()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.MAX_TEXT_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        probs_np = probs.cpu().numpy()

        label = self._index_to_label(pred_idx)
        confidence = float(probs_np[pred_idx]) * 100
        prob_dict = {
            "Tiêu cực 😞": float(probs_np[0]),
            "Tích cực 😊": float(probs_np[1]),
        }

        return label, confidence, prob_dict

    def predict_formatted(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Convenience wrapper used by the Gradio UI.
        Returns (markdown_result, prob_dict).
        """
        if not text or text.strip() == "":
            return "Vui lòng nhập văn bản!", {}

        try:
            label, confidence, prob_dict = self.predict(text)
            result_text = f"**Kết quả:** {label}\n**Độ tin cậy:** {confidence:.2f}%"
            return result_text, prob_dict
        except Exception as exc:
            logger.exception("Sentiment prediction failed")
            return f"❌ Lỗi khi dự đoán: {exc}", {}

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _validate(text: str) -> None:
        if not text or not text.strip():
            raise ValueError("Input text must not be empty")

    @staticmethod
    def _index_to_label(index: int) -> str:
        if settings.LABEL_0_IS_NEGATIVE:
            return settings.POSITIVE_LABEL if index == 1 else settings.NEGATIVE_LABEL
        return settings.POSITIVE_LABEL if index == 0 else settings.NEGATIVE_LABEL


# Module-level convenience instance
sentiment_service = SentimentService()
