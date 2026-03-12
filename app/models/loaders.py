"""
Model loading utilities.
Handles downloading and initialising PhoBERT and PhoWhisper models.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
)

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Return the best available device (CUDA → CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    return device


def load_sentiment_model(
    model_name: str, device: torch.device | None = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, torch.device]:
    """
    Load the PhoBERT sentiment classification model and tokenizer.

    Args:
        model_name: HuggingFace model identifier.
        device: Target device. Auto-detected if *None*.

    Returns:
        (model, tokenizer, device) tuple.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    if device is None:
        device = get_device()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        model.to(device)
        logger.info(
            "PhoBERT sentiment model loaded successfully on %s", device
        )
        return model, tokenizer, device
    except Exception as exc:
        logger.error("Failed to load sentiment model '%s': %s", model_name, exc)
        raise RuntimeError(f"Cannot load sentiment model: {exc}") from exc


def load_speech_pipeline(model_name: str):
    """
    Load the PhoWhisper ASR pipeline.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A ``transformers.pipeline`` for automatic speech recognition.

    Raises:
        RuntimeError: If the pipeline cannot be created.
    """
    try:
        device_id = 0 if torch.cuda.is_available() else -1
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device_id,
        )
        logger.info("PhoWhisper ASR pipeline loaded successfully")
        return asr_pipeline
    except Exception as exc:
        logger.error("Failed to load speech model '%s': %s", model_name, exc)
        raise RuntimeError(f"Cannot load speech model: {exc}") from exc
