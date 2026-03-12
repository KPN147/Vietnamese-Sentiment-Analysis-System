"""
Application entry point.
Provides a FastAPI server with:
  - Gradio UI mounted at /
  - REST API endpoints at /api/v1/
  - Health check at /health
"""

from __future__ import annotations

import logging

import gradio as gr
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from app.settings import settings
from app.schemas import SentimentRequest, SentimentResponse, HealthResponse
from app.models.registry import registry
from app.services.sentiment_service import sentiment_service
from app.ui.gradio_app import create_gradio_app

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(
    title="Vietnamese Sentiment Analysis API",
    description="Sentiment analysis for Vietnamese text and audio using PhoBERT & PhoWhisper",
    version="1.0.0",
)


# ── Startup event ────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Pre-load models when the server starts."""
    logger.info("🚀 Starting Vietnamese Sentiment Analysis System…")
    try:
        registry.load_all()
        logger.info("✅ All models loaded successfully")
    except Exception as exc:
        logger.error("⚠️ Model loading failed: %s", exc)
        logger.info("The app will attempt lazy loading when first request arrives.")


# ── REST API endpoints ───────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Return system health and model readiness status."""
    return HealthResponse(
        status="ok",
        sentiment_model_loaded=registry.sentiment_ready,
        speech_model_loaded=registry.speech_ready,
    )


@app.post("/api/v1/predict", response_model=SentimentResponse, tags=["sentiment"])
async def predict_sentiment(request: SentimentRequest):
    """Classify sentiment of Vietnamese text."""
    try:
        label, confidence, prob_dict = sentiment_service.predict(request.text)
        return SentimentResponse(
            text=request.text,
            label=label,
            confidence=confidence,
            probabilities=prob_dict,
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Prediction error: {exc}"},
        )


# ── Mount Gradio UI ──────────────────────────────────────────────────
gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/")


# ── CLI entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Starting server on %s:%s", settings.SERVER_NAME, settings.SERVER_PORT
    )
    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_NAME,
        port=settings.SERVER_PORT,
        reload=False,
    )
