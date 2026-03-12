"""
Gradio web interface for Vietnamese Sentiment Analysis.
Builds the UI and wires it to the service layer.
"""

from __future__ import annotations

import gradio as gr

from app.services.sentiment_service import sentiment_service
from app.services.speech_service import speech_service


def create_gradio_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""

    with gr.Blocks(title="Phân tích cảm xúc", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎭 Hệ thống Phân tích Cảm xúc
            Ứng dụng phân tích cảm xúc từ văn bản và âm thanh sử dụng **PhoBERT** & **PhoWhisper**
            """
        )

        with gr.Tabs():
            # ── Tab 1: Text Sentiment ────────────────────────────────
            with gr.Tab("📝 Phân tích từ văn bản"):
                gr.Markdown("### Nhập văn bản để phân tích cảm xúc")
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Nhập văn bản",
                            placeholder="Nhập văn bản tiếng Việt của bạn...",
                            lines=5,
                        )
                        text_submit_btn = gr.Button(
                            "Phân tích", variant="primary"
                        )

                    with gr.Column():
                        text_output = gr.Textbox(
                            label="Kết quả dự đoán",
                            interactive=False,
                            lines=3,
                        )
                        text_prob_output = gr.Label(
                            label="Phân bố xác suất",
                            num_top_classes=2,
                        )

                gr.Examples(
                    examples=[
                        ["Sản phẩm này thật tuyệt vời!"],
                        ["Tôi rất thất vọng về dịch vụ."],
                        ["Đồ ăn ngon, không gian đẹp, nhân viên thân thiện."],
                        ["Chất lượng kém, không đáng tiền."],
                    ],
                    inputs=text_input,
                )

            # ── Tab 2: Speech-to-Text ────────────────────────────────
            with gr.Tab("🎤 Chuyển giọng nói thành văn bản"):
                gr.Markdown("### Tải lên file audio để chuyển thành văn bản")
                with gr.Row():
                    with gr.Column():
                        audio_input_stt = gr.Audio(
                            label="Tải lên file audio",
                            type="filepath",
                        )
                        stt_submit_btn = gr.Button(
                            "Chuyển đổi", variant="primary"
                        )

                    with gr.Column():
                        stt_output = gr.Textbox(
                            label="Văn bản nhận dạng",
                            interactive=False,
                            lines=5,
                        )

            # ── Tab 3: Audio Sentiment ───────────────────────────────
            with gr.Tab("🎵 Phân tích cảm xúc từ âm thanh"):
                gr.Markdown("### Tải lên file audio để phân tích cảm xúc")
                with gr.Row():
                    with gr.Column():
                        audio_input_sentiment = gr.Audio(
                            label="Tải lên file audio",
                            type="filepath",
                        )
                        audio_submit_btn = gr.Button(
                            "Phân tích", variant="primary"
                        )

                    with gr.Column():
                        audio_text_output = gr.Textbox(
                            label="Văn bản nhận dạng",
                            interactive=False,
                            lines=3,
                        )
                        audio_sentiment_output = gr.Textbox(
                            label="Kết quả phân tích cảm xúc",
                            interactive=False,
                            lines=3,
                        )
                        audio_prob_output = gr.Label(
                            label="Phân bố xác suất",
                            num_top_classes=2,
                        )

        # ── Event handlers ───────────────────────────────────────────
        text_submit_btn.click(
            fn=sentiment_service.predict_formatted,
            inputs=text_input,
            outputs=[text_output, text_prob_output],
        )

        stt_submit_btn.click(
            fn=speech_service.transcribe_for_ui,
            inputs=audio_input_stt,
            outputs=stt_output,
        )

        audio_submit_btn.click(
            fn=speech_service.analyze_audio_for_ui,
            inputs=audio_input_sentiment,
            outputs=[audio_text_output, audio_sentiment_output, audio_prob_output],
        )

        gr.Markdown(
            """
            ---
            💡 **Hướng dẫn sử dụng:**
            - **Tab 1**: Nhập văn bản trực tiếp để phân tích cảm xúc
            - **Tab 2**: Tải file audio lên để chuyển thành văn bản
            - **Tab 3**: Tải file audio lên để vừa chuyển thành văn bản vừa phân tích cảm xúc
            """
        )

    return demo
