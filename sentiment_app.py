import gradio as gr
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self, sentiment_model_name , speech_model_name):
        print(" Đang tải models...")

        try:
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
            self.sentiment_model.eval()

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.sentiment_model.to(self.device)
            print(f" PhoBERT sentiment model đã tải thành công! (Device: {self.device})")
        except Exception as e:
            print(f" Lỗi khi tải PhoBERT model: {e}")
            self.sentiment_model = None
            self.sentiment_tokenizer = None

        try:
            self.speech_pipeline = pipeline(
                "automatic-speech-recognition",
                model=speech_model_name,
                device=0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
            )
            print(f" PhoWhisper model đã tải thành công!")
        except Exception as e:
            print(f" Lỗi khi tải PhoWhisper model: {e}")
            self.speech_pipeline = None

    def predict_sentiment_from_text(self, text):
        if not text or text.strip() == "":
            return "Vui lòng nhập văn bản!", {}

        if self.sentiment_model is None or self.sentiment_tokenizer is None:
            return " Model chưa được tải!", {}

        try:

            inputs = self.sentiment_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)

            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            prediction = torch.argmax(outputs.logits, dim=1).item()

            probabilities = probabilities.cpu().numpy()

            sentiment = "Tích cực ✅" if prediction == 1 else "Tiêu cực ❌"
            confidence = float(probabilities[prediction]) * 100

            result_text = f"**Kết quả:** {sentiment}\n**Độ tin cậy:** {confidence:.2f}%"
            prob_dict = {
                "Tiêu cực 😞": float(probabilities[0]),
                "Tích cực 😊": float(probabilities[1])
            }

            return result_text, prob_dict

        except Exception as e:
            return f" Lỗi khi dự đoán: {str(e)}", {}

    def speech_to_text(self, audio_path):
        if audio_path is None:
            return "Vui lòng tải lên file audio!"

        if self.speech_pipeline is None:
            return " PhoWhisper model chưa được tải!"

        try:
            result = self.speech_pipeline(audio_path)
            text = result["text"]

            if not text or text.strip() == "":
                return "⚠️ Không nhận dạng được văn bản từ audio. Vui lòng thử file khác."

            return text

        except Exception as e:
            return f"❌ Lỗi khi chuyển đổi audio: {str(e)}"

    def predict_sentiment_from_audio(self, audio_path):

        if audio_path is None:
            return "Vui lòng tải lên file audio!", "", {}


        text = self.speech_to_text(audio_path)


        sentiment, prob_dict = self.predict_sentiment_from_text(text)

        return text, sentiment, prob_dict


SENTIMENT_MODEL_NAME =  "KPN14/phobert_sentiment"
SPEECH_MODEL_NAME = "vinai/PhoWhisper-large"


print(" Đang khởi động ứng dụng...")
analyzer = SentimentAnalyzer(
    sentiment_model_name=SENTIMENT_MODEL_NAME,
    speech_model_name=SPEECH_MODEL_NAME
)


with gr.Blocks(title="Phân tích cảm xúc", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎭 Hệ thống Phân tích Cảm xúc
        Ứng dụng phân tích cảm xúc từ văn bản và âm thanh
        """
    )

    with gr.Tabs():

        with gr.Tab("📝 Phân tích từ văn bản"):
            gr.Markdown("### Nhập văn bản để phân tích cảm xúc")
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Nhập văn bản",
                        placeholder="Nhập văn bản tiếng Việt của bạn...",
                        lines=5
                    )
                    text_submit_btn = gr.Button("Phân tích", variant="primary")

                with gr.Column():
                    text_output = gr.Textbox(
                        label="Kết quả dự đoán",
                        interactive=False,
                        lines=3
                    )
                    text_prob_output = gr.Label(
                        label="Phân bố xác suất",
                        num_top_classes=2
                    )

            gr.Examples(
                examples=[
                    ["Sản phẩm này thật tuyệt vời!"],
                    ["Tôi rất thất vọng về dịch vụ."],
                    ["Đồ ăn ngon, không gian đẹp, nhân viên thân thiện."],
                    ["Chất lượng kém, không đáng tiền."]
                ],
                inputs=text_input
            )


        with gr.Tab("🎤 Chuyển giọng nói thành văn bản"):
            gr.Markdown("### Tải lên file audio để chuyển thành văn bản")
            with gr.Row():
                with gr.Column():
                    audio_input_stt = gr.Audio(
                        label="Tải lên file audio",
                        type="filepath"
                    )
                    stt_submit_btn = gr.Button("Chuyển đổi", variant="primary")

                with gr.Column():
                    stt_output = gr.Textbox(
                        label="Văn bản nhận dạng",
                        interactive=False,
                        lines=5
                    )


        with gr.Tab("🎵 Phân tích cảm xúc từ âm thanh"):
            gr.Markdown("### Tải lên file audio để phân tích cảm xúc")
            with gr.Row():
                with gr.Column():
                    audio_input_sentiment = gr.Audio(
                        label="Tải lên file audio",
                        type="filepath"
                    )
                    audio_submit_btn = gr.Button("Phân tích", variant="primary")

                with gr.Column():
                    audio_text_output = gr.Textbox(
                        label="Văn bản nhận dạng",
                        interactive=False,
                        lines=3
                    )
                    audio_sentiment_output = gr.Textbox(
                        label="Kết quả phân tích cảm xúc",
                        interactive=False,
                        lines=3
                    )
                    audio_prob_output = gr.Label(
                        label="Phân bố xác suất",
                        num_top_classes=2
                    )


    text_submit_btn.click(
        fn=analyzer.predict_sentiment_from_text,
        inputs=text_input,
        outputs=[text_output, text_prob_output]
    )

    stt_submit_btn.click(
        fn=analyzer.speech_to_text,
        inputs=audio_input_stt,
        outputs=stt_output
    )

    audio_submit_btn.click(
        fn=analyzer.predict_sentiment_from_audio,
        inputs=audio_input_sentiment,
        outputs=[audio_text_output, audio_sentiment_output, audio_prob_output]
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


if __name__ == "__main__":
    demo.launch(
        share= False,
        server_name="0.0.0.0",
        server_port=7860
    )
