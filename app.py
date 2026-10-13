"""
Web Application Interface for Wav2Vec 2.0 ASR Tool using Gradio.
Provides browser-based file upload and live microphone transcription.
"""

import os
import sys
import tempfile
import gradio as gr
from asr_engine import Wav2Vec2ASR

# Initialize engine globally for Web UI
print("[*] Initializing Wav2Vec 2.0 ASR Engine for Web Application...")
asr_engine = Wav2Vec2ASR()

def process_audio(audio_path_or_tuple):
    """
    Callback function to handle audio input from Gradio (file upload or mic).
    """
    if audio_path_or_tuple is None:
        return (
            "⚠️ Please upload an audio file or record speech via microphone.",
            "0.0s",
            "0.0s",
            asr_engine.device,
            None
        )

    try:
        result = asr_engine.transcribe(audio_path_or_tuple)
        text = result["text"]
        duration = f"{result['duration_sec']}s"
        latency = f"{result['inference_time_sec']}s"
        device = result["device"].upper()

        if not text:
            display_text = "*(No intelligible speech detected)*"
        else:
            display_text = text

        # Save temporary transcript file for download
        temp_dir = tempfile.gettempdir()
        download_file = os.path.join(temp_dir, "transcription.txt")
        with open(download_file, "w", encoding="utf-8") as f:
            f.write(text + "\n")

        return display_text, duration, latency, device, download_file
    except Exception as e:
        error_msg = f"❌ Error during transcription: {str(e)}"
        return error_msg, "N/A", "N/A", asr_engine.device, None

# Define custom CSS for a modern, sleek aesthetic
custom_css = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.header-box {
    text-align: center;
    padding: 24px 16px;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 12px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
.header-box h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-box p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
}
.stat-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
"""

def create_ui():
    with gr.Blocks(css=custom_css, title="Wav2Vec 2.0 ASR Tool") as demo:
        with gr.Column():
            gr.HTML(
                """
                <div class="header-box">
                    <h1>🎙️ Wav2Vec 2.0 Speech Recognition Tool</h1>
                    <p>State-of-the-Art Automatic Speech Recognition powered by Hugging Face Transformers & PyTorch</p>
                </div>
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.TabItem("📁 File Upload"):
                            file_audio = gr.Audio(
                                sources=["upload"],
                                type="filepath",
                                label="Upload Audio (WAV, MP3, M4A, FLAC, OGG)"
                            )
                            file_submit_btn = gr.Button("🚀 Transcribe Audio File", variant="primary")

                        with gr.TabItem("🎙️ Microphone"):
                            mic_audio = gr.Audio(
                                sources=["microphone"],
                                type="filepath",
                                label="Record from Microphone"
                            )
                            mic_submit_btn = gr.Button("🚀 Transcribe Voice Recording", variant="primary")

                    gr.Markdown("### ℹ️ Instructions")
                    gr.Markdown(
                        "- **Upload:** Select any common speech audio file.\n"
                        "- **Microphone:** Click the record button, speak clearly, then click Transcribe.\n"
                        "- Audio is automatically resampled to 16kHz mono and processed through Wav2Vec 2.0 CTC."
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Transcription Output")
                    output_text = gr.Textbox(
                        label="Transcribed Text",
                        placeholder="Transcription will appear here...",
                        lines=7,
                        show_copy_button=True
                    )

                    with gr.Row():
                        stat_duration = gr.Textbox(label="Audio Duration", value="-", interactive=False)
                        stat_latency = gr.Textbox(label="Inference Time", value="-", interactive=False)
                        stat_device = gr.Textbox(label="Compute Device", value=asr_engine.device.upper(), interactive=False)

                    download_file_btn = gr.File(label="📥 Download Transcription (.txt)", interactive=False)

            # Button callbacks
            file_submit_btn.click(
                fn=process_audio,
                inputs=[file_audio],
                outputs=[output_text, stat_duration, stat_latency, stat_device, download_file_btn]
            )

            mic_submit_btn.click(
                fn=process_audio,
                inputs=[mic_audio],
                outputs=[output_text, stat_duration, stat_latency, stat_device, download_file_btn]
            )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    # Launch server locally on port 7860
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
