"""
audio_studio_tab.py - Audio & Speech-to-Text Studio for TrainerBoi.
Real-time GPU transcription and custom speech model fine-tuning with OpenAI Whisper.
"""

import gradio as gr
import train_audio

def run_transcription(audio_file, model_choice):
    if audio_file is None:
        return "Please upload an audio file or record speech."
    return train_audio.transcribe_audio(audio_file, model_choice)

def run_whisper_training(model_choice, epochs):
    return train_audio.fine_tune_whisper_demo(epochs=int(epochs))

def build_tab():
    with gr.TabItem("🎙️ Audio", id="audio"):
        gr.Markdown("### 🎙️ Audio Transcription & Speech-to-Text Studio (OpenAI Whisper)")
        gr.Markdown("Transcribe audio in real time on your GPU or fine-tune Whisper on domain-specific speech.")
        
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("#### 🎧 Instant GPU Audio Transcriber")
                audio_input = gr.Audio(type="filepath", label="Upload Audio or Record Voice")
                whisper_model = gr.Dropdown(
                    choices=["openai/whisper-tiny", "openai/whisper-base"],
                    value="openai/whisper-tiny",
                    label="Whisper Architecture"
                )
                transcribe_btn = gr.Button("⚡ Transcribe Audio on GPU", variant="primary")
                transcription_result = gr.Textbox(label="Transcription Output", lines=4)
                
            with gr.Column(scale=3):
                gr.Markdown("#### 🛠️ Whisper Domain Adaptation / Fine-Tuning")
                gr.Markdown("Adapt Whisper to specialized terminology, foreign accents, or industry vocabulary.")
                dataset_audio = gr.File(label="Upload Audio Dataset (.zip of wav files + transcript.csv)")
                audio_epochs = gr.Slider(minimum=1, maximum=5, step=1, value=2, label="Fine-Tuning Epochs")
                train_whisper_btn = gr.Button("🚀 Start Whisper Fine-Tuning")
                train_whisper_status = gr.Markdown("ℹ️ Upload audio pairs and start fine-tuning.")
                
        transcribe_btn.click(
            run_transcription,
            inputs=[audio_input, whisper_model],
            outputs=[transcription_result]
        )
        train_whisper_btn.click(
            run_whisper_training,
            inputs=[whisper_model, audio_epochs],
            outputs=[train_whisper_status]
        )
        
    return {
        "transcribe_btn": transcribe_btn
    }
