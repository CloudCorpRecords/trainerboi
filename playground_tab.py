"""
playground_tab.py - Interactive Model Playground & Chat for TrainerBoi.
Allows users to instantly test and chat with pretrained models or locally trained LoRA adapters.
"""

import os
import glob
import json
import torch
import gradio as gr
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import device_utils
import trainer_utils

_CURRENT_MODEL = None
_CURRENT_TOKENIZER = None
_CURRENT_MODEL_ID = None

def get_available_models():
    """Scans for available local trained checkpoints and common pretrained models."""
    models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "unsloth/Llama-3.2-1B-Instruct"
    ]
    
    # Check for local LoRA adapters in runs/
    for adapter_file in glob.glob("runs/**/adapter_config.json", recursive=True):
        adapter_dir = os.path.dirname(adapter_file)
        if adapter_dir not in models:
            models.append(adapter_dir)
            
    # Check for output_qlora
    if os.path.exists("output_qlora/final_adapter"):
        models.append("output_qlora/final_adapter")
        
    return models

def load_model_pipeline(model_path, load_4bit=True):
    global _CURRENT_MODEL, _CURRENT_TOKENIZER, _CURRENT_MODEL_ID
    
    if _CURRENT_MODEL_ID == model_path and _CURRENT_MODEL is not None:
        stats = trainer_utils.get_gpu_telemetry()
        return f"✅ Already loaded: **{model_path}** ({stats['vram_allocated_gb']} GB VRAM in use)"
    
    # Unload existing model
    unload_model()
    
    quant_config, note = device_utils.quant_config_4bit(load_4bit, "the playground")
    if note:
        print(note)
    load_kw = device_utils.model_load_kwargs()

    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        from peft import PeftModel
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            base_id = json.load(f).get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=quant_config,
            trust_remote_code=True,
            **load_kw,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            trust_remote_code=True,
            **load_kw,
        )
    model = device_utils.to_device(model)

    model.eval()
    _CURRENT_MODEL = model
    _CURRENT_TOKENIZER = tokenizer
    _CURRENT_MODEL_ID = model_path
    
    stats = trainer_utils.get_gpu_telemetry()
    return (f"🚀 Successfully loaded **{model_path}** on {device_utils.device_label()} "
            f"({stats['vram_allocated_gb']} GB VRAM allocated)")

def unload_model():
    global _CURRENT_MODEL, _CURRENT_TOKENIZER, _CURRENT_MODEL_ID
    _CURRENT_MODEL = None
    _CURRENT_TOKENIZER = None
    _CURRENT_MODEL_ID = None
    trainer_utils.clear_gpu_memory()
    stats = trainer_utils.get_gpu_telemetry()
    return f"🧹 Model unloaded. VRAM freed! ({stats['vram_free_gb']} GB available)"

def stream_chat_response(message, history, system_prompt, temperature, max_tokens):
    global _CURRENT_MODEL, _CURRENT_TOKENIZER
    
    if _CURRENT_MODEL is None or _CURRENT_TOKENIZER is None:
        yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": "⚠️ No model is loaded! Please select a model and click **'Load Model'** above first."}]
        return

    # Build messages list
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
        
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
        
    messages.append({"role": "user", "content": message})
    
    # Try chat template
    try:
        inputs = _CURRENT_TOKENIZER.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
    except Exception:
        # Fallback to plain prompt string
        prompt_text = ""
        for m in messages:
            prompt_text += f"{m['role'].capitalize()}: {m['content']}\n"
        prompt_text += "Assistant: "
        inputs = _CURRENT_TOKENIZER(prompt_text, return_tensors="pt")
        
    device = _CURRENT_MODEL.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    streamer = TextIteratorStreamer(_CURRENT_TOKENIZER, skip_prompt=True, skip_special_tokens=True)
    
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(max_tokens),
        do_sample=(temperature > 0),
        temperature=float(temperature) if temperature > 0 else 0.7,
        top_p=0.9,
        pad_token_id=_CURRENT_TOKENIZER.pad_token_id or _CURRENT_TOKENIZER.eos_token_id
    )
    
    thread = Thread(target=_CURRENT_MODEL.generate, kwargs=gen_kwargs)
    thread.start()
    
    partial_reply = ""
    current_chat = list(history) + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    
    for chunk in streamer:
        partial_reply += chunk
        current_chat[-1]["content"] = partial_reply
        yield current_chat

def build_tab():
    with gr.TabItem("💬 Chat", id="chat"):
        gr.Markdown("### Test & Chat with Any Model in Real-Time on Your GPU")
        gr.Markdown("Pick an off-the-shelf model or one of your locally fine-tuned checkpoints to interact with it right now.")
        
        with gr.Row():
            with gr.Column(scale=3):
                model_choices = get_available_models()
                selected_model = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0] if model_choices else "Qwen/Qwen2.5-0.5B-Instruct",
                    label="Model or Local Adapter Directory",
                    allow_custom_value=True
                )
            with gr.Column(scale=1):
                load_4bit = gr.Checkbox(label="Load in 4-bit (Saves VRAM)", value=True)
                refresh_models_btn = gr.Button("🔄 Refresh Local Models", size="sm")
            with gr.Column(scale=2):
                load_btn = gr.Button("🚀 Load Model to GPU", variant="primary")
                unload_btn = gr.Button("🧹 Unload Model / Free VRAM")
                
        status_md = gr.Markdown("ℹ️ Model not loaded yet. Click **'Load Model to GPU'** to begin.")
        
        refresh_models_btn.click(
            lambda: gr.update(choices=get_available_models()),
            outputs=[selected_model]
        )
        
        load_btn.click(
            load_model_pipeline,
            inputs=[selected_model, load_4bit],
            outputs=[status_md]
        )
        
        unload_btn.click(
            unload_model,
            outputs=[status_md]
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label="Chat Conversation", height=450)
                with gr.Row():
                    user_msg = gr.Textbox(
                        placeholder="Type a message or instruction and press Enter...",
                        label="Your Message",
                        scale=5
                    )
                    send_btn = gr.Button("Send 📨", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️ Clear Chat History", size="sm")
                
            with gr.Column(scale=2):
                gr.Markdown("#### Generation Controls")
                temp_slider = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=0.7, label="Temperature (Creativity)")
                tokens_slider = gr.Slider(minimum=32, maximum=1024, step=32, value=350, label="Max Tokens")
                sys_prompt = gr.Textbox(
                    label="System Prompt (Optional)",
                    placeholder="e.g. You are a helpful, concise AI assistant.",
                    lines=3
                )
                
        def on_send(msg, hist, sys_p, temp, tok):
            if not msg or not msg.strip():
                yield hist, ""
                return
            for update in stream_chat_response(msg, hist, sys_p, temp, tok):
                yield update, ""

        user_msg.submit(
            on_send,
            inputs=[user_msg, chatbot, sys_prompt, temp_slider, tokens_slider],
            outputs=[chatbot, user_msg]
        )
        send_btn.click(
            on_send,
            inputs=[user_msg, chatbot, sys_prompt, temp_slider, tokens_slider],
            outputs=[chatbot, user_msg]
        )
        clear_btn.click(lambda: [], outputs=[chatbot])

    return {
        "selected_model": selected_model,
        "load_btn": load_btn,
        "chatbot": chatbot
    }
