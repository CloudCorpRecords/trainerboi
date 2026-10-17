"""
arena_tab.py - Side-by-Side Model Comparison Arena for TrainerBoi Studio.
Pit a base model against your fine-tuned model with identical prompts to evaluate before-and-after improvements.
"""

import os
import time
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import device_utils
import trainer_utils
import playground_tab

_ARENA_CACHE = {}

def get_arena_model(path, is_adapter=False):
    if path in _ARENA_CACHE:
        return _ARENA_CACHE[path]
        
    quant_config, note = device_utils.quant_config_4bit(True, "the arena")
    if note:
        print(note)
    load_kw = device_utils.model_load_kwargs()

    adapter_cfg = os.path.join(path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        import json
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            base_id = json.load(f).get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=quant_config,
            attn_implementation="sdpa",
            trust_remote_code=True,
            **load_kw,
        )
        model = PeftModel.from_pretrained(base_model, path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            path,
            quantization_config=quant_config,
            attn_implementation="sdpa",
            trust_remote_code=True,
            **load_kw,
        )
    model = device_utils.to_device(model)

    model.eval()
    _ARENA_CACHE[path] = (model, tokenizer)
    return model, tokenizer

def generate_answer(model, tokenizer, prompt, max_tokens, temperature):
    messages = [{"role": "user", "content": prompt}]
    try:
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    except Exception:
        inputs = tokenizer(f"User: {prompt}\nAssistant: ", return_tensors="pt")
        
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=(temperature > 0),
            temperature=float(temperature) if temperature > 0 else 0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    elapsed = time.time() - t0
    
    input_len = inputs["input_ids"].shape[1]
    gen_tokens = out.shape[1] - input_len
    tok_per_sec = gen_tokens / elapsed if elapsed > 0 else 0
    
    text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
    stats = f"⏱️ {elapsed:.2f}s · {gen_tokens} tokens ({tok_per_sec:.1f} tok/s)"
    return text, stats

def compare_models(prompt, model_a_path, model_b_path, max_tokens, temp):
    if not prompt or not prompt.strip():
        return "Please enter a test prompt.", "", "Please enter a test prompt.", ""
        
    try:
        model_a, tok_a = get_arena_model(model_a_path)
        ans_a, stats_a = generate_answer(model_a, tok_a, prompt, max_tokens, temp)
    except Exception as e:
        ans_a, stats_a = f"Error with Model A: {str(e)}", "Failed"
        
    try:
        model_b, tok_b = get_arena_model(model_b_path)
        ans_b, stats_b = generate_answer(model_b, tok_b, prompt, max_tokens, temp)
    except Exception as e:
        ans_b, stats_b = f"Error with Model B: {str(e)}", "Failed"
        
    return ans_a, stats_a, ans_b, stats_b

def clear_arena_cache():
    global _ARENA_CACHE
    _ARENA_CACHE.clear()
    trainer_utils.clear_gpu_memory()
    return "🧹 Arena memory cleared."

def build_tab():
    with gr.TabItem("⚔️ Compare", id="arena"):
        gr.Markdown("### ⚔️ Side-by-Side Model Arena (Before vs. After)")
        gr.Markdown("Test a base model against your fine-tuned model on identical prompts to evaluate quality improvements in real time.")
        
        with gr.Row():
            avail_models = playground_tab.get_available_models()
            with gr.Column():
                model_a = gr.Dropdown(
                    choices=avail_models,
                    value=avail_models[0] if avail_models else "Qwen/Qwen2.5-0.5B-Instruct",
                    label="Model A (e.g. Original Base Model)",
                    allow_custom_value=True
                )
            with gr.Column():
                model_b = gr.Dropdown(
                    choices=avail_models,
                    value=avail_models[1] if len(avail_models) > 1 else (avail_models[0] if avail_models else "output_qlora/final_adapter"),
                    label="Model B (e.g. Your Fine-Tuned Model / LoRA)",
                    allow_custom_value=True
                )
                
        with gr.Row():
            prompt_input = gr.Textbox(
                label="Evaluation Prompt / Question",
                placeholder="Type a test question to ask both models...",
                lines=3,
                value="Explain why mixed precision training is faster on modern GPUs."
            )
            
        with gr.Row():
            max_tok = gr.Slider(minimum=64, maximum=512, step=32, value=256, label="Max Tokens")
            temp_slider = gr.Slider(minimum=0.0, maximum=1.2, step=0.05, value=0.7, label="Temperature")
            compare_btn = gr.Button("⚔️ Run Side-by-Side Comparison", variant="primary", scale=2)
            clear_btn = gr.Button("🧹 Clear VRAM Cache", scale=1)
            
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 🅰️ Model A Response")
                response_a = gr.Textbox(label="Output", lines=10, interactive=False)
                stats_a = gr.Markdown("Tokens / Speed")
            with gr.Column():
                gr.Markdown("#### 🅱️ Model B Response")
                response_b = gr.Textbox(label="Output", lines=10, interactive=False)
                stats_b = gr.Markdown("Tokens / Speed")
                
        arena_status = gr.Markdown("")
        
        compare_btn.click(
            compare_models,
            inputs=[prompt_input, model_a, model_b, max_tok, temp_slider],
            outputs=[response_a, stats_a, response_b, stats_b]
        )
        clear_btn.click(
            clear_arena_cache,
            outputs=[arena_status]
        )
        
    return {
        "compare_btn": compare_btn
    }
