"""
chat_llm.py - Chat with a base model, a LoRA adapter, or a merged/full fine-tuned model (streaming).

Examples:
  python chat_llm.py --model runs/sft/final                 # adapter dir: base model is read from adapter_config.json
  python chat_llm.py --model Qwen/Qwen2.5-1.5B-Instruct --load_4bit
  python chat_llm.py --model runs/pretrain/final --raw       # plain text completion (no chat template)
"""

import os
import json
import argparse
from threading import Thread

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

import device_utils
import llm_data
from model_ids import resolve_base


def load(model_path, load_4bit=False):
    quant, note = device_utils.quant_config_4bit(load_4bit, "chat")
    if note:
        print(note)
    load_kw = device_utils.model_load_kwargs()
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        from peft import PeftModel
        with open(adapter_cfg) as f:
            base_id = resolve_base(json.load(f)["base_model_name_or_path"])
        print(f"--> Base {base_id} + adapter {model_path}")
        model = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=quant, **load_kw)
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quant, **load_kw)
    model = device_utils.to_device(model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model.eval(), tokenizer


def main(args):
    model, tokenizer = load(args.model, args.load_4bit)
    if not args.raw:
        llm_data.ensure_chat_template(tokenizer)
    history = [{"role": "system", "content": args.system}] if args.system else []
    print("Type a message ('/reset' clears history, '/quit' exits).\n")
    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user == "/quit":
            break
        if user == "/reset":
            history = history[:1] if args.system else []
            continue
        if not user:
            continue

        if args.raw:
            inputs = tokenizer(user, return_tensors="pt").to(model.device)
        else:
            history.append({"role": "user", "content": user})
            inputs = tokenizer.apply_chat_template(history, add_generation_prompt=True,
                                                   return_tensors="pt", return_dict=True).to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen = dict(**inputs, streamer=streamer, max_new_tokens=args.max_new_tokens,
                   do_sample=args.temperature > 0, temperature=args.temperature or None, top_p=0.9)
        Thread(target=model.generate, kwargs=gen).start()
        print("bot > ", end="", flush=True)
        reply = ""
        for chunk in streamer:
            print(chunk, end="", flush=True)
            reply += chunk
        print("\n")
        if not args.raw:
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Interactive chat with a trained model")
    p.add_argument("--model", required=True, help="HF id, merged model dir, or LoRA adapter dir")
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--raw", action="store_true", help="Plain completion, no chat template")
    p.add_argument("--system", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=512)
    main(p.parse_args())
