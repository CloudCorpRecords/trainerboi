"""
merge_lora.py - Merge a LoRA adapter into its base model and save a standalone 16-bit model.

The merged folder can be loaded like any HF model, pushed to the Hub, or converted to GGUF
for llama.cpp / Ollama / LM Studio with llama.cpp's convert_hf_to_gguf.py.

  python merge_lora.py --adapter runs/sft/final --out runs/sft/merged
"""

import os
import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from model_ids import resolve_base
import device_utils


def merge(adapter, out):
    with open(os.path.join(adapter, "adapter_config.json")) as f:
        base_id = resolve_base(json.load(f)["base_model_name_or_path"])
    print(f"--> Loading base {base_id} (merging needs unquantized weights)")
    model = AutoModelForCausalLM.from_pretrained(base_id, **device_utils.model_load_kwargs())
    model = device_utils.to_device(model)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
    print(f"--> Merged model saved to {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    merge(a.adapter, a.out or os.path.join(os.path.dirname(a.adapter.rstrip("/\\")), "merged"))
