"""
api_server.py - High-Performance OpenAI-Compatible Local Inference API Server for TrainerBoi.
Runs on http://127.0.0.1:8000/v1 to power Cursor, Continue.dev, LibreChat, and custom apps with your GPU models.
"""

import os
import time
import json
import uuid
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
from threading import Thread
import device_utils
import trainer_utils

app = FastAPI(title="TrainerBoi OpenAI-Compatible Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_LOADED_MODEL = None
_LOADED_TOKENIZER = None
_LOADED_MODEL_ID = None

def get_or_load_model(model_name: str):
    global _LOADED_MODEL, _LOADED_TOKENIZER, _LOADED_MODEL_ID
    
    if _LOADED_MODEL_ID == model_name and _LOADED_MODEL is not None:
        return _LOADED_MODEL, _LOADED_TOKENIZER
        
    trainer_utils.clear_gpu_memory()
    quant_config, _ = device_utils.quant_config_4bit(device_utils.supports_4bit(), "the API server")
    load_kw = device_utils.model_load_kwargs()

    adapter_cfg = os.path.join(model_name, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            base_id = json.load(f).get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=quant_config,
            attn_implementation="sdpa",
            trust_remote_code=True,
            **load_kw,
        )
        model = PeftModel.from_pretrained(base, model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            attn_implementation="sdpa",
            trust_remote_code=True,
            **load_kw,
        )
    model = device_utils.to_device(model)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.eval()
    _LOADED_MODEL = model
    _LOADED_TOKENIZER = tokenizer
    _LOADED_MODEL_ID = model_name
    return _LOADED_MODEL, _LOADED_TOKENIZER

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "TrainerBoi OpenAI-Compatible API",
        "device": device_utils.device_name(),
        "docs": "http://127.0.0.1:8000/docs"
    }

@app.get("/v1/models")
def list_models():
    models = ["Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct"]
    if os.path.exists("output_qlora/final_adapter"):
        models.append("output_qlora/final_adapter")
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "owned_by": "trainerboi"} for m in models]
    }

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    try:
        model, tokenizer = get_or_load_model(req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model {req.model}: {str(e)}")

    conv = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        inputs = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    except Exception:
        text = "\n".join([f"{m.role}: {m.content}" for m in req.messages]) + "\nassistant: "
        inputs = tokenizer(text, return_tensors="pt")

    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    if req.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=req.max_tokens,
            do_sample=(req.temperature > 0),
            temperature=req.temperature or 0.7,
            top_p=req.top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
        Thread(target=model.generate, kwargs=gen_kwargs).start()

        def stream_generator():
            for chunk in streamer:
                data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    else:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=(req.temperature > 0),
                temperature=req.temperature or 0.7,
                top_p=req.top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        input_len = inputs["input_ids"].shape[1]
        reply = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": out.shape[1] - input_len,
                "total_tokens": out.shape[1]
            }
        }

if __name__ == "__main__":
    print("--> Starting TrainerBoi OpenAI-Compatible API on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
