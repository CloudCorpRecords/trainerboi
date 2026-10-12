"""
llm_data.py - Shared dataset loading & normalization for the LLM training scripts.

Accepts a Hugging Face Hub dataset id OR a local file (.jsonl / .json / .csv / .parquet / .txt)
and normalizes common schemas into formats TRL understands natively:

  * {"messages": [{"role": ..., "content": ...}, ...]}       conversational (preferred)
  * {"prompt": ..., "completion": ...}                         prompt/completion
  * {"text": ...}                                              raw text (pretraining / continued PT)
  * {"prompt": ..., "chosen": ..., "rejected": ...}            preference pairs (DPO)
"""

import os
from datasets import load_dataset, Dataset, DatasetDict

LOCAL_EXT = {".jsonl": "json", ".json": "json", ".csv": "csv", ".parquet": "parquet", ".txt": "text"}

# Fallback for base models that ship without a chat template (ChatML format).
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

DEMO_ROWS = [
    ("Explain quantum computing in one sentence.", "Quantum computers use qubits that can be in superpositions, letting some problems be solved far faster than with classical bits."),
    ("Write a Python function for factorial.", "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"),
    ("What is LoRA?", "LoRA fine-tunes a model by training small low-rank adapter matrices instead of all weights, cutting VRAM use dramatically."),
    ("What is CUDA?", "CUDA is NVIDIA's parallel computing platform for running general-purpose code on GPUs."),
    ("How does gradient descent work?", "It repeatedly nudges parameters in the direction that most reduces the loss, scaled by a learning rate."),
]


def load_raw(source, split="train", subset=None):
    """Load a dataset split from a local path or the Hugging Face Hub. 'demo' gives a tiny built-in set."""
    if source == "demo":
        return Dataset.from_dict({
            "messages": [[{"role": "user", "content": q}, {"role": "assistant", "content": a}] for q, a in DEMO_ROWS] * 20
        })
    if os.path.isdir(source):
        # A Dataset Builder folder: sharded parquet (train-00000.parquet ...) or train.jsonl.
        import glob
        prefix = "eval" if split in ("eval", "validation", "test") else "train"
        for pattern, builder in ((f"{prefix}-*.parquet", "parquet"), (f"{prefix}*.jsonl", "json")):
            files = sorted(glob.glob(os.path.join(source, pattern)))
            if files:
                return load_dataset(builder, data_files=files, split="train")
        raise ValueError(f"No {prefix} data (.parquet shards or .jsonl) found in folder {source}")
    if os.path.exists(source):
        ext = os.path.splitext(source)[1].lower()
        if ext not in LOCAL_EXT:
            raise ValueError(f"Unsupported file type '{ext}'. Use one of: {', '.join(LOCAL_EXT)}")
        return load_dataset(LOCAL_EXT[ext], data_files=source, split="train")
    return load_dataset(source, subset, split=split)


def _to_messages(turns):
    """ShareGPT style [{'from': 'human', 'value': ...}] -> OpenAI style messages."""
    role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
                "system": "system", "model": "assistant"}
    return [{"role": role_map.get(t.get("from", t.get("role")), "user"),
             "content": t.get("value", t.get("content", ""))} for t in turns]


def normalize_sft(ds, text_field=None):
    """Map common instruction-tuning schemas to 'messages' / 'prompt'+'completion' / 'text'."""
    cols = set(ds.column_names)
    if text_field:
        return ds.rename_column(text_field, "text").select_columns(["text"]) if text_field != "text" else ds.select_columns(["text"])
    if "messages" in cols:
        return ds.select_columns(["messages"])
    if "conversations" in cols:  # ShareGPT
        return ds.map(lambda r: {"messages": _to_messages(r["conversations"])}, remove_columns=list(cols))
    if {"instruction", "output"} <= cols:  # Alpaca
        def alpaca(r):
            user = r["instruction"] + (f"\n\n{r['input']}" if r.get("input") else "")
            return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": r["output"]}]}
        return ds.map(alpaca, remove_columns=list(cols))
    if {"question", "answer"} <= cols:
        return ds.map(lambda r: {"messages": [{"role": "user", "content": r["question"]},
                                              {"role": "assistant", "content": r["answer"]}]}, remove_columns=list(cols))
    if {"prompt", "completion"} <= cols:
        return ds.select_columns(["prompt", "completion"])
    if "text" in cols:
        return ds.select_columns(["text"])
    raise ValueError(f"Can't infer dataset format from columns {sorted(cols)}. "
                     "Expected messages / conversations / instruction+output / question+answer / prompt+completion / text, "
                     "or pass --text_field.")


def render_text(ds, tokenizer):
    """Flatten messages / prompt+completion into a single 'text' column using the chat template."""
    cols = ds.column_names
    if cols == ["text"]:
        return ds

    def render(r):
        if "messages" in r:
            msgs = r["messages"]
        else:
            msgs = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["completion"]}]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}
    return ds.map(render, remove_columns=cols)


def normalize_dpo(ds):
    cols = set(ds.column_names)
    if not {"chosen", "rejected"} <= cols:
        raise ValueError(f"DPO data needs 'chosen' and 'rejected' columns (plus optional 'prompt'); got {sorted(cols)}")
    keep = [c for c in ("prompt", "chosen", "rejected") if c in cols]
    return ds.select_columns(keep)


def split_train_eval(ds, eval_ratio, seed=42):
    if not eval_ratio or eval_ratio <= 0 or len(ds) < 20:
        return ds, None
    parts = ds.train_test_split(test_size=eval_ratio, seed=seed)
    return parts["train"], parts["test"]


def ensure_chat_template(tokenizer):
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = CHATML_TEMPLATE
        print("--> Tokenizer had no chat template; using ChatML.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
