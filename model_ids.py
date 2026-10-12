"""
model_ids.py - tiny helper: turn Unsloth's pre-quantised repo ids back into the normal model id.

Unsloth QLoRA runs record the base model as e.g. 'unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit' in adapter_config.json. That repo holds
4-bit weights, so loading it as a normal bf16 model fails, and anyone you share the adapter with would need the wrong model. The adapter
works with the regular full-precision model, so we point at that instead.
"""

import re

_SUFFIX = re.compile(r"-(unsloth-)?bnb-4bit$|-(unsloth-)?bnb-8bit$", re.I)


def resolve_base(model_id):
    if not model_id or not isinstance(model_id, str):
        return model_id
    if not _SUFFIX.search(model_id):
        return model_id
    name = _SUFFIX.sub("", model_id)
    owner, _, rest = name.partition("/")
    if owner.lower() == "unsloth" and rest.lower().startswith("qwen"):
        return "Qwen/" + rest
    if owner.lower() == "unsloth" and rest.lower().startswith("smollm"):
        return "HuggingFaceTB/" + rest
    return name
