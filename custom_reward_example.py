"""
custom_reward_example.py - write your own "checker" and train a model with it (reinforcement learning from verifiable rewards).

Use it:   python train_llm_grpo.py --reward custom --reward_file custom_reward_example.py --dataset my_questions.jsonl

Your dataset needs a `prompt` (or `question`) column. Any extra columns (like `answer`) are passed to reward() as keyword lists.
reward() gets the model's answers and returns one number per answer: higher = better. Anything you can CHECK with code works:
  * "the SQL query runs and returns the right rows"      * "the JSON matches my schema"       * "the regex finds the invoice number"
  * "the Python function passes these unit tests"        * "the reply is under 3 sentences"   * "the translation keeps every number"
Big labs optimise for public benchmarks. A checker for YOUR narrow task is something nobody else will build for you.
"""

import json
import re


def _text(c):
    return c[0]["content"] if isinstance(c, list) else str(c)


def reward(prompts, completions, answer=None, **kwargs):
    """Example: the reply must be valid JSON with keys 'name' and 'total', and total must match the `answer` column."""
    scores = []
    for i, c in enumerate(completions):
        t = _text(c)
        m = re.search(r"\{.*\}", t, re.S)
        s = 0.0
        try:
            obj = json.loads(m.group(0)) if m else None
            if isinstance(obj, dict):
                s += 0.3                                   # valid JSON object
                if {"name", "total"} <= set(obj):
                    s += 0.3                               # has the right keys
                    if answer is not None and str(obj["total"]).strip() == str(answer[i]).strip():
                        s += 0.4                           # and the right value
        except Exception:
            pass
        scores.append(s)
    return scores
