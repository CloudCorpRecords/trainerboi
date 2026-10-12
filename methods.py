"""
methods.py - Plain-English catalogue of training methods + a recommender ("which one should I use?").
Used by the Methods tab and the Home screen. `task` is the key of live_dashboard.TASKS when the app can run it.
"""

METHODS = [
    {
        "id": "sft", "name": "Fine-tuning (SFT)", "icon": "🎓", "level": "Start here",
        "task": "SFT fine-tune",
        "tagline": "Show the model examples of good answers and it learns to imitate them.",
        "analogy": "Like a student studying solved example problems.",
        "data": "Question → answer pairs (or chats, documents turned into Q&A).",
        "gpu": "6-16 GB", "time": "Minutes to hours",
        "use_when": "You want a model that knows your topic, follows your format or speaks in your voice.",
        "avoid_when": "You only have 'good vs bad' ratings, or you can check answers automatically (see GRPO).",
    },
    {
        "id": "lora", "name": "LoRA / QLoRA (cheap fine-tuning)", "icon": "🪶", "level": "Start here",
        "task": "SFT fine-tune",
        "tagline": "Train a tiny add-on instead of the whole model. QLoRA also shrinks the base model to 4-bit.",
        "analogy": "Like adding sticky notes to a textbook instead of rewriting it.",
        "data": "Same as fine-tuning.",
        "gpu": "4-12 GB", "time": "Minutes to hours",
        "use_when": "Almost always. It is how a laptop GPU can tune a 3-8 billion parameter model.",
        "avoid_when": "You need to change the model deeply (new language, new skills) - consider full fine-tuning or continued pretraining.",
    },
    {
        "id": "dpo", "name": "Preference tuning (DPO)", "icon": "⚖️", "level": "Next step",
        "task": "DPO preference tuning",
        "tagline": "Show pairs of a good and a bad answer; the model learns to lean toward the good one.",
        "analogy": "Like a coach saying 'do it this way, not that way'.",
        "data": "Prompt + chosen answer + rejected answer.",
        "gpu": "8-16 GB", "time": "Under an hour for small sets",
        "use_when": "The model already works but its tone, safety or helpfulness needs polish.",
        "avoid_when": "The model doesn't know the subject yet - teach it first with fine-tuning.",
    },
    {
        "id": "orpo", "name": "ORPO (teach + polish in one go)", "icon": "⚡", "level": "Next step",
        "task": "ORPO (SFT + preference in one step)",
        "tagline": "Learns the style from the good answers and avoids the bad ones - in a single training run.",
        "analogy": "Like studying solved examples and common mistakes at the same time.",
        "data": "Same pairs as DPO (prompt / chosen / rejected).",
        "gpu": "6-14 GB", "time": "Faster than SFT then DPO",
        "use_when": "You have preference pairs and want one simple run. Needs less memory than DPO.",
        "avoid_when": "You have no 'bad' answers to contrast against.",
    },
    {
        "id": "kto", "name": "KTO (thumbs up / thumbs down)", "icon": "👍", "level": "Next step",
        "task": "KTO (thumbs up / down feedback)",
        "tagline": "Learns from simple good/bad labels on single answers - no matched pairs needed.",
        "analogy": "Like a like/dislike button on a chatbot's replies.",
        "data": "Prompt + answer + a good/bad label. Pairs also work (they are split automatically).",
        "gpu": "6-14 GB", "time": "Under an hour for small sets",
        "use_when": "You collected user feedback (thumbs up/down) but never wrote better-vs-worse pairs.",
        "avoid_when": "The model doesn't know the subject yet - teach it first with fine-tuning.",
    },
    {
        "id": "mathlab", "name": "Verified self-improvement (Math Lab)", "icon": "🧮", "level": "Unique",
        "task": "GRPO (reinforcement learning)", "tab": "math",
        "tagline": "The model practises endless self-checking problems, keeps only the attempts a checker proves correct, and learns from those.",
        "analogy": "Like a student doing practice problems, marking them against an answer key, and re-studying only their own correct work.",
        "data": "None. Problems are generated with their exact answers (maths today; any task you can write a checker for).",
        "gpu": "6-16 GB", "time": "About 10-20 min per round on a small model",
        "use_when": "You can CHECK answers with code (maths, code, formats, rules). No labels, no teacher model, aims practice at weak skills.",
        "avoid_when": "Answers can't be verified automatically (opinions, creative writing) - use fine-tuning or preference methods.",
    },
    {
        "id": "grpo", "name": "Reinforcement learning (GRPO)", "icon": "🎯", "level": "Advanced",
        "task": "GRPO (reinforcement learning)",
        "tagline": "The model tries several answers, a scoring rule rewards the good ones, and it improves by trial and error.",
        "analogy": "Like practicing a game and keeping what scores points. The method behind 'reasoning' models.",
        "data": "Only questions - plus a way to score answers (right math answer, short length, a required format...).",
        "gpu": "8-16 GB", "time": "Slow (it generates a lot)",
        "use_when": "Answers can be checked automatically (math, code, formats), or you want shorter/longer/structured replies.",
        "avoid_when": "You have no way to score answers. Start with 0.5-1.5B models; bigger ones are very slow on a laptop.",
    },
    {
        "id": "pretrain", "name": "Train from scratch (pretraining)", "icon": "🏗️", "level": "Learn how it works",
        "task": "Pretrain from scratch",
        "tagline": "Build a brand-new model from random numbers and teach it a language from raw text.",
        "analogy": "Like raising a child who has never heard a word - slow, but you see everything happen.",
        "data": "Lots of plain text.",
        "gpu": "6-16 GB (tiny models)", "time": "Minutes (tiny) to days",
        "use_when": "Education, research, or a very small specialised model.",
        "avoid_when": "You want a useful assistant - fine-tuning an existing model is 1000x cheaper.",
    },
    {
        "id": "full", "name": "Full fine-tuning", "icon": "🔧", "level": "Advanced",
        "task": "SFT fine-tune",
        "tagline": "Update every weight of a small model. Most powerful, most memory hungry.",
        "analogy": "Like rewriting the whole textbook.",
        "data": "Same as fine-tuning, usually more of it.",
        "gpu": "12-16 GB for models up to ~1B", "time": "Hours",
        "use_when": "Small models (under ~1B) where you want maximum change.",
        "avoid_when": "Bigger models or small datasets (it forgets old knowledge and overfits). Use LoRA.",
    },
    {
        "id": "distill", "name": "Distillation (learn from a bigger model)", "icon": "🧪", "level": "Next step",
        "task": "SFT fine-tune",
        "tagline": "A bigger 'teacher' model writes the training data; a small 'student' learns from it. The Dataset Builder's synthetic mode does this for you.",
        "analogy": "Like a master craftsperson writing lessons for an apprentice.",
        "data": "None needed - just your documents or a topic. The teacher writes the questions and answers.",
        "gpu": "8-16 GB", "time": "Data generation + a fine-tune",
        "use_when": "You have documents but no Q&A data. This is the easiest way to start with zero data.",
        "avoid_when": "The teacher model is wrong about your topic - always spot-check generated data in the Library.",
    },
]

BY_ID = {m["id"]: m for m in METHODS}

# --- "help me choose" -----------------------------------------------------------------------
HAVE = {
    "Nothing yet - just an idea or topic": "nothing",
    "Documents, notes, PDFs, a website or a code repo": "docs",
    "Example questions and answers / chats": "qa",
    "Pairs of good vs. bad answers": "pairs",
    "Answers rated thumbs up / thumbs down": "ratings",
    "Only questions, but I can check the answers automatically": "checkable",
    "Just lots of raw text": "raw",
}
WANT = {
    "Know my subject (documents, products, code)": "knowledge",
    "Talk in a certain style or persona": "style",
    "Be more helpful, safer or more polite": "polish",
    "Reason better / get answers right (math, code)": "reason",
    "Learn how language models work": "learn",
}


def recommend(have, want):
    """Return (method_id, why, alternatives[list of ids]) for a (have, want) pair of keys from HAVE / WANT."""
    if want == "learn" or have == "raw":
        return "pretrain", "Training from scratch on raw text is the clearest way to see how language models are built.", ["sft"]
    if have == "nothing":
        if want == "reason":
            return "mathlab", "No data needed: Math Lab generates unlimited practice problems with exact answers, checks every attempt, and the model learns only from its own verified successes.", ["grpo"]
        return "distill", "No data? A bigger model can write practice material for you from a topic or from public text, then your model learns from it.", ["sft"]
    if have == "docs":
        return "distill", "Your documents get turned into question-and-answer practice automatically, then your model is fine-tuned on it with LoRA.", ["lora"]
    if have == "checkable" or want == "reason":
        return "grpo", "When answers can be scored automatically, reinforcement learning improves results without writing answer examples.", ["sft"]
    if have == "ratings":
        return "kto", "Single answers with good/bad labels are exactly what KTO learns from - no paired answers required.", ["dpo"]
    if have == "pairs":
        return "orpo", "Preference pairs are best used directly. ORPO teaches and polishes in one run; DPO is the classic alternative.", ["dpo"]
    if want == "polish":
        return "dpo", "Polishing tone and helpfulness is exactly what preference tuning is for - use a public preference set if you have none.", ["orpo"]
    return "lora", "Example answers + LoRA is the proven, cheap and reliable way to teach style or knowledge.", ["sft"]
