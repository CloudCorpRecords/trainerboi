# 🚂 TrainerBoi launch kit

Copy-paste marketing material for launch day and beyond. Images live in
[`assets/`](../assets/): `hero.jpg` (night banner), `social-card.jpg` (day banner
with the full cast — good for social previews), `screenshots/` (real UI captures).

Tip: upload `assets/social-card.jpg` as the repo's social preview
(Settings → General → Social preview) so every shared link shows Boi.

---

## Positioning

**Tagline:** Train your own AI on your own computer.

**One-liner:** Open-source, local-first LLM training — a beginner gets a good
result in a few clicks; experts keep every setting.

**Paragraph:** TrainerBoi is a free, open-source desktop app for training and
fine-tuning language models on your own hardware — NVIDIA GPU or Apple Silicon.
Pick a goal, drop in data (or none), and it chooses the model, method, and
settings, then shows a live run with plain-English errors and automatic recovery.
No code, no cloud, no account. Everything stays on your machine.

**Feature bullets (pick 3–5 per channel):**
- Guided training: goal → data → time budget, live run view with loss curves,
  samples, GPU readings, and before/after answers
- Math Lab: teach a model maths with zero data — unlimited self-checking
  problems, learning only from verified answers
- Data tools: paste links / repos / datasets / a topic; A–F quality grades,
  one-click cleaning, synthetic Q&A grounded in your sources
- Every method: SFT, LoRA/QLoRA, DPO, ORPO, KTO, GRPO, pretraining from scratch
- Share anywhere: chat UI, zip export, Hugging Face publish, temporary public link
- Safe by default: local-only server, hot-GPU protection, no tokens stored
- Runs on Windows (NVIDIA) and macOS (Apple Silicon / Metal)

---

## X / Twitter thread

**1/** 🚂 I built TrainerBoi — train & fine-tune your own AI models on your own
computer. No code, no cloud, no account.

Pick a goal → drop in data (or none) → it trains live on your GPU.
Free & open-source (MIT). [hero image]

**2/** The guided flow does the scary parts for you:
- picks the model, method & settings for your goal
- grades your data A–F, cleans it in one click
- shows a live run: loss curve, samples, GPU temp
- retries crashes from the last checkpoint

**3/** My favorite part: Math Lab 🧮
No dataset. The model generates its own maths problems, a checker verifies every
answer, and it fine-tunes only on the ones proven correct. Verified
self-improvement, watching accuracy climb per skill.

**4/** Under the hood it's the real stack, not a toy:
SFT, LoRA/QLoRA, DPO, ORPO, KTO, GRPO, pretraining — PyTorch + Transformers +
TRL + PEFT. Then chat, export to GGUF/Ollama, or publish to Hugging Face.

**5/** And it's cute on purpose 🌸
Meet Boi, the little train: his eyes follow your mouse, he cheers milestones,
your garden grows a flower per model, and runs play out as a storybook journey.
Training should feel like this.

**6/** Windows (NVIDIA) + macOS (Apple Silicon) supported. Grab it here 👇
github.com/CloudCorpRecords/trainerboi
Stars + feedback welcome — what should Boi learn to do next?

---

## Reddit (r/LocalLLaMA)

**Title:** TrainerBoi — open-source app to fine-tune your own LLMs locally, no code needed (Windows + macOS)

**Body:**

Hey all — I built **TrainerBoi**, a free MIT-licensed desktop app for training
and fine-tuning language models on your own hardware. The goal: a total beginner
gets a good result in a few clicks, and experts keep every knob.

How a run goes:
1. Pick a goal (chat assistant, docs Q&A, personality, code explainer…)
2. Drop in files / links / repos / HF datasets — or nothing at all
3. It picks the model + method + settings, shows you *why*, and trains live
   with loss curves, samples, GPU readings, and before/after answers

Things I'm proud of:
- **Math Lab**: zero-data training — unlimited generated problems with verified
  answers; the model only learns from attempts a checker proves correct
  (documented smoke test in the README)
- **Data doctor**: A–F grades, finds boilerplate/placeholders/loops/PII/
  leakage, one-click cleaned copy
- **Honest engineering**: pre-flight checks, plain-English errors (OOM, hot GPU,
  gated model…), auto-retry from checkpoints, hot-GPU pause protection
  (laptop GPUs silently miscompute when very hot — we measured it)
- Full method menu: SFT, LoRA/QLoRA, DPO, ORPO, KTO, GRPO, pretraining
- Share via chat UI, zip, Hugging Face, or temp public link; GGUF/Ollama export

Stack: PyTorch + Transformers + TRL + PEFT, Electron + React shell, Gradio
classic studio embedded. Runs on NVIDIA (Windows) and Apple Silicon via MPS
(macOS). Everything local — local-only server, no tokens stored.

Screenshots in the README. Would love feedback, especially: what's the first
model you'd train? [link]

---

## Hacker News (Show HN)

**Title:** Show HN: TrainerBoi – Train and fine-tune LLMs on your own computer, no code

**Body:**

TrainerBoi is an open-source desktop app for local LLM training. Guided flow
(goal → data → time budget), live run dashboard, Math Lab (verified
self-improvement with zero data), data quality grading + cleaning, and the full
method menu (SFT/LoRA/DPO/ORPO/KTO/GRPO/pretrain). Exports to Hugging Face,
GGUF/Ollama, or a shareable zip. Windows + macOS, MIT licensed.

The differentiating detail I keep coming back to: laptop GPUs can silently
compute wrong results when very hot (measured ~86°C on a 3080 Ti Laptop), so
training pauses at a configurable temperature and resumes cool, with checkpoint
retries — plus a built-in GPU health check that detects it.

Happy to answer anything about the training pipeline, the MPS port, or the
verified self-improvement loop. [link]

---

## LinkedIn

🚂 Excited to share **TrainerBoi** — an open-source desktop app that lets anyone
train and fine-tune their own AI models, on their own computer.

Why it matters: fine-tuning is still gated behind scripts, cloud bills, and
accounts. TrainerBoi removes all three. Pick a goal, add your data (or none),
and watch your model train live — with plain-English guidance at every step.

Built for both worlds: beginners get recipes and a friendly guide (meet Boi, the
little train 🚂); practitioners get SFT, LoRA/QLoRA, DPO, ORPO, KTO, GRPO,
pretraining, and full hyperparameter control. Local-first and private by design.

Available for Windows and macOS. MIT licensed — contributions welcome, including
first-timers. [link + hero image]

#OpenSource #MachineLearning #LLM #LocalAI #FineTuning

---

## Short bios (directories, awesome-lists)

- **Short:** Open-source desktop app to train & fine-tune LLMs on your own GPU.
  No code, no cloud, no account.
- **Medium:** TrainerBoi is a free, local-first LLM training studio: guided
  fine-tuning, zero-data Math Lab, data quality tools, every major method
  (SFT/LoRA/DPO/GRPO…), and one-click sharing. Windows + macOS.
