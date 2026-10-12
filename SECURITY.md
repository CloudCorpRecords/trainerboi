# Security

TrainerBoi runs entirely on your own computer. This page explains what it does to stay safe and how to report a problem.

## How it protects you

* **Local only.** The server listens on `127.0.0.1` and refuses requests whose `Host` is not local (blocks DNS-rebinding) or whose `Origin` /
  `Sec-Fetch-Site` shows another website (blocks cross-site requests), so a web page you visit cannot drive your training or read your models.
* **Desktop shell.** The Electron window runs with context isolation, no Node integration, all browser permissions denied, and cannot navigate away
  from its own local server (external links open in your normal browser).
* **Paths are validated.** Run, dataset and model names from the UI are checked for path separators and `..`; downloads are restricted to the
  `exports/` folder and "show in folder" to the app folder (symlinks resolved).
* **No secrets stored.** A Hugging Face token you paste to publish a model is used for that single upload and is never written to disk or logged.
* **Your data stays here.** Nothing is uploaded unless you press Publish / Create link. Models, datasets and logs live in `runs/`, `data/` and
  `exports/` (all git-ignored).
* **Dependencies** are pinned in `requirements-app.txt`; the JavaScript dependencies are checked with `npm audit`.

## Things to be aware of

* **Temporary public chat link** (My models → Share) exposes a chat page through Gradio's relay to anyone with the link, until you stop it or about a
  week passes. Only share links with people you trust.
* **Custom reward files** (`--reward custom --reward_file x.py`) and notebooks are Python and run with your user's permissions - only use files you trust.
* **Models and datasets from the internet** carry their own licences and may contain bad or private data. Check the licence (shown on the model card)
  before sharing, and use the Data doctor to look for PII and junk.
* **Unsigned installers** trigger Windows SmartScreen. Build from source, or verify the checksum of release files.

## Reporting a vulnerability

Please open a private security advisory on GitHub (Security → Report a vulnerability) rather than a public issue. Include steps to reproduce and the
version. We aim to respond within a few days.
