# Contributing

Thanks for helping make model training easier for everyone. You don't need to be an ML expert to help: fixing a confusing message, improving a doc, or adding a recipe are all great first contributions.

## Your first contribution

1. Pick an issue labelled [`good first issue`](https://github.com/CloudCorpRecords/trainerboi/labels/good%20first%20issue) or [`help wanted`](https://github.com/CloudCorpRecords/trainerboi/labels/help%20wanted), and leave a comment so others know you're on it.
2. Fork the repo, make a branch, and follow *Run it from source* below. No NVIDIA GPU? UI work (`ui/`) and docs don't need one.
3. Open a pull request; the template lists the checks. Small PRs are welcome and get reviewed fastest.

Stuck or have an idea that isn't an issue yet? Ask in [Discussions](https://github.com/CloudCorpRecords/trainerboi/discussions). Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Run it from source

```powershell
setup.bat                     # Python 3.11 env + PyTorch for your GPU + libraries + UI (needs uv and Node 20+)
START.bat                     # opens http://127.0.0.1:7860
```

## Layout

| Path | What |
|---|---|
| `ui/` | React front end (Vite). `npm run dev` proxies the API to :7860; `npm run build` writes `ui/dist`. |
| `studio_server.py` | FastAPI backend + static UI + embedded classic studio at `/classic`. |
| `desktop/` | Electron shell, first-run installer (Python + PyTorch), packaging (`npm run dist`). |
| `train_llm_*.py`, `pretrain_llm.py`, `math_lab.py`, `math_gym.py` | Training methods. |
| `dataset_builder.py`, `data_analyzer.py`, `data_quality.py`, `data_intake.py` | Data collection, quality rules and the Data doctor. |
| `autopilot*.py` | The "pick everything for me" planner and runner behind Guided training. |
| `hw_info.py`, `gpu_guard.py` | Live hardware readings and the hot-GPU safety pause. |
| `models_registry.py` | Finds your models and builds shareable packages. |
| `app.py`, `*_tab.py` | The classic Gradio studio (advanced tools). |

## Guidelines

* Keep it beginner-friendly: plain-English messages, sensible defaults, no jargon in the main screens.
* Anything that can fail on a user's machine (out of memory, hot GPU, no internet) should explain what happened and what to do (`friendly_errors.py`).
* Never log or store tokens. Validate any name or path that comes from the browser (`nm()` in `studio_server.py`).
* Before opening a PR: `python -m compileall -q .` and `cd ui && npm run build` should pass. Describe what you tested and on which GPU.
* New training methods: add a script, register it in `methods.py`, and (optionally) a launcher entry in `live_dashboard.py`.

## Reporting bugs

Include your GPU, driver version, Windows version, and the technical log from the failing run (Run page → Show technical log). Remove anything private.
