# Contributing

Thanks for helping make model training easier for everyone.

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
