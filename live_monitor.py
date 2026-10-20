"""
live_monitor.py - Streams live training telemetry to disk so the Studio dashboard can show it in real time.

Each run writes to <output_dir>/live/:
  status.json    current phase, progress, speed, ETA, latest metrics (rewritten ~1x/sec)
  metrics.jsonl  one line per log/eval event (loss, lr, grad norm, VRAM, ...)
  samples.jsonl  periodic text generations from a fixed prompt, so you can watch the model learn
  console.log    full stdout/stderr (written by the Studio when it launches the run)

Usage inside a training script:
  live = LiveMonitor(output_dir, task="sft", config=vars(args), tokenizer=tok, sample_prompt="...")
  live.phase("Loading model")
  trainer = SFTTrainer(..., callbacks=[live.callback()])
  with live.guard(): trainer.train()
"""

import os
import json
import math
import time
import traceback
import contextlib
import subprocess

import device_utils

from transformers import TrainerCallback


_NVML_INIT = False
_NVML_DEV = None

def _nvml_init():
    global _NVML_INIT, _NVML_DEV
    if not _NVML_INIT:
        try:
            import pynvml
            pynvml.nvmlInit()
            _NVML_DEV = pynvml.nvmlDeviceGetHandleByIndex(0)
            _NVML_INIT = True
        except Exception:
            _NVML_INIT = False

def nvidia_smi():
    """Whole-GPU stats using direct C-API NVML (<1ms), with fallback to nvidia-smi.

    On Apple Silicon returns MPS memory stats instead (util/temp/power are None:
    macOS does not expose them). Returns {} when no accelerator data is available.
    """
    _nvml_init()
    if _NVML_INIT:
        try:
            import pynvml
            util = float(pynvml.nvmlDeviceGetUtilizationRates(_NVML_DEV).gpu)
            mem = pynvml.nvmlDeviceGetMemoryInfo(_NVML_DEV)
            used = float(mem.used) / (1024 ** 2)
            total = float(mem.total) / (1024 ** 2)
            temp = float(pynvml.nvmlDeviceGetTemperature(_NVML_DEV, pynvml.NVML_TEMPERATURE_GPU))
            power = float(pynvml.nvmlDeviceGetPowerUsage(_NVML_DEV) / 1000.0)
            clock = float(pynvml.nvmlDeviceGetClockInfo(_NVML_DEV, pynvml.NVML_CLOCK_SM))
            return {"gpu_util": util, "vram_used_gb": used / 1024, "vram_total_gb": total / 1024,
                    "temp_c": temp, "power_w": power, "sm_clock_mhz": clock}
        except Exception:
            pass

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip().splitlines()[0]
        util, used, total, temp, power, clock = [float(x) if x.strip() not in ("[N/A]", "") else None for x in out.split(",")]
        return {"gpu_util": util, "vram_used_gb": used / 1024, "vram_total_gb": total / 1024,
                "temp_c": temp, "power_w": power, "sm_clock_mhz": clock}
    except Exception:
        pass

    if device_utils.accelerator() == "mps":
        try:
            import psutil
            total = psutil.virtual_memory().total / 1024 ** 3
            used = device_utils.allocated_gb() or 0.0
            return {"gpu_util": None, "vram_used_gb": used, "vram_total_gb": total,
                    "temp_c": None, "power_w": None, "sm_clock_mhz": None}
        except Exception:
            pass
    return {}


class LiveMonitor:
    def __init__(self, output_dir, task, config=None, tokenizer=None, sample_prompt=None, chat=True,
                 sample_every=50, sample_tokens=80, before_generate=None, after_generate=None):
        self.dir = os.path.join(output_dir, "live")
        os.makedirs(self.dir, exist_ok=True)
        self.tokenizer = tokenizer
        self.sample_prompt = sample_prompt
        self.chat = chat
        self.sample_every = sample_every
        self.sample_tokens = sample_tokens
        self.before_generate = before_generate
        self.after_generate = after_generate
        self.t_start = time.time()
        self.t_train = None
        self._last_write = 0.0
        # Fresh run -> clear old curves (resumed runs keep appending).
        if not (config or {}).get("resume"):
            for f in ("metrics.jsonl", "samples.jsonl"):
                with contextlib.suppress(FileNotFoundError):
                    os.remove(os.path.join(self.dir, f))
        self.status = {
            "task": task, "pid": os.getpid(), "phase": "Starting", "state": "running",
            "started_at": self.t_start, "config": {k: v for k, v in (config or {}).items() if _jsonable(v)},
            "step": 0, "max_steps": 0, "epoch": 0.0, "latest": {}, "best_eval_loss": None,
            "checkpoints": [], "error": None,
        }
        self._write_status(force=True)

    # ---------- public helpers ----------
    def phase(self, name, **extra):
        print(f"--> [{name}]", flush=True)
        self.status["phase"] = name
        self.status.update(extra)
        self._write_status(force=True)

    def callback(self):
        return _LiveCallback(self)

    @contextlib.contextmanager
    def guard(self):
        """Wrap trainer.train() so crashes / Ctrl+C show up in the dashboard."""
        try:
            yield
        except KeyboardInterrupt:
            self.status.update(state="stopped", phase="Stopped by user")
            self._write_status(force=True)
            raise
        except Exception as e:
            self.status.update(state="error", phase="Crashed", error=f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
            self._write_status(force=True)
            raise

    def finish(self, final_dir=None, **extra):
        self.status.update(state="done", phase="Finished", final_dir=final_dir, finished_at=time.time(), **extra)
        self._write_status(force=True)

    # ---------- internals ----------
    def _append(self, fname, row):
        with open(os.path.join(self.dir, fname), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _write_status(self, force=False):
        now = time.time()
        if not force and now - self._last_write < 1.0:
            return
        self._last_write = now
        self.status["updated_at"] = now
        self.status["elapsed_s"] = now - self.t_start
        tmp = os.path.join(self.dir, "status.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.status, f)
        # os.replace can briefly fail on Windows if the dashboard is reading the file.
        for _ in range(5):
            try:
                os.replace(tmp, os.path.join(self.dir, "status.json"))
                break
            except PermissionError:
                time.sleep(0.05)

    def generate_sample(self, model, step):
        if not (self.sample_prompt and self.tokenizer):
            return
        import torch
        was_training = model.training
        try:
            if self.before_generate:
                self.before_generate(model)
            model.eval()
            if self.chat and getattr(self.tokenizer, "chat_template", None):
                inputs = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": self.sample_prompt}], add_generation_prompt=True,
                    return_tensors="pt", return_dict=True)
            else:
                inputs = self.tokenizer(self.sample_prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad(), device_utils.autocast():
                out = model.generate(**inputs, max_new_tokens=self.sample_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                                     pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            cleaned = text.strip()
            if not cleaned:
                raw_tokens = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False).strip()
                cleaned = f"(model produced only whitespace/EOS: {repr(raw_tokens)[:60]})" if raw_tokens else "(model emitted immediate end-of-text)"
            self._append("samples.jsonl", {"step": step, "time": time.time(), "prompt": self.sample_prompt, "output": cleaned})
        except Exception as e:
            self._append("samples.jsonl", {"step": step, "time": time.time(), "prompt": self.sample_prompt,
                                           "output": f"(sample generation failed: {type(e).__name__}: {e})"})
        finally:
            if self.after_generate:
                self.after_generate(model)
            model.train(was_training)


class _LiveCallback(TrainerCallback):
    def __init__(self, mon):
        self.m = mon
        self.step_times = []
        from gpu_guard import ThermalGuard

        def paused(t):
            print(f"--> GPU at {t:.0f} °C - pausing to cool down (hot GPUs can compute wrong results)", flush=True)
            mon.status.update(phase=f"Cooling GPU ({t:.0f} °C)", cooling=True)
            mon._write_status(force=True)

        def resumed(t, secs):
            mon.status.update(phase="Training", cooling=False, cool_pauses=self.guard.pauses,
                              cool_paused_s=self.guard.paused_s)
            mon._write_status(force=True)
            self.step_times = []  # don't let the pause distort speed / ETA

        self.guard = ThermalGuard(on_pause=paused, on_resume=resumed)
        mon.status["max_gpu_temp"] = self.guard.max_temp

    def on_train_begin(self, args, state, control, **kw):
        m = self.m
        m.t_train = time.time()
        m.status.update(phase="Training", max_steps=state.max_steps, num_epochs=args.num_train_epochs,
                        start_step=state.global_step,
                        effective_batch=args.per_device_train_batch_size * args.gradient_accumulation_steps)
        m._write_status(force=True)
        if m.sample_every and state.global_step == 0:
            model = kw.get("model")
            if model is not None:
                m.phase("Generating baseline sample")
                m.generate_sample(model, 0)
                m.phase("Training")

    def on_step_end(self, args, state, control, **kw):
        m = self.m
        self.guard.maybe_cool()
        now = time.time()
        self.step_times.append(now)
        self.step_times = self.step_times[-50:]
        done = state.global_step - m.status.get("start_step", 0)
        sps = (len(self.step_times) - 1) / (self.step_times[-1] - self.step_times[0]) if len(self.step_times) > 1 else None
        remaining = state.max_steps - state.global_step
        g = self.guard
        m.status.update(step=state.global_step, epoch=state.epoch or 0.0, steps_per_s=sps,
                        eta_s=remaining / sps if sps else None, steps_done=done,
                        gpu_temp=g.last_temp, rest_pct=round(g.rest * 100), rested_s=round(g.rested_s))
        m._write_status()
        if m.sample_every and state.global_step % m.sample_every == 0 and kw.get("model") is not None:
            m.phase("Generating sample")
            m.generate_sample(kw["model"], state.global_step)
            m.phase("Training")

    def on_log(self, args, state, control, logs=None, **kw):
        import torch
        m = self.m
        row = {"step": state.global_step, "epoch": state.epoch, "time": time.time() - m.t_start}
        row.update({k: v for k, v in (logs or {}).items() if isinstance(v, (int, float))})
        if torch.cuda.is_available():
            row["torch_vram_gb"] = torch.cuda.memory_allocated() / 1024**3
            row["torch_peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        elif device_utils.accelerator() == "mps":
            used = device_utils.allocated_gb()
            if used is not None:
                row["torch_vram_gb"] = used
                row["torch_peak_vram_gb"] = used
        # Tokens/sec from TRL's cumulative token counter.
        if "num_tokens" in row and m.t_train:
            row["tokens_per_s"] = row["num_tokens"] / max(time.time() - m.t_train, 1e-6)
        loss = row.get("loss")
        if isinstance(loss, float) and not math.isfinite(loss):
            # The numbers blew up - on a hot laptop GPU that's usually wrong arithmetic, not a real learning problem.
            # Stop now instead of training on garbage; resilient_run rolls back to an earlier checkpoint and carries on.
            m.status.update(phase="Loss became NaN - stopping")
            m._write_status(force=True)
            raise RuntimeError(f"Training loss is nan at step {state.global_step} (the GPU computed invalid numbers).")
        m._append("metrics.jsonl", row)
        m.status["latest"].update({k: v for k, v in row.items() if k not in ("time",)})
        if "eval_loss" in row:
            best = m.status.get("best_eval_loss")
            m.status["best_eval_loss"] = row["eval_loss"] if best is None else min(best, row["eval_loss"])
        m._write_status(force=True)

    def on_save(self, args, state, control, **kw):
        self.m.status["checkpoints"].append({"step": state.global_step, "time": time.time()})
        self.m._write_status(force=True)

    def on_train_end(self, args, state, control, **kw):
        self.m.phase("Saving model")


def _jsonable(v):
    try:
        json.dumps(v)
        return True
    except TypeError:
        return False
