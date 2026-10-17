"""
train_cli.py - Rich Interactive Terminal Control Center for TrainerBoi.
"""

import os
import sys
import time
import subprocess
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
import trainer_utils

console = Console()

def print_header():
    console.clear()
    telemetry = trainer_utils.get_gpu_telemetry()
    
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Property", style="dim", width=24)
    table.add_column("Current Value", style="bold green")
    
    table.add_row("Device", str(telemetry.get("device_name", "N/A")))
    cc = telemetry.get("compute_capability")
    table.add_row("Compute Capability", str(cc) + " (Ampere BF16/FP16)" if cc else "n/a (no NVIDIA GPU)")
    table.add_row("Total VRAM", f"{telemetry.get('vram_total_gb', 0)} GB")
    table.add_row("VRAM Allocated", f"{telemetry.get('vram_allocated_gb', 0)} GB")
    table.add_row("Free VRAM", f"{telemetry.get('vram_free_gb', 0)} GB")
    table.add_row("GPU Temperature", f"{telemetry.get('temperature_c', 'N/A')} °C")
    table.add_row("GPU Utilization", f"{telemetry.get('utilization_pct', 'N/A')}%")
    
    console.print(Panel(table, title="[bold cyan]🚀 TrainerBoi - GPU Studio Control Center[/bold cyan]", border_style="cyan"))

def main_menu():
    while True:
        print_header()
        console.print("[bold yellow]Choose a training or diagnostic action:[/bold yellow]\n")
        console.print("  [bold green]1.[/bold green] 🩺 Run Comprehensive Environment Diagnostic")
        console.print("  [bold green]2.[/bold green] ⚡ Run GPU Compute & TFLOPS Benchmark")
        import device_utils
        _qlora_ok = device_utils.supports_4bit()
        console.print("  [bold green]3.[/bold green] 📊 Train Tabular Model (XGBoost on 100k samples)")
        console.print("  [bold green]4.[/bold green] 👁️ Train Vision Classifier (timm ResNet)")
        console.print(f"  [bold green]5.[/bold green] 🧠 Fine-Tune LLM with {'4-Bit QLoRA' if _qlora_ok else 'LoRA'} (SmolLM2 / Llama)")
        console.print(f"  [bold green]6.[/bold green] 🦥 LLM SFT on a Real Dataset ({'Unsloth QLoRA' if _qlora_ok else 'HF LoRA'})")
        console.print("  [bold green]7.[/bold green] 🏗️ Pretrain a Tiny LLM From Scratch (TinyStories)")
        console.print("  [bold green]8.[/bold green] 💬 Chat With a Trained Model")
        console.print("  [bold green]9.[/bold green] 🌐 Launch Local Web Studio (Gradio UI on port 7860)")
        console.print("  [bold green]10.[/bold green] 🧹 Flush GPU VRAM Memory Cache")
        console.print("  [bold red]11.[/bold red] ❌ Exit\n")

        choice = Prompt.ask("Enter option", choices=[str(i) for i in range(1, 12)], default="1")
        # Map new numbering onto the original actions below.
        choice = {"9": "web", "10": "flush", "11": "exit"}.get(choice, choice)
        
        if choice == "1":
            console.print("\n[bold cyan]Running Diagnostic...[/bold cyan]")
            import check_environment
            check_environment.check_env()
            Prompt.ask("\nPress [bold]Enter[/bold] to return to menu")
            
        elif choice == "2":
            console.print("\n[bold cyan]Running Benchmark...[/bold cyan]")
            import trainer_utils
            res = trainer_utils.benchmark_gpu()
            console.print(Panel(res, title="Benchmark Results", border_style="green"))
            Prompt.ask("\nPress [bold]Enter[/bold] to return to menu")
            
        elif choice == "3":
            console.print("\n[bold cyan]Starting Tabular XGBoost Training on GPU...[/bold cyan]")
            import train_tabular
            train_tabular.train_tabular_model(n_estimators=100)
            Prompt.ask("\nPress [bold]Enter[/bold] to return to menu")
            
        elif choice == "4":
            console.print("\n[bold cyan]Starting Computer Vision Training...[/bold cyan]")
            import train_vision
            train_vision.train_vision_model(model_name="resnet18", epochs=2)
            Prompt.ask("\nPress [bold]Enter[/bold] to return to menu")
            
        elif choice == "5":
            console.print("\n[bold cyan]Starting 4-bit QLoRA LLM Fine-Tuning on GPU...[/bold cyan]")
            import train_llm_qlora
            train_llm_qlora.train_qlora(epochs=1)
            Prompt.ask("\nPress [bold]Enter[/bold] to return to menu")
            
        elif choice == "6":
            model = Prompt.ask("Base model", default="Qwen/Qwen2.5-1.5B-Instruct")
            dataset = Prompt.ask("Dataset (HF id or local .jsonl/.csv)", default="yahma/alpaca-cleaned")
            steps = Prompt.ask("Max steps", default="200")
            out = Prompt.ask("Output dir", default="runs/sft")
            subprocess.run([sys.executable, "train_llm_sft.py", "--model", model, "--dataset", dataset,
                            "--max_steps", steps, "--output_dir", out])
            Prompt.ask("\nPress [bold]Enter[/bold] to return to menu")

        elif choice == "7":
            steps = Prompt.ask("Max steps", default="3000")
            subprocess.run([sys.executable, "pretrain_llm.py", "--size", "tiny", "--max_steps", steps])
            Prompt.ask("\nPress [bold]Enter[/bold] to return to menu")

        elif choice == "8":
            model = Prompt.ask("Model / adapter path", default="runs/sft/final")
            subprocess.run([sys.executable, "chat_llm.py", "--model", model])

        elif choice == "web":
            console.print("\n[bold green]Launching TrainerBoi Web Studio on http://127.0.0.1:7860...[/bold green]")
            console.print("[dim]Press Ctrl+C in this terminal to stop the web server.[/dim]\n")
            cmd = [sys.executable, "app.py"]
            subprocess.run(cmd)
            
        elif choice == "flush":
            trainer_utils.clear_gpu_memory()
            console.print("\n[bold green]GPU VRAM cache successfully cleared![/bold green]")
            time.sleep(1)
            
        elif choice == "exit":
            console.print("\n[bold cyan]Goodbye![/bold cyan]")
            break

if __name__ == "__main__":
    main_menu()
