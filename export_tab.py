"""
export_tab.py - Model Export, Standalone Weights Merging & Ollama Deployment Studio.
Allows users to merge LoRA adapters into standalone models and export to Ollama / GGUF in 1 click.
"""

import os
import glob
import json
import gradio as gr
import merge_lora
import trainer_utils

def find_available_adapters():
    adapters = []
    if os.path.exists("output_qlora/final_adapter"):
        adapters.append("output_qlora/final_adapter")
        
    for p in glob.glob("runs/**/adapter_config.json", recursive=True):
        parent = os.path.dirname(p)
        if parent not in adapters:
            adapters.append(parent)
            
    return adapters if adapters else ["(No adapters found yet - train one first!)"]

def execute_merge(adapter_path, export_name, quant_choice):
    if not adapter_path or "(No adapters" in adapter_path:
        return "❌ Please select a valid adapter path first.", None, None
        
    cfg_file = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(cfg_file):
        return f"❌ Missing adapter_config.json in {adapter_path}", None, None
        
    trainer_utils.clear_gpu_memory()
    out_dir = os.path.join("runs", "exported", export_name or "my_merged_model")
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        # 1. Merge into standalone weights
        merge_lora.merge(adapter_path, out_dir)
        
        abs_path = os.path.abspath(out_dir).replace("\\", "/")
        # 2. Generate Ollama Modelfile
        modelfile_content = f"""# Ollama Modelfile for {export_name}
FROM {abs_path}

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
        modelfile_path = os.path.join(out_dir, "Modelfile")
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)
            
        # 3. Generate 1-click batch script for Ollama
        run_bat_content = f"""@echo off
echo Importing {export_name} into Ollama...
ollama create {export_name} -f "%~dp0Modelfile"
if errorlevel 1 (
    echo Error importing into Ollama. Make sure Ollama is installed and running.
    pause
    exit /b 1
)
echo.
echo Model successfully imported! Starting chat...
ollama run {export_name}
pause
"""
        bat_path = os.path.join(out_dir, "run_in_ollama.bat")
        with open(bat_path, "w", encoding="utf-8") as f:
            f.write(run_bat_content)
            
        success_msg = f"""### ✅ Model Successfully Exported!
- **Standalone 16-Bit Weights:** `{os.path.abspath(out_dir)}`
- **Ollama Modelfile Generated:** `{os.path.abspath(modelfile_path)}`
- **1-Click Ollama Launcher:** `{os.path.abspath(bat_path)}`

#### How to use with Ollama:
Double-click `{bat_path}` or run in terminal:
```powershell
ollama create {export_name} -f "{modelfile_path}"
ollama run {export_name}
```
"""
        return success_msg, modelfile_content, bat_path
    except Exception as e:
        return f"❌ Error during model export: {str(e)}", None, None

def build_tab():
    with gr.TabItem("📦 Export", id="export"):
        gr.Markdown("### 📦 Export, Merge & Deploy Your Trained Models")
        gr.Markdown("Turn fine-tuned LoRA adapters into standalone 16-bit models and generate 1-click **Ollama / LM Studio** packages.")
        
        with gr.Row():
            with gr.Column(scale=3):
                adapter_choices = find_available_adapters()
                adapter_dropdown = gr.Dropdown(
                    choices=adapter_choices,
                    value=adapter_choices[0] if adapter_choices else None,
                    label="Select Trained LoRA Adapter",
                    allow_custom_value=True
                )
                refresh_adapters_btn = gr.Button("🔄 Refresh Adapter List", size="sm")
                
                export_name_input = gr.Textbox(
                    label="Export Model Name (e.g. for Ollama)",
                    value="my-custom-model"
                )
                
                export_format = gr.Radio(
                    choices=["Standalone 16-Bit (Safetensors) + Ollama Modelfile", "Standalone Weights Only"],
                    value="Standalone 16-Bit (Safetensors) + Ollama Modelfile",
                    label="Export Target"
                )
                
                merge_btn = gr.Button("🚀 Merge & Export Model", variant="primary")
                
            with gr.Column(scale=3):
                status_box = gr.Markdown("ℹ️ Select an adapter on the left and click **'Merge & Export Model'** to create deployment packages.")
                modelfile_preview = gr.Code(label="Generated Ollama Modelfile", language="dockerfile", lines=8)
                download_bat = gr.File(label="Download 1-Click Ollama Launcher (.bat)")
                
        refresh_adapters_btn.click(
            lambda: gr.update(choices=find_available_adapters()),
            outputs=[adapter_dropdown]
        )
        
        merge_btn.click(
            execute_merge,
            inputs=[adapter_dropdown, export_name_input, export_format],
            outputs=[status_box, modelfile_preview, download_bat]
        )
        
    return {
        "adapter_dropdown": adapter_dropdown,
        "merge_btn": merge_btn
    }
