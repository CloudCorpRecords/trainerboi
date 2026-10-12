"""
guide_tab.py - Beginner's Guide & Cheatsheet for TrainerBoi.
Plain-English explanations, VRAM sizing charts, and hyperparameter recommendations for new users.
"""

import gradio as gr
import trainer_utils

def calculate_recommendations(dataset_size, goal_type):
    size = int(dataset_size)
    if goal_type == "Chatbot / Persona":
        model = "Qwen/Qwen2.5-1.5B-Instruct or SmolLM2-135M (Fast)"
        epochs = 3 if size < 1000 else 2
        lr = "2e-4"
        lora_r = 16
        time_est = "3 to 10 minutes"
    elif goal_type == "Coding Assistant":
        model = "Qwen/Qwen2.5-3B-Instruct or Qwen2.5-7B"
        epochs = 2
        lr = "1e-4"
        lora_r = 32
        time_est = "10 to 25 minutes"
    elif goal_type == "Document Q&A / Knowledge":
        model = "Qwen/Qwen2.5-1.5B-Instruct or Llama-3.2-3B"
        epochs = 3
        lr = "2e-4"
        lora_r = 16
        time_est = "5 to 15 minutes"
    else:
        model = "Qwen/Qwen2.5-0.5B-Instruct (Lightweight)"
        epochs = 2
        lr = "2e-4"
        lora_r = 16
        time_est = "2 to 5 minutes"
        
    return f"""### 🎯 Recommended Setup for Your Project:
- **Base Model:** `{model}`
- **Training Epochs:** `{epochs}`
- **Learning Rate:** `{lr}`
- **LoRA Rank (r):** `{lora_r}` (Alpha = `{lora_r * 2}`)
- **Estimated Training Time:** ~{time_est} on a modern GPU
- **Quantization:** 4-bit NF4 (Recommended)
"""

def _specs_md():
    """Your computer's real specs (read from the hardware, not typed in)."""
    import stats_manager
    import psutil
    hw = stats_manager.hw_identity()
    gpu = hw.get("gpu") or "No NVIDIA GPU detected"
    vram = f" (**{hw['vram_gb']:.0f} GB VRAM**)" if hw.get("vram_gb") else ""
    ram = psutil.virtual_memory().total / 1024 ** 3
    return f"""### 🖥️ Your Computer's Specs & Capacities
- **GPU:** {gpu}{vram}
- **CPU:** {hw.get('cpu')} ({hw.get('cores')} cores / {hw.get('threads')} threads)
- **System memory:** {ram:.0f} GB RAM
- **Rule of thumb:** with {hw['vram_gb']:.0f} GB of VRAM, QLoRA fits roughly {max(1, hw['vram_gb'] / 2):.0f}B-parameter models comfortably.

---
""" if hw.get("vram_gb") else """### 🖥️ Your Computer's Specs
No NVIDIA GPU was detected. Training needs one; the app can still be explored.

---
"""


def build_tab():
    with gr.TabItem("📖 Beginner's Guide & Cheatsheet", id="guide"):
        gr.Markdown("## 🎓 Machine Learning Made Simple: A Beginner's Guide")
        gr.Markdown("Everything you need to know to train, fine-tune, and understand AI models without needing a PhD in math.")
        
        with gr.Tabs():
            # SUBTAB 1: CHEATSHEET & VRAM
            with gr.TabItem("⚡ Hardware & VRAM Guide"):
                gr.Markdown(_specs_md())
                gr.Markdown("""
### 📊 Model Size vs. VRAM Table (what fits on a 16 GB GPU - scale for yours)

| Model Class | Parameter Count | VRAM (4-bit QLoRA) | VRAM (16-bit Full) | Training Speed | Recommended Use Case |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SmolLM2 / TinyLlama** | 135M – 1.1B | **~1.5 GB** | ~3.5 GB | ⚡ Blazing (< 2 mins) | Quick experiments, fast CPU/edge deployment |
| **Qwen 2.5 0.5B / 1.5B** | 0.5B – 1.5B | **~2.5 GB** | ~5.0 GB | ⚡ Very Fast (~3-5 mins) | Highly capable chatbots, instruction tuning |
| **Llama 3.2 3B / Qwen 3B** | 3.0B – 3.2B | **~5.5 GB** | ~11.0 GB | 🚀 Fast (~8-15 mins) | Outstanding reasoning & coding balance |
| **Llama 3.1 8B / Qwen 7B** | 7.0B – 8.0B | **~9.5 GB** | *Exceeds 16GB* | ⏱️ Moderate (~20-45 mins) | Production-grade reasoning, deep knowledge |
| **Qwen 2.5 14B** | 14.0B | **~14.5 GB** | *Exceeds 16GB* | ⏱️ Heavy (~1-2 hours) | Complex domain tasks (small batch size) |

> 💡 **Tip:** On 16 GB of VRAM, 1.5B to 3B models offer the **sweet spot** between training speed and high intelligence on a 16 GB card!
                """)
                
            # SUBTAB 2: INTERACTIVE ADVISOR
            with gr.TabItem("🎯 Interactive Hyperparameter Advisor"):
                gr.Markdown("Not sure what settings to choose? Tell us about your goal and dataset, and we'll calculate the best setup.")
                with gr.Row():
                    with gr.Column():
                        goal_input = gr.Dropdown(
                            choices=["Chatbot / Persona", "Coding Assistant", "Document Q&A / Knowledge", "Fast Test / Quick Experiment"],
                            value="Chatbot / Persona",
                            label="What are you training?"
                        )
                        size_input = gr.Slider(
                            minimum=50,
                            maximum=50000,
                            step=50,
                            value=500,
                            label="Approximate Number of Examples in Your Dataset"
                        )
                        calc_btn = gr.Button("✨ Calculate Recommended Settings", variant="primary")
                    with gr.Column():
                        recommendation_output = gr.Markdown(calculate_recommendations(500, "Chatbot / Persona"))
                        
                calc_btn.click(calculate_recommendations, inputs=[size_input, goal_input], outputs=[recommendation_output])
                
            # SUBTAB 3: CORE CONCEPTS
            with gr.TabItem("🧠 Core Concepts Explained"):
                gr.Markdown("""
### 1. What is Fine-Tuning?
Think of base AI models like graduates who already know how to read, write, and think. **Fine-tuning** doesn't teach them language from scratch; it teaches them a **specific style, persona, domain, or company knowledge**.

### 2. What is LoRA & QLoRA?
- **Standard Fine-Tuning**: Modifies all billions of parameters in the model. This requires 4x more VRAM and is very slow.
- **LoRA (Low-Rank Adaptation)**: Freezes 99% of the model and adds small, trainable "adapter layers".
- **QLoRA (Quantized LoRA)**: Compresses the base model weights into **4-bit numbers** so they fit effortlessly in your GPU VRAM, while training the adapters with full precision!

### 3. What is an Epoch?
One **epoch** means the model has seen every single example in your training dataset once.
- **Too few epochs (e.g. 0.5)**: Model underfits and doesn't learn your pattern.
- **Too many epochs (e.g. 20+)**: Model overfits and memorizes your examples word-for-word instead of generalizing.
- **Recommended**: Usually **2 to 3 epochs** is ideal.

### 4. What is Learning Rate?
The **learning rate** determines how large a step the optimizer takes when updating weights.
- Typical LLM fine-tuning learning rate: `2e-4` (0.0002) for LoRA.

### 5. What is Loss?
**Loss** is a number measuring how wrong the model's predictions are.
- As the model trains, you want the loss number to go **down**!
- If loss drops steadily from ~3.0 to ~1.2, your model is learning well.
                """)
                
            # SUBTAB 4: DATA FORMATTING
            with gr.TabItem("📁 How to Format Your Data"):
                gr.Markdown("""
### Format 1: Multi-Turn Conversation (Recommended for Chatbots)
File format: `.jsonl` (one JSON object per line)
```json
{"messages": [{"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Click 'Forgot Password' on the login screen and follow the email link."}]}
{"messages": [{"role": "user", "content": "What are your business hours?"}, {"role": "assistant", "content": "We are open Monday through Friday from 9 AM to 6 PM EST."}]}
```

### Format 2: Instruction & Response
```json
{"instruction": "Translate the following English phrase to French: Good morning", "response": "Bonjour"}
{"instruction": "Summarize the key takeaway of photosynthesis.", "response": "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen."}
```

### Format 3: Tabular Data (CSV)
For tabular models (XGBoost):
```csv
age,tenure_months,monthly_spend,support_calls,churn
34,12,79.50,1,0
58,3,110.20,4,1
29,24,45.00,0,0
```
                """)
