"""
classic_tab.py - the original one-click trainers, grouped on a single page: quick QLoRA fine-tune, tabular (XGBoost) and vision (timm).
"""

import gradio as gr
import train_tabular
import train_vision
import train_llm_qlora


def handle_train_tabular(csv_file, task_type, n_est, depth, lr):
    try:
        csv_path = csv_file.name if hasattr(csv_file, "name") else csv_file
        model, out_file = train_tabular.train_tabular_model(
            csv_path=csv_path, task_type=task_type, n_estimators=int(n_est), max_depth=int(depth), learning_rate=float(lr))
        return f"Successfully trained XGBoost model on GPU!\nModel saved to: {out_file}", out_file
    except Exception as e:
        return f"Error during training: {e}", None


def handle_train_vision(arch, classes, epochs, batch_size):
    try:
        out_file = train_vision.train_vision_model(model_name=arch, num_classes=int(classes), epochs=int(epochs),
                                                   batch_size=int(batch_size))
        return f"Vision model training completed!\nWeights saved to: {out_file}", out_file
    except Exception as e:
        return f"Error during vision training: {e}", None


def handle_train_qlora(model_id, lora_r, epochs):
    try:
        out_path = train_llm_qlora.train_qlora(model_id=model_id, epochs=int(epochs), lora_r=int(lora_r))
        return f"4-bit QLoRA fine-tuning completed successfully!\nLoRA adapter weights saved at: {out_path}"
    except Exception as e:
        return f"Error during QLoRA training: {e}"


def build_tab():
    with gr.TabItem("🧪 Classic trainers", id="classic"):
        gr.Markdown("Simple one-click trainers for other kinds of models. For chat models use **Guided training** or **Live training**.")
        with gr.Tabs():
            with gr.TabItem("🧠 Quick QLoRA"):
                with gr.Row():
                    with gr.Column():
                        llm_id = gr.Dropdown(["HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct",
                                              "meta-llama/Llama-3.2-1B-Instruct"], value="HuggingFaceTB/SmolLM2-135M-Instruct",
                                             label="Base model (or any Hugging Face id)", allow_custom_value=True)
                        lora_r = gr.Slider(4, 64, step=4, value=16, label="LoRA rank")
                        epochs = gr.Slider(1, 10, step=1, value=1, label="Epochs")
                        btn = gr.Button("🚀 Start QLoRA training", variant="primary")
                    status = gr.Textbox(label="Progress & output", lines=12, interactive=False)
                btn.click(handle_train_qlora, [llm_id, lora_r, epochs], status)

            with gr.TabItem("📊 Tabular (XGBoost)"):
                with gr.Row():
                    with gr.Column():
                        csv = gr.File(label="CSV dataset (leave empty for a 100k-row synthetic one)", file_types=[".csv"])
                        sample = gr.Button("✨ Load sample customer-churn data (2,000 rows)", size="sm")
                        sample.click(lambda: "data/samples/customer_churn.csv", outputs=[csv])
                        task = gr.Radio(["classification", "regression"], value="classification", label="Task")
                        n_est = gr.Slider(20, 1000, step=20, value=100, label="Trees")
                        depth = gr.Slider(3, 12, step=1, value=6, label="Max depth")
                        lr = gr.Slider(0.01, 0.5, step=0.01, value=0.1, label="Learning rate")
                        tbtn = gr.Button("⚡ Train on GPU", variant="primary")
                    with gr.Column():
                        tstatus = gr.Textbox(label="Metrics & log", lines=10, interactive=False)
                        tdl = gr.File(label="Download model (JSON)")
                tbtn.click(handle_train_tabular, [csv, task, n_est, depth, lr], [tstatus, tdl])

            with gr.TabItem("👁️ Vision (timm)"):
                with gr.Row():
                    with gr.Column():
                        arch = gr.Dropdown(["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"], value="resnet18",
                                           label="Architecture")
                        classes = gr.Slider(2, 100, step=1, value=5, label="Number of classes")
                        vepochs = gr.Slider(1, 10, step=1, value=2, label="Epochs")
                        batch = gr.Dropdown([16, 32, 64, 128], value=64, label="Batch size")
                        vbtn = gr.Button("⚡ Train on GPU", variant="primary")
                    with gr.Column():
                        vstatus = gr.Textbox(label="Progress", lines=10, interactive=False)
                        vdl = gr.File(label="Download weights (.pt)")
                vbtn.click(handle_train_vision, [arch, classes, vepochs, batch], [vstatus, vdl])
