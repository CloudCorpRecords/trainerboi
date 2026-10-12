"""
embedding_studio_tab.py - Domain Embeddings & RAG Fine-Tuning Studio for TrainerBoi.
Fine-tunes vector search models for private enterprise RAG pipelines with live similarity scoring.
"""

import os
import gradio as gr
import train_embeddings

def handle_train_embeddings(model_id, dataset_file, epochs, lr, batch_size):
    try:
        ds_path = dataset_file.name if dataset_file is not None else None
        out_dir = train_embeddings.train_embedding_model(
            model_id=model_id,
            dataset_path=ds_path,
            epochs=int(epochs),
            learning_rate=float(lr),
            batch_size=int(batch_size)
        )
        return f"✅ Embedding model fine-tuning complete!\nModel saved to: `{out_dir}`\nReady for Chroma, Qdrant, LanceDB, or FAISS."
    except Exception as e:
        return f"❌ Error during embedding training: {str(e)}"

def test_similarity(model_choice, text_a, text_b):
    if not text_a or not text_b:
        return "Please enter both texts."
    try:
        score = train_embeddings.compute_similarity(model_choice, text_a, text_b)
        pct = round(score * 100, 1)
        return f"### Semantic Similarity: **{score}** ({pct}% match)"
    except Exception as e:
        return f"Error: {str(e)}"

def build_tab():
    with gr.TabItem("🔍 RAG", id="rag"):
        gr.Markdown("### 🔍 Vector Search & Embedding Model Fine-Tuning (RAG)")
        gr.Markdown("Train specialized text embedding models for private search, document retrieval, and vector databases.")
        
        with gr.Row():
            with gr.Column(scale=3):
                model_dropdown = gr.Dropdown(
                    choices=[
                        "sentence-transformers/all-MiniLM-L6-v2",
                        "BAAI/bge-small-en-v1.5",
                        "sentence-transformers/all-mpnet-base-v2"
                    ],
                    value="sentence-transformers/all-MiniLM-L6-v2",
                    label="Base Embedding Model"
                )
                dataset_upload = gr.File(
                    label="Upload Query-Positive JSONL Dataset (Leave empty to use built-in technical RAG dataset)",
                    file_types=[".jsonl", ".json"]
                )
                with gr.Row():
                    emb_epochs = gr.Slider(minimum=1, maximum=5, step=1, value=2, label="Epochs")
                    emb_lr = gr.Dropdown(choices=["2e-5", "5e-5", "1e-4"], value="2e-5", label="Learning Rate")
                    emb_batch = gr.Dropdown(choices=[8, 16, 32], value=16, label="Batch Size")
                train_emb_btn = gr.Button("🚀 Fine-Tune Embedding Model on GPU", variant="primary")
                status_box = gr.Markdown("ℹ️ Configure settings above and click **Fine-Tune Embedding Model**.")
                
            with gr.Column(scale=3):
                gr.Markdown("#### 🧪 Interactive Semantic Similarity Playground")
                gr.Markdown("Test how closely your fine-tuned model relates two sentences in vector space:")
                eval_model = gr.Dropdown(
                    choices=[
                        "runs/embeddings/final",
                        "sentence-transformers/all-MiniLM-L6-v2"
                    ],
                    value="runs/embeddings/final" if os.path.exists("runs/embeddings/final") else "sentence-transformers/all-MiniLM-L6-v2",
                    label="Model to Test"
                )
                text_input_a = gr.Textbox(label="Query / Sentence A", value="How to speed up PyTorch?")
                text_input_b = gr.Textbox(label="Document / Sentence B", value="Enable TF32 and BF16 acceleration on NVIDIA Ampere.")
                calc_sim_btn = gr.Button("⚡ Calculate Semantic Similarity", variant="secondary")
                similarity_result = gr.Markdown("### Semantic Similarity: -")
                
        train_emb_btn.click(
            handle_train_embeddings,
            inputs=[model_dropdown, dataset_upload, emb_epochs, emb_lr, emb_batch],
            outputs=[status_box]
        )
        calc_sim_btn.click(
            test_similarity,
            inputs=[eval_model, text_input_a, text_input_b],
            outputs=[similarity_result]
        )
        
    return {
        "train_emb_btn": train_emb_btn
    }
