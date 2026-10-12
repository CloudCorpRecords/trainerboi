"""
train_embeddings.py - GPU-Accelerated Domain Embedding / RAG Model Fine-Tuning.
Fine-tunes sentence-transformer models (like all-MiniLM-L6-v2) for specialized semantic search and retrieval.
"""

import os
import time
import json
import torch
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
import trainer_utils

def train_embedding_model(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    dataset_path=None,
    output_dir="runs/embeddings/final",
    epochs=2,
    batch_size=16,
    learning_rate=2e-5
):
    trainer_utils.clear_gpu_memory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("EMBEDDING & RAG MODEL FINE-TUNING PIPELINE")
    print("=" * 65)
    print(f"--> Base Model       : {model_id}")
    print(f"--> Device           : {device} ({torch.cuda.get_device_name(0)})")
    
    # Sample domain query-passage dataset if none provided
    if dataset_path and os.path.exists(dataset_path):
        print(f"--> Loading dataset from: {dataset_path}")
        records = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        train_data = Dataset.from_list(records)
    else:
        print("--> Using built-in technical domain RAG dataset (50 query-document pairs)...")
        sample_pairs = [
            {"anchor": "How to optimize PyTorch on RTX 3080 Ti?", "positive": "Enable TF32 tensor cores, use BF16 mixed precision, and set expandable_segments in the CUDA allocator."},
            {"anchor": "What is QLoRA?", "positive": "QLoRA combines 4-bit NormalFloat quantization with low-rank adapters to train large models on consumer GPUs."},
            {"anchor": "How does vector search work?", "positive": "Vector search converts text into dense embeddings and calculates cosine similarity in high-dimensional space."},
            {"anchor": "What is channels-last memory format?", "positive": "Channels-last layout (NHWC) aligns image tensors with GPU tensor core hardware units to double convolution throughput."},
            {"anchor": "Why use gradient checkpointing?", "positive": "Gradient checkpointing recomputes intermediate activations during backward pass to reduce peak VRAM by up to 75%."},
            {"anchor": "What is semantic caching in LLMs?", "positive": "Semantic caching uses embedding distance to serve cached responses for semantically identical questions."},
            {"anchor": "How to prevent OOM on Windows?", "positive": "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to prevent memory pool fragmentation."},
            {"anchor": "What does learning rate warmup do?", "positive": "Warmup gradually ramps up the learning rate from zero to prevent chaotic parameter updates in early training."}
        ] * 8
        train_data = Dataset.from_list(sample_pairs)
        
    print(f"--> Dataset size: {len(train_data)} training pairs")
    
    # Load base model
    print(f"--> Loading {model_id}...")
    model = SentenceTransformer(model_id, device=device)
    loss = MultipleNegativesRankingLoss(model)
    
    os.makedirs(output_dir, exist_ok=True)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        save_strategy="no",
        report_to="none"
    )
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        loss=loss
    )
    
    print("--> Starting GPU embedding fine-tuning loop...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    
    print("-" * 65)
    print(f"--> Embedding model trained in {elapsed:.2f} seconds!")
    model.save_pretrained(output_dir)
    print(f"--> Model successfully saved to: {output_dir}")
    print("=" * 65)
    return output_dir

def compute_similarity(model_path_or_id, text_a, text_b):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path_or_id, device=device)
    emb_a = model.encode(text_a, convert_to_tensor=True)
    emb_b = model.encode(text_b, convert_to_tensor=True)
    score = torch.nn.functional.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
    return round(score, 4)

if __name__ == "__main__":
    train_embedding_model(epochs=1)
