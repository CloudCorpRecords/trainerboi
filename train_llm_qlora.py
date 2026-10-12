"""
train_llm_qlora.py - End-to-end 4-bit QLoRA fine-tuning for Large Language Models.
"""

import os
import time
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import trainer_utils

def train_qlora(
    model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    output_dir="./output_qlora",
    epochs=1,
    batch_size=2,
    learning_rate=2e-4,
    lora_r=16,
    lora_alpha=32
):
    trainer_utils.clear_gpu_memory()
    print("=" * 65)
    print("QLoRA 4-BIT LLM FINE-TUNING PIPELINE")
    print("=" * 65)
    print(f"--> Base Model       : {model_id}")
    print(f"--> LoRA Rank (r)    : {lora_r}")
    print(f"--> LoRA Alpha       : {lora_alpha}")
    print(f"--> Device           : cuda ({torch.cuda.get_device_name(0)})")
    
    # 1. 4-bit NormalFloat4 Quantization Configuration
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"--> Compute Dtype    : {compute_dtype}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype
    )
    
    # 2. Load Tokenizer & Quantized Model
    print(f"--> Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"--> Loading model in 4-bit NF4 quantization with native SDPA...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    # 3. Configure LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )
    model = get_peft_model(model, peft_config)
    print("\n--> Trainable Parameters:")
    model.print_trainable_parameters()
    
    # 4. Prepare Sample Instruction Dataset
    sample_texts = [
        "Instruction: Explain quantum computing.\nResponse: Quantum computing uses qubits that exist in superpositions.",
        "Instruction: Write a Python function for factorial.\nResponse: def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)",
        "Instruction: What is machine learning?\nResponse: Machine learning is an approach where computers learn patterns from data.",
        "Instruction: Why use LoRA?\nResponse: LoRA injects low-rank adapter matrices to train billions of parameters with tiny VRAM.",
        "Instruction: What is CUDA?\nResponse: CUDA is NVIDIA's parallel computing platform and programming model for GPUs.",
        "Instruction: How does gradient descent work?\nResponse: It iteratively updates parameters in the direction of the negative gradient."
    ] * 10  # 60 samples for demonstration
    
    def tokenize_func(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")
        
    dataset = Dataset.from_dict({"text": sample_texts})
    tokenized_dataset = dataset.map(tokenize_func, batched=True)
    
    # 5. High-Efficiency Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        warmup_steps=5
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    print("\n--> Commencing QLoRA training loop...")
    t0 = time.time()
    train_res = trainer.train()
    elapsed = time.time() - t0
    
    print("-" * 65)
    print(f"--> Training completed in {elapsed:.2f} seconds!")
    print(f"--> Final Training Loss: {train_res.training_loss:.4f}")
    
    # Save adapter
    adapter_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"--> LoRA Adapter successfully saved to: {adapter_path}")
    print("=" * 65)
    return adapter_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA LLM fine-tuner")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./output_qlora")
    args = parser.parse_args()
    
    train_qlora(model_id=args.model, output_dir=args.output_dir, epochs=args.epochs)
