import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

def test_lora_setup():
    print("--> Initializing Hugging Face Transformers + PEFT LoRA test...")
    model_id = "distilbert/distilbert-base-uncased"
    
    print(f"--> Loading base model: {model_id}")
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"]
    )
    
    lora_model = get_peft_model(model, peft_config)
    lora_model.print_trainable_parameters()
    
    # Send to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_model.to(device)
    print(f"--> LoRA model successfully moved to {device} ({torch.cuda.get_device_name(0)})")
    
    # Quick forward/backward pass test
    dummy_input = torch.randint(0, 1000, (4, 32), device=device)
    dummy_labels = torch.tensor([0, 1, 0, 1], device=device)
    
    outputs = lora_model(input_ids=dummy_input, labels=dummy_labels)
    loss = outputs.loss
    loss.backward()
    print(f"--> Forward + backward pass passed! Loss: {loss.item():.4f}")
    print("--> LoRA and Hugging Face training pipeline is fully operational!")

if __name__ == "__main__":
    test_lora_setup()
