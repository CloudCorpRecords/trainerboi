"""
train_vision.py - GPU-accelerated computer vision training with timm and PyTorch AMP.
Runs on CUDA, Apple MPS, or CPU.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import timm
import device_utils
import trainer_utils

def train_vision_model(model_name="resnet18", num_classes=5, epochs=3, batch_size=64, img_size=224):
    trainer_utils.clear_gpu_memory()
    acc = device_utils.accelerator()
    device = torch.device(acc)
    print("=" * 60)
    print("COMPUTER VISION TRAINING (timm + PyTorch AMP)")
    print("=" * 60)
    print(f"--> Target Model Architecture: {model_name}")
    print(f"--> Target Device            : {device} ({device_utils.device_name()})")
    print(f"--> Classes                  : {num_classes}")

    # Instantiate timm model with pretrained weights head replaced
    print(f"--> Instantiating '{model_name}' with {num_classes} output classes...")
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model = model.to(device, memory_format=torch.channels_last)

    # Create sample image dataset (3 channels, img_size x img_size)
    n_samples = 2000
    print(f"--> Generating synthetic image dataset ({n_samples} images, 3x{img_size}x{img_size})...")
    X = torch.randn(n_samples, 3, img_size, img_size)
    y = torch.randint(0, num_classes, (n_samples,))

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        **trainer_utils.get_optimal_dataloader_kwargs())

    use_amp = acc == "cuda"
    use_bf16 = use_amp and device_utils.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if use_amp:
        print(f"--> Using high-efficiency {amp_dtype} with Channels-Last memory format...")
    else:
        print(f"--> Using full precision on {device_utils.device_label()}...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda" if use_amp else "cpu", enabled=(use_amp and not use_bf16))
    
    print(f"--> Commencing training for {epochs} epochs...")
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        ep_start = time.time()
        
        for imgs, labels in loader:
            imgs = imgs.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            if not use_amp or use_bf16:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        scheduler.step()
        ep_duration = time.time() - ep_start
        used = device_utils.allocated_gb()
        mem_txt = f"{used * 1024:.1f} MB" if used is not None else "n/a"
        print(f"    Epoch {epoch}/{epochs} | Loss: {running_loss/total:.4f} | Acc: {(correct/total)*100:.2f}% | Mem: {mem_txt} | Time: {ep_duration:.2f}s")
        
    print("-" * 60)
    print(f"--> Vision model training completed in {time.time() - t0:.2f} seconds!")
    
    out_path = f"model_vision_{model_name}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"--> Saved model weights to: {out_path}")
    print("=" * 60)
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train vision model")
    parser.add_argument("--model", type=str, default="resnet18", help="timm model architecture name")
    parser.add_argument("--classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    
    train_vision_model(args.model, args.classes, args.epochs, args.batch_size)
