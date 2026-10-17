import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import device_utils

class ClassifierNet(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def run_training():
    use_amp = device_utils.accelerator() == "cuda"
    device = torch.device(device_utils.accelerator())
    print(f"--> Training on device: {device} ({device_utils.device_name()})")
    
    # 1. Generate synthetic dataset (50,000 samples, 256 features, 10 classes)
    print("--> Generating synthetic dataset (50,000 samples)...")
    torch.manual_seed(42)
    x_data = torch.randn(50000, 256)
    y_data = torch.randint(0, 10, (50000,))
    
    dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=use_amp)

    # 2. Instantiate model, loss, optimizer, and scaler for mixed precision
    model = ClassifierNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda" if use_amp else "cpu", enabled=use_amp)

    print(f"--> Starting 5-epoch training loop{' with mixed precision (AMP)' if use_amp else ''}...")
    total_start = time.time()
    
    model.train()
    for epoch in range(1, 6):
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        used = device_utils.allocated_gb()
        mem_txt = f"{used * 1024:.1f} MB" if used is not None else "n/a"
        print(f"    Epoch {epoch}/5 | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | Memory: {mem_txt} | Time: {epoch_time:.2f}s")
        
    print(f"--> Training completed successfully in {time.time() - total_start:.2f}s!")

if __name__ == "__main__":
    run_training()
