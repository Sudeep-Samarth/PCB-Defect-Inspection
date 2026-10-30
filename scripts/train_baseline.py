# Step4

 
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

#Add root folder to sys.path so modules can be imported directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.dataset import get_dataloaders
from models.baseline_cnn import DefectBaselineCNN

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss=0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss +=loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total +=labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total 
    return val_loss, val_acc

def run_training(
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = "models/baseline_cnn_best.pth"
):
    device = get_device()
    print(f"Training on device: {device}")


    # Load data
    train_loader, test_loader, classes = get_dataloaders(
        data_dir="data",
        batch_size=batch_size,
        num_workers=2,
        img_size=224
    )

    model = DefectBaselineCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc=0.0
    Path("models").mkdir(exist_ok=True)

    print("\n--- Starting Baseline CNN Training ---")
    start_time = time.time()


    for epoch in range(1, epochs+1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"| Train Loss: {train_loss:.4f} - Acc: {train_acc * 100:.2f}% "
            f"| Test Loss: {val_loss:.4f} - Acc: {val_acc * 100:.2f}% "
            f"| Time: {epoch_duration:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best baseline model ({val_acc * 100:.2f}%) to {save_path}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes.")
    print(f"Best Test Accuracy: {best_val_acc * 100:.2f}%")

if __name__ == "__main__":
    run_training(epochs=15, batch_size=32, lr=1e-3)