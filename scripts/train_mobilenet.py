import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# Add root folder to sys.path so modules can be imported directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.dataset import get_dataloaders
from models.mobilenet_v2 import get_mobilenet_v2


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


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
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def run_mobilenet_training(
    warmup_epochs: int = 4,
    finetune_epochs: int = 8,
    batch_size: int = 32,
    save_path: str = "models/mobilenet_v2_best.pth"
):
    device = get_device()
    print(f"Training on device: {device}")

    train_loader, test_loader, classes = get_dataloaders(
        data_dir="data",
        batch_size=batch_size,
        num_workers=2,
        img_size=224
    )

    # Initialize with frozen feature backbone
    model = get_mobilenet_v2(num_classes=len(classes), freeze_features=True).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Phase 1: Train classification head only
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0
    Path("models").mkdir(exist_ok=True)

    print("\n--- Phase 1: Training Classification Head (Feature Extractor Frozen) ---")
    for epoch in range(1, warmup_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        duration = time.time() - t0

        print(
            f"Warmup [{epoch:02d}/{warmup_epochs:02d}] "
            f"| Train Loss: {train_loss:.4f} - Acc: {train_acc * 100:.2f}% "
            f"| Test Loss: {val_loss:.4f} - Acc: {val_acc * 100:.2f}% "
            f"| Time: {duration:.1f}s"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    # Phase 2: Unfreeze top bottleneck layers for fine-tuning
    print("\n--- Phase 2: Unfreezing Top Feature Layers for Fine-Tuning ---")
    for param in model.features[-4:].parameters():
        param.requires_grad = True

    # Differential learning rate: lower LR on features, moderate on head
    optimizer = optim.Adam([
        {"params": model.features[-4:].parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 3e-4}
    ], weight_decay=1e-4)

    for epoch in range(1, finetune_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        duration = time.time() - t0

        print(
            f"Finetune [{epoch:02d}/{finetune_epochs:02d}] "
            f"| Train Loss: {train_loss:.4f} - Acc: {train_acc * 100:.2f}% "
            f"| Test Loss: {val_loss:.4f} - Acc: {val_acc * 100:.2f}% "
            f"| Time: {duration:.1f}s"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best MobileNetV2 ({val_acc * 100:.2f}%) to {save_path}")

    print(f"\nMobileNetV2 training completed. Best Test Accuracy: {best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    run_mobilenet_training()