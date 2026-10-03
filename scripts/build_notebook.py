import json
import uuid
from pathlib import Path

def create_full_pipeline_notebook(output_path: str = "scripts/pcb_defect_inspection_pipeline.ipynb"):
    cells = []
    cell_counter = 0

    def add_md(source: str):
        nonlocal cell_counter
        cell_counter += 1
        lines = [line + "\n" for line in source.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cells.append({
            "cell_type": "markdown",
            "id": f"md-cell-{cell_counter:03d}",
            "metadata": {},
            "source": lines
        })

    def add_code(source: str):
        nonlocal cell_counter
        cell_counter += 1
        lines = [line + "\n" for line in source.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cells.append({
            "cell_type": "code",
            "id": f"code-cell-{cell_counter:03d}",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

    # =========================================================================
    # Header & Overview
    # =========================================================================
    add_md(r"""# Automated PCB Defect Inspection Pipeline (End-to-End)
### Deep Learning Benchmark Study on DeepPCB Dataset
**Author:** AI & Data Science Prototype Team  
**Dataset:** [DeepPCB Benchmark (Tang et al., 2019)](https://github.com/tangsanli5201/DeepPCB)

---
### 📌 Pipeline Overview
1. **Data Extraction & Bounding Box Cropping:** Parse raw DeepPCB annotation files, add 15% contextual padding, and split into train/test subsets strictly at the board level.
2. **Data Augmentation & DataLoaders:** Apply stochastic transforms (flips, rotation, normalization) and construct PyTorch DataLoaders.
3. **Custom Baseline CNN:** Implement and train a 4-block Conv2D-BatchNorm-ReLU-MaxPool CNN from scratch.
4. **Transfer Learning (MobileNetV2):** Adapt ImageNet-pretrained MobileNetV2 using a two-phase training protocol (backbone freezing warmup followed by differential fine-tuning).
5. **Model Evaluation & Benchmarking:** Generate classification reports, latency benchmarks, memory footprint comparisons, and normalized side-by-side confusion matrices.
""")

    # =========================================================================
    # Step 0: Imports and Configuration
    # =========================================================================
    add_md(r"""## Step 0: Environment Setup, Dependencies & Global Configuration""")
    add_code(r"""import os
import sys
import time
import glob
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Device Configuration
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Using Compute Device: {DEVICE}")

# Global Directory Path Auto-Discovery
def resolve_base_dir():
    # Check current working dir and parents for raw_dataset or data
    cwd = Path.cwd()
    candidates = [cwd, cwd.parent, Path("/content"), Path("/content/Prototype1")]
    for candidate in candidates:
        if (candidate / "raw_dataset").exists() or (candidate / "data").exists():
            return candidate
    return cwd

BASE_DIR = resolve_base_dir()

RAW_DATASET_DIR = BASE_DIR / "raw_dataset" / "PCBData"
if not RAW_DATASET_DIR.exists() and (BASE_DIR / "PCBData").exists():
    RAW_DATASET_DIR = BASE_DIR / "PCBData"

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# DeepPCB Class Index Mapping (1 to 6)
LABEL_MAP = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin_hole"
}

PADDING_RATIO = 0.15
RANDOM_STATE = 42
IMG_SIZE = 224
BATCH_SIZE = 32

print(f"Base Directory: {BASE_DIR}")
print(f"Raw Dataset Path: {RAW_DATASET_DIR}")
print(f"Output Data Path: {DATA_DIR}")
print(f"Target Defect Classes: {list(LABEL_MAP.values())}")
""")

    # =========================================================================
    # Step 1: Data Preprocessing & Cropping
    # =========================================================================
    add_md(r"""## Step 1: Data Preprocessing & Contextual Defect Patch Cropping
- Read bounding box coordinates: `(x1, y1, x2, y2, defect_type)` from annotation text files.
- Apply **15% contextual padding** around each bounding box.
- Perform **Board-Level Train/Test Split (80/20)** to prevent data leakage.
- Save cropped defect patches into `data/train/<class>` and `data/test/<class>`.
""")
    add_code(r"""def setup_output_dirs():
    for split in ["train", "test"]:
        for class_name in LABEL_MAP.values():
            (DATA_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

def parse_annotation_file(txt_path):
    boxes = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                x1, y1, x2, y2, defect_type = map(int, parts[:5])
                boxes.append((x1, y1, x2, y2, defect_type))
    return boxes

def crop_and_save(img, box, output_path):
    h, w = img.shape[:2]
    x1, y1, x2, y2, defect_type = box

    bw = x2 - x1
    bh = y2 - y1

    pad_w = int(bw * PADDING_RATIO)
    pad_h = int(bh * PADDING_RATIO)

    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(w, x2 + pad_w)
    ny2 = min(h, y2 + pad_h)

    crop = img[ny1:ny2, nx1:nx2]
    if crop.size > 0:
        cv2.imwrite(str(output_path), crop)

def locate_test_image(ann_path):
    board_id = ann_path.stem.replace("_not", "")
    parent_dir = ann_path.parent

    # 1. Sibling image directory
    if parent_dir.name.endswith("_not"):
        sibling_img_dir = parent_dir.parent / parent_dir.name[:-4]
        candidate = sibling_img_dir / f"{board_id}_test.jpg"
        if candidate.exists():
            return candidate

    # 2. Check same directory fallback
    candidate = parent_dir / f"{board_id}_test.jpg"
    if candidate.exists():
        return candidate

    # 3. Fuzzy search for board_id containing 'test'
    candidates = list(parent_dir.parent.glob(f"**/{board_id}_test.jpg"))
    if candidates:
        return candidates[0]

    return None

def process_dataset(test_ratio=0.2, random_state=42):
    setup_output_dirs()

    # Check if already processed
    existing_train_imgs = list((DATA_DIR / "train").glob("*/*.jpg"))
    existing_test_imgs = list((DATA_DIR / "test").glob("*/*.jpg"))
    if len(existing_train_imgs) > 0 and len(existing_test_imgs) > 0:
        print(f"Data already prepared: found {len(existing_train_imgs)} train patches and {len(existing_test_imgs)} test patches.")
        return

    ann_files = list(RAW_DATASET_DIR.glob("**/*_not/*.txt"))
    if not ann_files:
        ann_files = [f for f in RAW_DATASET_DIR.glob("**/*.txt") if "trainval" not in f.name and "test" not in f.name]

    print(f"Found {len(ann_files)} annotation board files.")
    if len(ann_files) == 0:
        print("Warning: No raw annotation files found. Please verify raw_dataset/PCBData path.")
        return

    # Split dataset at the image/board level to prevent data leakage
    train_files, test_files = train_test_split(ann_files, test_size=test_ratio, random_state=random_state)
    splits = [("train", train_files), ("test", test_files)]

    stats = {
        "train": {name: 0 for name in LABEL_MAP.values()},
        "test": {name: 0 for name in LABEL_MAP.values()}
    }

    for split_name, file_list in splits:
        print(f"Processing {split_name} split ({len(file_list)} boards)...")
        for ann_path in file_list:
            test_img_path = locate_test_image(ann_path)
            if test_img_path is None or not test_img_path.exists():
                continue

            img = cv2.imread(str(test_img_path))
            if img is None:
                continue

            boxes = parse_annotation_file(ann_path)
            for idx, box in enumerate(boxes):
                _, _, _, _, defect_type = box
                if defect_type in LABEL_MAP:
                    class_name = LABEL_MAP[defect_type]
                    out_filename = f"{test_img_path.stem}_defect_{idx}_{class_name}.jpg"
                    out_file_path = DATA_DIR / split_name / class_name / out_filename

                    crop_and_save(img, box, out_file_path)
                    stats[split_name][class_name] += 1

    print("--- Cropping Complete ---")
    for split_name in ["train", "test"]:
        print(f"{split_name.upper()} counts: ")
        for defect_name, count in stats[split_name].items():
            print(f"  {defect_name:12}: {count}")

# Execute Cropping Pipeline
process_dataset(test_ratio=0.2, random_state=RANDOM_STATE)
""")

    # =========================================================================
    # Step 2: DataLoaders & Augmentation
    # =========================================================================
    add_md(r"""## Step 2: Data Augmentation & PyTorch DataLoaders
- **Training Transformations:** Random horizontal/vertical flips, ±10° random rotations, and ImageNet standardization.
- **Testing Transformations:** Deterministic resizing and ImageNet standardization.
- **PyTorch DataLoaders:** Batch size = 32 with pinned memory.
""")
    add_code(r"""IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return train_transform, test_transform

def get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    train_dir = Path(data_dir) / "train"
    test_dir = Path(data_dir) / "test"

    train_tf, test_tf = get_transforms(img_size=img_size)

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_tf)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=test_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader, train_dataset.classes

train_loader, test_loader, class_names = get_dataloaders()
print(f"Detected Classes ({len(class_names)}): {class_names}")
print(f"Train Batches: {len(train_loader)} | Test Batches: {len(test_loader)}")

sample_imgs, sample_lbls = next(iter(train_loader))
print(f"Sample Batch Shape: Images {sample_imgs.shape}, Labels {sample_lbls.shape}")
""")

    # =========================================================================
    # Step 3: Model 1 - Custom Baseline CNN
    # =========================================================================
    add_md(r"""## Step 3: Architecture 1 — Custom Baseline CNN
- 4-block Conv2D $\rightarrow$ BatchNorm2D $\rightarrow$ ReLU $\rightarrow$ MaxPool2D feature extractor ($3 \to 32 \to 64 \to 128 \to 256$ channels).
- Spatial pooling via `AdaptiveAvgPool2d((4,4))` yielding a consistent 4,096-dimensional flattened vector.
- Classification Head: `Linear(4096, 256)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(p=0.4)` $\rightarrow$ `Linear(256, 6)`.
""")
    add_code(r"""class DefectBaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 6, dropout_p: float = 0.4):
        super(DefectBaselineCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels | 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32 -> 64 channels | 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 -> 128 channels | 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 128 -> 256 channels | 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return logits

# Smoke Test
baseline_test_model = DefectBaselineCNN(num_classes=len(class_names)).to(DEVICE)
dummy_out = baseline_test_model(sample_imgs[:2].to(DEVICE))
print("Baseline CNN initialized successfully. Output shape:", dummy_out.shape)
""")

    # =========================================================================
    # Step 4: Model 2 - Pretrained MobileNetV2
    # =========================================================================
    add_md(r"""## Step 4: Architecture 2 — Transfer Learning (MobileNetV2)
- Pretrained on **ImageNet-1K** with depthwise separable convolutions and inverted residual bottlenecks.
- Custom classification head: `Dropout(p=0.3) -> Linear(1280, 6)`.
- Implements dynamic feature freezing/unfreezing for two-phase transfer learning.
""")
    add_code(r"""def get_mobilenet_v2(num_classes: int = 6, freeze_features: bool = True):
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

# Smoke Test
mobilenet_test_model = get_mobilenet_v2(num_classes=len(class_names), freeze_features=True).to(DEVICE)
dummy_out = mobilenet_test_model(sample_imgs[:2].to(DEVICE))
print("MobileNetV2 initialized successfully. Output shape:", dummy_out.shape)
""")

    # =========================================================================
    # Step 5: Training Utility Functions
    # =========================================================================
    add_md(r"""## Step 5: Shared Training & Evaluation Loops""")
    add_code(r"""def train_one_epoch(model, dataloader, criterion, optimizer, device):
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

def evaluate_epoch(model, dataloader, criterion, device):
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
""")

    # =========================================================================
    # Step 6: Training Baseline CNN
    # =========================================================================
    add_md(r"""## Step 6: Train Custom Baseline CNN (15 Epochs)
- Optimizer: `Adam(lr=1e-3, weight_decay=1e-4)`
- Loss Function: `CrossEntropyLoss()`
""")
    add_code(r"""baseline_save_path = str(MODELS_DIR / "baseline_cnn_best.pth")
baseline_model = DefectBaselineCNN(num_classes=len(class_names)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(baseline_model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs_baseline = 15
best_val_acc_baseline = 0.0

print("=== Starting Baseline CNN Training ===")
start_time = time.time()

for epoch in range(1, epochs_baseline + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(baseline_model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc = evaluate_epoch(baseline_model, test_loader, criterion, DEVICE)
    duration = time.time() - t0

    print(f"Epoch [{epoch:02d}/{epochs_baseline:02d}] "
          f"| Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% "
          f"| Test Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}% "
          f"| Time: {duration:.1f}s")

    if val_acc > best_val_acc_baseline:
        best_val_acc_baseline = val_acc
        torch.save(baseline_model.state_dict(), baseline_save_path)
        print(f"  --> Saved new best baseline model ({val_acc*100:.2f}%) to {baseline_save_path}")

print(f"\nBaseline Training Finished in {(time.time() - start_time)/60:.2f} minutes.")
print(f"Best Baseline Test Accuracy: {best_val_acc_baseline*100:.2f}%")
""")

    # =========================================================================
    # Step 7: Training MobileNetV2 (Two Phases)
    # =========================================================================
    add_md(r"""## Step 7: Train MobileNetV2 with Two-Phase Transfer Learning
- **Phase 1 (Warmup - 4 Epochs):** Feature backbone frozen, train only classification head (`lr=1e-3`).
- **Phase 2 (Fine-Tuning - 8 Epochs):** Unfreeze top 4 bottleneck blocks with differential learning rates (`lr=1e-4` on features, `3e-4` on head).
""")
    add_code(r"""mobilenet_save_path = str(MODELS_DIR / "mobilenet_v2_best.pth")
mobilenet_model = get_mobilenet_v2(num_classes=len(class_names), freeze_features=True).to(DEVICE)
criterion = nn.CrossEntropyLoss()

best_val_acc_mobilenet = 0.0

# Phase 1: Train Classification Head Only (Warmup)
warmup_epochs = 4
optimizer_p1 = optim.Adam(mobilenet_model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

print("\n--- Phase 1: Training Classification Head (Backbone Frozen) ---")
for epoch in range(1, warmup_epochs + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(mobilenet_model, train_loader, criterion, optimizer_p1, DEVICE)
    val_loss, val_acc = evaluate_epoch(mobilenet_model, test_loader, criterion, DEVICE)
    duration = time.time() - t0

    print(f"Warmup [{epoch:02d}/{warmup_epochs:02d}] "
          f"| Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% "
          f"| Test Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}% "
          f"| Time: {duration:.1f}s")
    if val_acc > best_val_acc_mobilenet:
        best_val_acc_mobilenet = val_acc
        torch.save(mobilenet_model.state_dict(), mobilenet_save_path)

# Phase 2: Unfreeze top bottleneck layers for Fine-Tuning
finetune_epochs = 8
for param in mobilenet_model.features[-4:].parameters():
    param.requires_grad = True

optimizer_p2 = optim.Adam([
    {"params": mobilenet_model.features[-4:].parameters(), "lr": 1e-4},
    {"params": mobilenet_model.classifier.parameters(), "lr": 3e-4}
], weight_decay=1e-4)

print("\n--- Phase 2: Unfreezing Top Feature Layers for Fine-Tuning ---")
for epoch in range(1, finetune_epochs + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(mobilenet_model, train_loader, criterion, optimizer_p2, DEVICE)
    val_loss, val_acc = evaluate_epoch(mobilenet_model, test_loader, criterion, DEVICE)
    duration = time.time() - t0

    print(f"Finetune [{epoch:02d}/{finetune_epochs:02d}] "
          f"| Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% "
          f"| Test Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}% "
          f"| Time: {duration:.1f}s")
    if val_acc > best_val_acc_mobilenet:
        best_val_acc_mobilenet = val_acc
        torch.save(mobilenet_model.state_dict(), mobilenet_save_path)
        print(f"  --> Saved new best MobileNetV2 ({val_acc*100:.2f}%) to {mobilenet_save_path}")

print(f"\nMobileNetV2 Training Finished. Best Test Accuracy: {best_val_acc_mobilenet*100:.2f}%")
""")

    # =========================================================================
    # Step 8: Comprehensive Evaluation & Benchmarking
    # =========================================================================
    add_md(r"""## Step 8: Comprehensive Benchmark, Metrics & Confusion Matrix Comparison
- Load the saved best checkpoints for both models.
- Compute single-patch inference latency (ms/patch) and disk weight sizes (MB).
- Generate classification reports across all 6 defect categories.
- Plot side-by-side normalized confusion matrices.
""")
    add_code(r"""def get_model_size_mb(path: str) -> float:
    return Path(path).stat().st_size / (1024 * 1024)

def benchmark_and_predict(model, dataloader, device, warmup_batches: int = 5):
    model.eval()
    all_preds = []
    all_targets = []
    latencies = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)

            if i < warmup_batches and device.type == "cuda":
                _ = model(images)
                torch.cuda.synchronize()
                continue

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            outputs = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0 / images.size(0)
            latencies.append(elapsed_ms)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    avg_latency = float(np.mean(latencies))
    return np.array(all_preds), np.array(all_targets), avg_latency

# Load Best Baseline CNN
best_baseline = DefectBaselineCNN(num_classes=len(class_names)).to(DEVICE)
best_baseline.load_state_dict(torch.load(baseline_save_path, map_location=DEVICE))

# Load Best MobileNetV2
best_mobilenet = get_mobilenet_v2(num_classes=len(class_names), freeze_features=False).to(DEVICE)
best_mobilenet.load_state_dict(torch.load(mobilenet_save_path, map_location=DEVICE))

print("Benchmarking Baseline CNN...")
b_preds, b_targets, b_latency = benchmark_and_predict(best_baseline, test_loader, DEVICE)
b_f1 = f1_score(b_targets, b_preds, average="macro")
b_size = get_model_size_mb(baseline_save_path)

print("Benchmarking MobileNetV2...")
m_preds, m_targets, m_latency = benchmark_and_predict(best_mobilenet, test_loader, DEVICE)
m_f1 = f1_score(m_targets, m_preds, average="macro")
m_size = get_model_size_mb(mobilenet_save_path)

print("\n" + "=" * 55)
print("           BASELINE CNN CLASSIFICATION REPORT")
print("=" * 55)
print(classification_report(b_targets, b_preds, target_names=class_names, digits=4))

print("\n" + "=" * 55)
print("          MOBILENETV2 CLASSIFICATION REPORT")
print("=" * 55)
print(classification_report(m_targets, m_preds, target_names=class_names, digits=4))

print("\n" + "=" * 65)
print(f"{'Metric':<25} | {'Baseline CNN':<16} | {'MobileNetV2':<16}")
print("-" * 65)
print(f"{'Macro F1-Score':<25} | {b_f1:<16.4f} | {m_f1:<16.4f}")
print(f"{'Avg Latency (ms/patch)':<25} | {b_latency:<16.2f} | {m_latency:<16.2f}")
print(f"{'Model Weight Size (MB)':<25} | {b_size:<16.2f} | {m_size:<16.2f}")
print("=" * 65)
""")

    # =========================================================================
    # Step 9: Plot Confusion Matrices
    # =========================================================================
    add_md(r"""## Step 9: Side-by-Side Confusion Matrix Visualization""")
    add_code(r"""cm_baseline = confusion_matrix(b_targets, b_preds)
cm_mobilenet = confusion_matrix(m_targets, m_preds)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, cm, title in zip(
    axes,
    [cm_baseline, cm_mobilenet],
    ["Baseline CNN Confusion Matrix", "MobileNetV2 Confusion Matrix"]
):
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=False
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

plt.tight_layout()
cm_save_path = str(REPORTS_DIR / "confusion_matrices.png")
plt.savefig(cm_save_path, dpi=300)
plt.show()
print(f"Saved confusion matrix chart to: {cm_save_path}")
""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)

    print(f"Successfully generated clean, validated pipeline notebook at: {output_path}")

if __name__ == "__main__":
    create_full_pipeline_notebook()
