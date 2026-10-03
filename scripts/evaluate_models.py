import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.dataset import get_dataloaders
from models.baseline_cnn import DefectBaselineCNN
from models.mobilenet_v2 import get_mobilenet_v2


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model_size_mb(path: str) -> float:
    """Returns file size in megabytes."""
    return Path(path).stat().st_size / (1024 * 1024)


def benchmark_and_predict(model, dataloader, device, warmup_batches: int = 5):
    """
    Collects predictions, ground truth labels, and measures single-patch inference latency.
    """
    model.eval()
    all_preds = []
    all_targets = []
    latencies = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)

            # Warmup runs to avoid initial CUDA context spin-up overhead
            if i < warmup_batches and device.type == "cuda":
                _ = model(images)
                torch.cuda.synchronize()
                continue

            # Latency measurement
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


def plot_side_by_side_cm(cm_baseline, cm_mobilenet, class_names, output_path: str = "reports/confusion_matrices.png"):
    """Generates and saves side-by-side normalized confusion matrices."""
    Path("reports").mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, cm, title in zip(
        axes,
        [cm_baseline, cm_mobilenet],
        ["Baseline CNN Confusion Matrix", "MobileNetV2 Confusion Matrix"]
    ):
        # Normalize per class row
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
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nSaved confusion matrices to: {output_path}")


def run_evaluation():
    device = get_device()
    print(f"Running evaluation benchmark on: {device}")

    # Use batch_size=1 or batch_size=32 for DataLoader; batch_size=32 with per-sample timing
    _, test_loader, class_names = get_dataloaders(data_dir="data", batch_size=32, num_workers=2, img_size=224)

    # 1. Load Baseline CNN
    baseline_path = "models/baseline_cnn_best.pth"
    baseline_model = DefectBaselineCNN(num_classes=len(class_names)).to(device)
    baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))

    # 2. Load MobileNetV2
    mobilenet_path = "models/mobilenet_v2_best.pth"
    mobilenet_model = get_mobilenet_v2(num_classes=len(class_names), freeze_features=False).to(device)
    mobilenet_model.load_state_dict(torch.load(mobilenet_path, map_location=device))

    # Benchmark Baseline
    print("\nEvaluating Baseline CNN...")
    b_preds, b_targets, b_latency = benchmark_and_predict(baseline_model, test_loader, device)
    b_f1 = f1_score(b_targets, b_preds, average="macro")
    b_size = get_model_size_mb(baseline_path)

    # Benchmark MobileNetV2
    print("Evaluating MobileNetV2...")
    m_preds, m_targets, m_latency = benchmark_and_predict(mobilenet_model, test_loader, device)
    m_f1 = f1_score(m_targets, m_preds, average="macro")
    m_size = get_model_size_mb(mobilenet_path)

    # Print Classification Reports
    print("\n" + "=" * 55)
    print("           BASELINE CNN CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(b_targets, b_preds, target_names=class_names, digits=4))

    print("\n" + "=" * 55)
    print("          MOBILENETV2 CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(m_targets, m_preds, target_names=class_names, digits=4))

    # Print Summary Comparison Table
    print("\n" + "=" * 65)
    print(f"{'Metric':<25} | {'Baseline CNN':<16} | {'MobileNetV2':<16}")
    print("-" * 65)
    print(f"{'Macro F1-Score':<25} | {b_f1:<16.4f} | {m_f1:<16.4f}")
    print(f"{'Avg Latency (ms/patch)':<25} | {b_latency:<16.2f} | {m_latency:<16.2f}")
    print(f"{'Model Weight Size (MB)':<25} | {b_size:<16.2f} | {m_size:<16.2f}")
    print("=" * 65)

    # Generate Confusion Matrix Figure
    cm_baseline = confusion_matrix(b_targets, b_preds)
    cm_mobilenet = confusion_matrix(m_targets, m_preds)
    plot_side_by_side_cm(cm_baseline, cm_mobilenet, class_names)


if __name__ == "__main__":
    run_evaluation()