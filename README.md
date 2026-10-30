# Automated PCB Defect Inspection Using Deep Learning

An end-to-end deep learning project comparing a **Custom 4-Block Baseline CNN** against a **Pretrained MobileNetV2** using transfer learning on the **DeepPCB** benchmark dataset.

---

# Part 1: Problem Statement, Dataset & Preprocessing

## 1. Problem Statement & Industrial Gap

* **The Problem:**  
  The Printed Circuit Board (PCB) is the backbone of modern electronics manufacturing. Today’s PCBs feature ultra-dense layouts with ultra-thin copper traces. Even a microscopic structural defect on a copper trace causes circuit failures, signal degradation, or catastrophic short circuits.

* **Current Bottlenecks in Quality Control:**  
  1. **Manual Visual Inspection (MVI):** Human inspectors working under microscopes suffer cognitive visual fatigue, leading to a defect escape rate of **15% to 30%**.  
  2. **Traditional Automated Optical Inspection (AOI):** Conventional AOI systems rely on rigid template matching and pixel subtraction. They are hypersensitive to slight lighting variations and surface glare, causing high false alarm rates.

* **Our Solution:**  
  Our system implements an end-to-end deep learning computer vision pipeline. Instead of brittle pixel subtraction, defect detection is formulated as a visual feature learning problem. The network ingests localized copper trace patches and classifies them across six critical manufacturing defect categories with over **98.8% accuracy**.

---

## 2. Dataset Overview: DeepPCB

The system utilizes the **DeepPCB Benchmark Dataset**, released by Peking University’s Intelligent Hardware Laboratory.

* **Dataset Characteristics:**
  * Contains **1,500 pairs of aligned high-resolution optical images** (approximately $640 \times 640$ pixels) captured using industrial-grade PCB scanners.
  * Each pair contains a **defect-free template board** (`*_temp.jpg`) and a **damaged test board** (`*_test.jpg`) containing genuine manufacturing defects.
  * Annotations provide bounding box coordinates `[x1, y1, x2, y2]` along with a class identifier for six defect categories:
    1. **Open:** An unintended break or physical gap along a conductive trace.
    2. **Short:** An unintended conductive copper bridge spanning between adjacent traces.
    3. **Mousebite:** An irregular edge erosion or bite taken out of the copper boundary.
    4. **Spur:** A sharp, unwanted copper flare or protrusion extending from a trace.
    5. **Copper (Spurious Copper):** An isolated island of unetched excess copper on the substrate.
    6. **Pin-hole:** A tiny void or hollow cavity internal to a copper track.

---

## 3. Dataset Preprocessing Pipeline

Raw full-board images cannot be fed directly into standard convolutional neural networks efficiently. The preprocessing engine performs the following:

1. **Defect Annotations:**  
   The dataset provides bounding box coordinates `[x1, y1, x2, y2]` for each defect. These coordinates are used to crop the localized region of interest (ROI) from the test board.

2. **Contextual Bounding Box Expansion (15% Padding):**  
   Raw annotation boxes tightly wrap only the defective pixels. Cropping strictly inside the box strips away surrounding trace continuity and ground planes. A 15% margin padding is added around every bounding box, enabling the network to perceive surrounding structural context while remaining clamped within image limits.

3. **Board-Level Split Isolation (Preventing Data Leakage):**  
   Splitting cropped defect patches randomly into train and test sets causes severe test data leakage, as multiple defect patches from the same board share identical lighting, etching color, and solder mask tint. To prevent this, `train_test_split` was executed strictly at the **PCB Board level (80% Train, 20% Test)**.
   * **Train Set:** **8,046** images (~80.35%)
   * **Test Set:** **1,967** images (~19.65%)
   * **Total Cropped Defect Patches:** **10,013** images

4. **Standardization & Data Augmentation:**  
   * All cropped patches are resized to a uniform dimension of **$224 \times 224$ pixels**.
   * **Training Augmentations:** `RandomHorizontalFlip(p=0.5)`, `RandomVerticalFlip(p=0.5)`, and `RandomRotation(degrees=10)` to prevent overfitting.
   * **Normalization:** Converted to tensors and normalized using ImageNet channel statistics (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).

---

## 4. Dataset & Preprocessing Q&A

**Q: Why crop defect patches instead of passing the entire 640×640 board to the CNN?**  
Passing full-board images directly to a classification CNN causes tiny defects (often only $15 \times 15$ pixels) to get washed out during successive pooling operations. Cropping localized patches standardizes the receptive field on the actual anomaly while drastically reducing memory consumption.

**Q: What is "data leakage" in this context, and how did you prevent it?**  
If all patches are cropped first and then randomly split, patches from the exact same physical board end up in both training and testing sets. The model would cheat by memorizing board-specific background artifacts (lighting, solder mask tint). This was prevented by splitting at the parent PCB board annotation level before cropping.

**Q: Why choose 15% padding specifically?**  
A 15% contextual margin provides sufficient boundary continuity for adjacent traces and substrate edges without capturing neighboring, unrelated defects into the same crop.

---

# Part 2: Custom Baseline CNN Architecture

## 1. Baseline Model Code

```python
import torch
import torch.nn as nn

class DefectBaselineCNN(nn.Module):
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

        # Adaptive pooling ensures consistent output size regardless of input spatial dimension
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),                 # 256 * 4 * 4 = 4096 flattened features
            nn.Linear(256 * 4 * 4, 256),  # Non-linear projection bottleneck
            nn.ReLU(inplace=True), 
            nn.Dropout(p=dropout_p),      # 40% neuron dropout against co-adaptation
            nn.Linear(256, num_classes)   # 6 raw defect logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return logits
```

---

## 2. Architecture Flow Diagram

```text
INPUT: Normalized Defect Patch (3 × 224 × 224)
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ BLOCK 1: Conv2d(3→32, k=3, p=1) ➔ BatchNorm ➔ ReLU ➔ MaxPool2d │ ➔ Output: 32 × 112 × 112
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ BLOCK 2: Conv2d(32→64, k=3, p=1) ➔ BatchNorm ➔ ReLU ➔ MaxPool2d│ ➔ Output: 64 × 56 × 56
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ BLOCK 3: Conv2d(64→128, k=3, p=1) ➔ BatchNorm ➔ ReLU ➔ MaxPool2d│➔ Output: 128 × 28 × 28
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ BLOCK 4: Conv2d(128→256, k=3, p=1) ➔ BatchNorm ➔ ReLU ➔ MaxPool │➔ Output: 256 × 14 × 14
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ SPATIAL POOLING: AdaptiveAvgPool2d((4, 4))                      │ ➔ Output: 256 × 4 × 4
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ CLASSIFIER HEAD:                                                │
│ Flatten (4096) ➔ Linear(4096 ➔ 256) ➔ ReLU ➔ Dropout(0.4) ➔    │
│ Linear(256 ➔ 6)                                                 │ ➔ Output: 6 Class Logits
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Breakdown

### A. The 4 Core Layers in Each Convolutional Block
The feature extractor is composed of 4 identical blocks, each executing a fixed sequence: `Conv2d` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d`.

#### 1. `Conv2d` (Convolutional Layer)
* **What it does:** Slides small $3\times3$ scanning filters across the image to detect local visual features like edges, lines, corners, and color transitions.
* **Role:** Looks directly at the PCB surface to spot copper trace boundaries, gaps, and abnormal copper shapes (e.g., breaks in traces or unwanted copper bridges).
* **Input (Block 1):** `Shape: (Batch, 3, 224, 224)` — Raw RGB defect crop (3 color channels, $224 \times 224$).
* **Output (Block 1):** `Shape: (Batch, 32, 224, 224)` — 32 distinct feature maps. Each channel represents one learned filter highlighting specific trace edges. Spatial size remains $224 \times 224$ because `padding=1` preserves borders.

#### 2. `BatchNorm2d` (Batch Normalization)
* **What it does:** Normalizes the activations produced by convolution so they have zero mean and unit variance.
* **Role:** During training, as weights update, activations can swing wildly and destabilize deeper layers. BatchNorm stabilizes these values, enabling faster and more reliable convergence with a learning rate of $10^{-3}$.
* **Input (Block 1):** `Shape: (Batch, 32, 224, 224)` — Unnormalized activations from `Conv2d`.
* **Output (Block 1):** `Shape: (Batch, 32, 224, 224)` — Normalized activations with identical dimensions.

#### 3. `ReLU` (Activation Function)
* **What it does:** Computes $f(x) = \max(0, x)$. Passes positive values through unaltered and converts negative values to zero.
* **Role:** Introduces **non-linearity**. Without ReLU, stacking convolution layers is mathematically just simple linear algebra, meaning the network could only learn straight lines. ReLU allows learning complex shapes like circular solder holes and curved traces.
* **Input (Block 1):** `Shape: (Batch, 32, 224, 224)` — Normalized feature maps from BatchNorm.
* **Output (Block 1):** `Shape: (Batch, 32, 224, 224)` — Feature maps with negative responses zeroed out.

#### 4. `MaxPool2d` (Spatial Pooling Layer)
* **What it does:** Partitions feature maps into $2\times2$ pixel tiles and selects only the single largest activation in each tile, discarding the rest.
* **Role:**
  1. **Shrinks spatial size by 50%:** Reduces $224 \times 224 \rightarrow 112 \times 112$, cutting spatial calculations by 75% per block.
  2. **Retains peak defect signals:** Defects like breaks or pin-holes create sharp, strong activation peaks; max pooling guarantees these peak signals survive while discarding flat, uniform background.
* **Input (Block 1):** `Shape: (Batch, 32, 224, 224)` — Activated feature maps from ReLU.
* **Output (Block 1):** `Shape: (Batch, 32, 112, 112)` — Downsampled feature maps with preserved channels.

This sequence repeats across all 4 blocks:
* **Block 1:** $3 \times 224 \times 224 \longrightarrow 32 \times 112 \times 112$
* **Block 2:** $32 \times 112 \times 112 \longrightarrow 64 \times 56 \times 56$
* **Block 3:** $64 \times 56 \times 56 \longrightarrow 128 \times 28 \times 28$
* **Block 4:** $128 \times 28 \times 28 \longrightarrow 256 \times 14 \times 14$

---

### B. Channel Progression Significance (3 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 128 $\rightarrow$ 256)
As spatial resolution decreases ($224 \rightarrow 14$), the channel count doubles ($3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$), representing **hierarchical feature abstraction**:
* **Block 1 ($3 \rightarrow 32$ channels):** Extracts low-level, high-frequency spatial features like straight copper trace edges and sharp corners.
* **Block 2 ($32 \rightarrow 64$ channels):** Aggregates features into mid-level geometric patterns like copper trace curvatures, angles, and surface textures.
* **Block 3 ($64 \rightarrow 128$ channels):** Identifies semantic structures like trace continuity and copper voids.
* **Block 4 ($128 \rightarrow 256$ channels):** Extracts high-level defect-specific representations, such as whether a void represents a localized pin-hole, an irregular mousebite notch, or a complete open circuit.

By trading spatial resolution for channel depth, each neuron in Block 4 views an extensive receptive field of the original patch.

---

### C. Spatial Pooling: `nn.AdaptiveAvgPool2d((4, 4))`
Following Block 4, the feature tensor is downsampled to $256 \times 14 \times 14$. Rather than immediately flattening $14 \times 14$ or using standard pooling, Adaptive Average Pooling is applied:
* **What it is:** Automatically calculates the required dynamic kernel size and stride to force the output to exactly $4 \times 4$ across all 256 feature channels.
* **Why we use it:**
  1. Guarantees a strictly fixed dimension ($256 \times 4 \times 4 = 4,096$ values) before entering fully connected layers, making the architecture robust to minor input variations.
  2. Computes spatial averages over localized zones, summarizing regional defect activations rather than selecting isolated extreme noise spikes.

---

### D. The Classification Head: Why Drop 4,096 to 256?
$$\text{Flatten}(4,096) \xrightarrow{\text{Linear}} 256 \xrightarrow{\text{ReLU}} \text{Dropout}(0.4) \xrightarrow{\text{Linear}} 6$$

1. **Non-Linear Feature Synthesis:** A direct projection ($4096 \rightarrow 6$) acts as a purely linear classifier. Adding the intermediate 256 hidden neurons with a ReLU activation creates a non-linear latent bottleneck that learns complex combinations of spatial features.
2. **Preventing Parameter Explosion & Overfitting:** Compressing 4,096 down to 256 acts as an **information bottleneck**, distilling the most salient discriminative signals.
3. **Dropout Regularization ($p=0.4$):** During training, 40% of the neurons are randomly set to zero on each forward pass, forcing neurons to learn independent, redundant defect cues.
4. **Final Linear Projection ($256 \rightarrow 6$):** Maps the 256 distilled features to 6 unnormalized log-probability scores (logits) corresponding to: `Open`, `Short`, `Mousebite`, `Spur`, `Copper`, and `Pin-hole`.

---

## 4. Baseline CNN Q&A

**Q1: What are "channels" (often casually called layers) and how do they progress from 3 (RGB) up to 256?**  
In deep learning, people sometimes casually say 'layers of the image', but their technical name is **Channels** (or **Feature Maps**). A layer is the mathematical operation (like `Conv2d`), whereas channels are the depth of information passing through it:
* **Starting with 3 Channels (RGB):** The input image is like a stack of 3 transparent sheets: Red intensity, Green intensity, and Blue intensity (shape `3 × 224 × 224`).
* **Progressing to 32, 64, 128, and 256 Channels:** `Conv2d` creates specialized inspection sheets:
  * Block 1 (32 channels): Filters highlighting basic physical edges (horizontal lines, vertical lines, corners).
  * Block 2 (64 channels): Geometric patterns (T-junctions, track bends, via holes).
  * Block 3 (128 channels): Continuous copper tracks, isolation, and voids.
  * Block 4 (256 channels): Specialized defect detector sheets (hairline breaks for Open, accidental bridges for Short, edge nibbles for Mousebite).

**Q2: How exactly does ReLU add "non-linearity"?**  
Convolution is purely linear math ($y = Wx + b$). Stacking multiple convolution layers without an activation function collapses mathematically into a single linear equation ($W_2(W_1 x) = W_{\text{combined}} x$), which can only draw straight-line decision boundaries.  
ReLU computes $f(x) = \max(0, x)$. It creates a sharp kink/threshold at zero: positive values pass with slope 1, and negative values become zero. By activating different neurons for different inputs, it bends and folds the decision space, allowing the network to learn complex curved copper tracks and irregular defect contours.

**Q3: In `MaxPool2d`, how does it make sure it doesn't filter out important defect data?**  
1. `Conv2d` has already highlighted the defect: abnormal copper bridges or missing voids produce strong, high-magnitude activation spikes, while normal uniform substrate produces low or near-zero values.
2. Max pooling specifically selects $\max(x_1, x_2, x_3, x_4)$, actively extracting the strongest anomaly signal in that neighborhood while discarding background noise.
3. Neighboring pixels in a $2\times2$ grid are spatially redundant; max pooling preserves feature presence while stripping away redundant pixel duplicates.

**Q4: How does adding layers one by one up to 256 channels increase complexity?**  
* **Receptive field expansion:** A Block 1 neuron only sees a $3\times3$ pixel window. By Block 4, through repeated pooling and convolutions, each neuron's receptive field spans almost the entire $224\times224$ patch, enabling global context understanding.
* **Hierarchical abstraction:** Early layers represent simple primitives (edges); deeper layers combine them into corners, continuous tracks, and full defect concepts.
* **Multi-class separation capacity:** Distinguishing 6 defect categories simultaneously requires 256 independent feature detectors.

**Q5: How does `AdaptiveAvgPool2d` capture defect info by averaging? Doesn't averaging blur the defect?**  
Averaging is applied at the end of Block 4 ($14\times14$), where convolutions have already detected the defect; the 256 channels are activation maps, not raw pixels. Pooling down to a $4\times4$ grid (16 regional zones) measures the **density and area** of defect activations, preventing single-pixel noise outliers from dominating classification while retaining spatial layout.

**Q6: What exactly are "6 unnormalized log-probability scores (logits)" and how do they correspond to output labels?**  
The final layer `Linear(256, 6)` outputs 6 raw real-valued numbers ($-\infty$ to $+\infty$). Each position corresponds alphabetically to one class:
* `Index 0:` **Copper**
* `Index 1:` **Mousebite**
* `Index 2:` **Open**
* `Index 3:` **Pin-hole**
* `Index 4:` **Short**
* `Index 5:` **Spur**  
Applying the **Softmax function** converts logits into percentage probabilities summing to 100%. During inference, `argmax(logits)` selects the predicted defect category.

**Q7: What happens if you remove `AdaptiveAvgPool2d` and just flatten directly after Block 4?**  
Directly flattening $14 \times 14 \times 256$ yields $50,176$ features. Connecting $50,176$ to 256 would require $12.8\text{ million}$ parameters in just one dense layer—an 800% increase in model size—severely overfitting the dataset. Adaptive pooling to $4\times4$ keeps total model parameters to a lightweight 1.43M.

**Q8: Why use `Dropout(0.4)` instead of 0.7 or 0.1?**  
A dropout rate of 0.4 provides an optimal balance: it deactivates enough neurons to prevent co-adaptation without causing underfitting or destabilizing gradient flow.

**Q9: Why use Batch Normalization before ReLU and not after?**  
Placing BatchNorm directly after the linear convolution ensures activations entering ReLU are centered at zero with unit variance, maximizing gradient throughput and avoiding dead neuron regimes.

---

# Part 3: Pretrained MobileNetV2 & Final Results

## 1. MobileNetV2 Model Code

### Architecture (`models/mobilenet_v2.py`)
```python
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights 

def get_mobilenet_v2(num_classes: int = 6, freeze_features: bool = True):
    # Load ImageNet-pretrained weights
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    # Phase 1: Freeze pretrained feature backbone
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head: 1280 input features -> Dropout(0.3) -> 6 defect classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model
```

### Two-Phase Training Setup (`scripts/train_mobilenet.py`)
```python
# Phase 1: Train Classification Head Only (4 Warmup Epochs)
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

# Phase 2: Unfreeze Top 4 Feature Blocks with Differential Learning Rates (8 Finetune Epochs)
for param in model.features[-4:].parameters():
    param.requires_grad = True

optimizer = optim.Adam([
    {"params": model.features[-4:].parameters(), "lr": 1e-4}, # Gentle rate for pre-trained features
    {"params": model.classifier.parameters(), "lr": 3e-4}      # Moderate rate for newly added head
], weight_decay=1e-4)
```

---

## 2. Why MobileNetV2 & Transfer Learning?

In industrial manufacturing, defect datasets are often limited. Transfer learning allows taking a model trained on over 1.4 million images (ImageNet) and adapting its rich visual knowledge directly to PCB defect inspection.

MobileNetV2 was selected for its **Inverted Residual Bottlenecks**:  
Unlike traditional ResNets that compress features first, MobileNetV2 expands channels inside the block (by $6\times$), applies depthwise convolution in the expanded space, and projects back down through a **linear bottleneck**. Keeping the bottleneck linear (no ReLU) prevents the network from destroying subtle edge information.

---

## 3. Two-Phase Training Strategy

To prevent the newly initialized head from destroying pretrained weights (*catastrophic forgetting*), training is divided into two phases:

* **Phase 1: Feature Backbone Freezing (Warmup — 4 Epochs)**  
  The feature backbone is completely frozen. Only the custom classification head (`Dropout(0.3) -> Linear(1280, 6)`) is trained with an Adam optimizer at a learning rate of $10^{-3}$, allowing the head to align with the pretrained feature extractor.
* **Phase 2: Fine-Tuning with Differential Learning Rates (8 Epochs)**  
  The top 4 convolutional blocks (`model.features[-4:]`) are unfrozen to adapt to copper trace textures using a differential learning rate:
  * A gentle learning rate of **$10^{-4}$** on the feature blocks so pretrained representations are not overwritten.
  * A moderate learning rate of **$3 \times 10^{-4}$** on the classification head.

---

## 4. Final Results & Side-by-Side Comparison

Both models were benchmarked on the exact same isolated test set of **1,967 cropped defect patches**:

| Metric | Baseline CNN (Custom) | MobileNetV2 (Pretrained) | Advantage / Difference |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | 98.07% | **98.83%** | **+0.76%** (+15 fewer defect escapes) |
| **Macro F1-Score** | 0.9816 | **0.9888** | **+0.0072** across all 6 classes |

---

## 5. Confusion Matrix & Diagnostic Highlights

1. **Massive Reduction in Fatal False Alarms (`Open` and `Short`):**  
   Open circuits and short circuits are the two most electrically dangerous defects. Baseline CNN achieved 95.00% precision on Open defects and 96.35% on Short defects. MobileNetV2 boosted Open precision to **98.96%** and Short precision to **99.31%**. In a production line, this drastically cuts false alarms on catastrophic flaws.
2. **Flawless Copper Classification:**  
   MobileNetV2 achieved a perfect **100% precision, 100% recall, and 1.0000 F1-score** on Spurious Copper (301 out of 301 samples correctly identified without a single mistake).
3. **Resolving Boundary Ambiguities:**  
   Subtle defects like Mousebites (tiny nibbles out of a trace edge) can easily look like Open circuits. MobileNetV2's deeper receptive field resolved these subtle edge ambiguities significantly better than the 4-layer custom baseline.

---

## 6. MobileNetV2 & Results Q&A

**Q1: What is Transfer Learning and why did you use it here?**  
Transfer learning takes visual knowledge (edge detectors, corner filters, textures) learned by a model trained on a massive benchmark like ImageNet (1.4 million images) and applies it to our specialized PCB defect domain. It allows achieving higher accuracy and faster convergence without needing millions of labeled PCB defect images.

**Q2: What is an Inverted Residual Block and why is the bottleneck linear (no ReLU)?**  
In standard ResNet blocks, features are compressed, processed, and expanded (wide $\rightarrow$ narrow $\rightarrow$ wide). MobileNetV2 inverts this: it expands low-dimensional features to a higher dimension ($6\times$), applies depthwise convolution in high-dimensional space, and then projects back down to a narrow bottleneck.  
Crucially, it uses **no non-linearity (no ReLU) at the narrow output bottleneck**. Applying ReLU to a low-dimensional manifold zeroes out negative numbers and collapses critical feature dimensions. Keeping the bottleneck strictly linear prevents information destruction.

**Q3: Why did you freeze the backbone in Phase 1 instead of training end-to-end immediately?**  
The newly initialized classification head (`Linear(1280, 6)`) starts with completely random weights. If the whole model were trained immediately, the massive gradient errors from this untrained head would backpropagate into the backbone and wipe out the delicate pretrained ImageNet weights—a phenomenon called **catastrophic forgetting**. Freezing ensures the backbone stays stable while the head learns to make sense of the features.

**Q4: What is a "differential learning rate" and why was it used in Phase 2?**  
In Phase 2, two different learning rates were used simultaneously:
* A slow rate ($10^{-4}$) for the top convolutional feature blocks.
* A higher rate ($3 \times 10^{-4}$) for the newly added classification head.  
This allows the feature extractor to gently adapt to PCB trace textures without disrupting pretrained patterns, while giving the classification head enough learning momentum to fine-tune its decision boundaries.

**Q5: Which model would you recommend for actual deployment in a factory?**  
It depends on the hardware budget:
* If deployed on ultra-low-power microcontrollers or embedded edge boards with tight memory constraints, **Baseline CNN** is ideal (5.51 MB, 49 ms latency, 98.07% accuracy).
* For production lines where defect escape costs are high, **MobileNetV2** is the recommended choice. Its 99.31% precision on Short defects and 98.96% precision on Open defects drastically reduce fatal circuit escapes, making the slight 8 ms latency trade-off completely worthwhile.
