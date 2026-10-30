# PCB Defect Inspection — Presentation Guide: Part 3
**Speaker:** Sudeep  
**Topic:** MobileNetV2 Model, Transfer Learning, Two-Phase Training Strategy & Final Results

---

## Code

### 1. MobileNetV2 Model Architecture (`models/mobilenet_v2.py`)
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

### 2. Two-Phase Training Setup (`scripts/train_mobilenet.py`)
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

## 🎙️ Spoken Presentation Script

### 1. Sudeep's Opening & Overview
> *"I will cover our second model architecture—**Pretrained MobileNetV2**—explain our **Two-Phase Transfer Learning Strategy**, and share our **Final Comparative Results and Benchmarks**."*

---

### 2. Why MobileNetV2 & Transfer Learning?
> *"In industrial manufacturing, defect datasets are often limited. Transfer learning allows us to take a model trained on over 1.4 million images (ImageNet) and adapt its rich visual knowledge directly to PCB defect inspection.*
>
> *We specifically selected **MobileNetV2** for two engineering breakthroughs:*
>***Inverted Residual Bottlenecks:***  
>    *Unlike traditional ResNets that compress features first, MobileNetV2 expands channels inside the block (by $6\times$), applies depthwise convolution in the expanded space, and then projects back down through a **linear bottleneck**. Keeping the bottleneck linear (no ReLU) prevents the network from destroying subtle edge information."*

---

### 3. Our Two-Phase Training Strategy
> We implemented a disciplined **Two-Phase Training Strategy**:*
>
> * **Phase 1: Feature Backbone Freezing (Warmup — 4 Epochs)**  
>   *We froze the entire feature backbone. Only our custom classification head (`Dropout(0.3) -> Linear(1280, 6)`) was trained with an Adam optimizer at a learning rate of $10^{-3}$. This gave the head time to align with the pretrained feature extractor.*
>
> * **Phase 2: Fine-Tuning with Differential Learning Rates (8 Epochs)**  
>   *Next, we unfroze the top 4 convolutional blocks (`model.features[-4:]`) to adapt them specifically to copper trace textures. We used a **differential learning rate**:*
>   * A very gentle learning rate of **$10^{-4}$** on the feature blocks so we didn't overwrite pretrained features.
>   * A moderate learning rate of **$3 \times 10^{-4}$** on the classification head.*

---

### 4. Final Results & Side-by-Side Comparison
> *"Both models were benchmarked on the exact same isolated test set of **1,967 cropped defect patches**.*
>
> *Here is the side-by-side performance comparison:*

| Metric | Baseline CNN (Custom) | MobileNetV2 (Pretrained) | Advantage / Difference |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | 98.07% | **98.83%** | **+0.76%** (+15 fewer defect escapes) |
| **Macro F1-Score** | 0.9816 | **0.9888** | **+0.0072** across all 6 classes |


---

### 5. Confusion Matrix & Diagnostic Highlights
> *"Looking closely at our confusion matrix and per-class metrics reveals three critical takeaways:*
>
> 1. ***Massive Reduction in Fatal False Alarms (`Open` and `Short`):***  
>    *Open circuits and short circuits are the two most electrically dangerous defects. Baseline CNN achieved 95.00% precision on Open defects and 96.35% on Short defects. MobileNetV2 boosted Open precision to **98.96%** and Short precision to **99.31%**. In a production line, this drastically cuts false alarms on catastrophic flaws.*
> 2. ***Flawless Copper Classification:***  
>    *MobileNetV2 achieved a perfect **100% precision, 100% recall, and 1.0000 F1-score** on Spurious Copper (301 out of 301 samples correctly identified without a single mistake).*
> 3. ***Resolving Boundary Ambiguities:***  
>    *Subtle defects like Mousebites (tiny nibbles out of a trace edge) can easily look like Open circuits. MobileNetV2's deeper receptive field resolved these subtle edge ambiguities significantly better than the 4-layer custom baseline.*

---

## 🎯 Potential Evaluator Questions & Answers

**Q1: What is Transfer Learning and why did you use it here?**  
> *"Transfer learning takes visual knowledge (edge detectors, corner filters, textures) learned by a model trained on a massive benchmark like ImageNet (1.4 million images) and applies it to our specialized PCB defect domain. It allows us to achieve higher accuracy and faster convergence without needing millions of labeled PCB defect images."*


---

**Q2: What is an Inverted Residual Block and why is the bottleneck linear (no ReLU)?**  
> *"In standard ResNet blocks, features are compressed, processed, and expanded (wide $\rightarrow$ narrow $\rightarrow$ wide). MobileNetV2 inverts this: it expands low-dimensional features to a higher dimension ($6\times$), applies depthwise convolution in high-dimensional space, and then projects back down to a narrow bottleneck.*  
> *Crucially, it uses **no non-linearity (no ReLU) at the narrow output bottleneck**. Applying ReLU to a low-dimensional manifold zeroes out negative numbers and collapses critical feature dimensions. Keeping the bottleneck strictly linear prevents information destruction."*

---

**Q4: Why did you freeze the backbone in Phase 1 instead of training end-to-end immediately?**  
> *"The newly initialized classification head (`Linear(1280, 6)`) starts with completely random weights. If we trained the whole model immediately, the massive gradient errors from this untrained head would backpropagate into the backbone and wipe out the delicate pretrained ImageNet weights—a phenomenon called **catastrophic forgetting**. Freezing ensures the backbone stays stable while the head learns to make sense of the features."*

---

**Q5: What is a "differential learning rate" and why was it used in Phase 2?**  
> *"In Phase 2, we used two different learning rates simultaneously:*  
> * A slow rate ($10^{-4}$) for the top convolutional feature blocks.  
> * A higher rate ($3 \times 10^{-4}$) for the newly added classification head.  
> *This allows the feature extractor to gently adapt to PCB trace textures without disrupting pretrained patterns, while giving the classification head enough learning momentum to fine-tune its decision boundaries."*

---

**Q6: Which model would you recommend for actual deployment in a factory?**  
> *"It depends on the hardware budget:*  
> * If deployed on ultra-low-power microcontrollers or embedded edge boards with tight memory constraints, **Baseline CNN** is ideal (5.51 MB, 49 ms latency, 98.07% accuracy).  
> * For production lines where defect escape costs are high, **MobileNetV2** is the recommended choice. Its 99.31% precision on Short defects and 98.96% precision on Open defects drastically reduce fatal circuit escapes, making the slight 8 ms latency trade-off completely worthwhile."*
