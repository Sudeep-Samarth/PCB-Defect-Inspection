# PCB Defect Inspection — Presentation Guide: Part 2
**Speaker:** Aryan  
**Topic:** Baseline CNN Architecture

---
## Code
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

## 🎙️ Spoken Presentation Script

### 1. Aryan's Opening & Overview
> *"In this section, I will walk you through our **Custom 4-Block Baseline Convolutional Neural Network (CNN)**.*

---

### 2. Detailed Component Breakdown

#### A. The 4 Core Layers in Each Convolutional Block
> *"Our feature extractor is built from 4 identical blocks. Each block has 4 layers working together in a fixed sequence: `Conv2d` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d`. Here is what each layer does, why we use it in this code, and what enters and leaves each step:*
>
> ---
>
> #### 1. `Conv2d` (Convolutional Layer)
> * **What it does:**  
>   *It slides small $3\times3$ scanning filters across the image to detect local visual features like edges, lines, corners, and color transitions.*
> * **Role**  
>   *It looks directly at the PCB surface to spot copper trace boundaries, gaps, and anomalous copper shapes. Instead of looking at the whole board at once, it learns filters for tiny visual cues (e.g., 'is there a break in this wire?' or 'is there an unwanted copper bridge?').*
> * **Input (for Block 1):**  
>   * `Shape: (Batch, 3, 224, 224)`  
>   * The raw RGB defect crop (3 color channels: Red, Green, Blue, sized $224 \times 224$).
> * **Output (for Block 1):**  
>   * `Shape: (Batch, 32, 224, 224)`  
>   * 32 distinct feature maps. Each channel represents one learned filter highlighting specific trace edges. The spatial size remains $224 \times 224$ because `padding=1` preserves the borders.
>
> ---
>
> #### 2. `BatchNorm2d` (Batch Normalization)
> * **What it does:**  
>   *It takes all the numbers produced by the convolution and normalizes them so they have an average of 0 and a consistent spread (variance of 1).*
> * **Role**  
>   *During training, as weights update, numbers passing between layers can become too large or too small, causing training to become unstable or completely stall. BatchNorm keeps the numbers balanced and well-behaved, allowing the model to train faster and much more reliably with a learning rate of $10^{-3}$.*
> * **Input (for Block 1):**  
>   * `Shape: (Batch, 32, 224, 224)`  
>   * The raw, unnormalized activations coming straight out of `Conv2d`.
> * **Output (for Block 1):**  
>   * `Shape: (Batch, 32, 224, 224)`  
>   * The exact same shape and channels, but with stabilized, normalized values ready for the activation function.
>
> ---
>
> #### 3. `ReLU` (Activation Function)
> * **Role**  
>   *It introduces **non-linearity**. Without ReLU, stacking convolution layers is just simple linear algebra, which means that the network would only ever be able to learn straight lines. ReLU allows the network to learn complex shapes like circular solder holes and curved traces.*
> * **Input (for Block 1):**  
>   * `Shape: (Batch, 32, 224, 224)`  
>   * The normalized feature maps from BatchNorm.
> * **Output (for Block 1):**  
>   * `Shape: (Batch, 32, 224, 224)`  
>   * Same shape, but all negative responses are zeroed out (acting as a filter that only keeps meaningful feature activations).
>
> ---
>
> #### 4. `MaxPool2d` (Spatial Pooling Layer)
> * **What it does:**  
>   *It divides the image into small $2\times2$ pixel tiles and picks only the single largest number in each tile, throwing away the other three.*
> * **Role in this code:**  
>   1. **Shrinks spatial size by 50%:** *Reduces $224 \times 224 \rightarrow 112 \times 112$, cutting the number of pixel calculations by 75% so the model runs fast.*  
>   2. **Retains peak defect signals:** *Defects like an open circuit or pin-hole create sharp, strong activation peaks. Max pooling guarantees those peak defect signals survive into the next block while flat, uninteresting background is discarded.*  
> * **Input (for Block 1):**  
>   * `Shape: (Batch, 32, 224, 224)`  
>   * The activated feature maps from ReLU.
> * **Output (for Block 1):**  
>   * `Shape: (Batch, 32, 112, 112)`  
>   * The height and width are cut in half, while preserving all 32 feature channels.
>
> ---
>
> *This exact 4-layer sequence repeats across all 4 blocks, progressively doubling the channels while halving the spatial dimensions:*
> * **Block 1:** $3 \times 224 \times 224 \longrightarrow 32 \times 112 \times 112$
> * **Block 2:** $32 \times 112 \times 112 \longrightarrow 64 \times 56 \times 56$
> * **Block 3:** $64 \times 56 \times 56 \longrightarrow 128 \times 28 \times 28$
> * **Block 4:** $128 \times 28 \times 28 \longrightarrow 256 \times 14 \times 14$
>
---

#### B. Channel Progression Significance (3 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 128 $\rightarrow$ 256)
>
> *As spatial resolution decreases ($224 \rightarrow 14$), the channel count doubles ($3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$). This is a fundamental computer vision principle representing **hierarchical feature abstraction**:*
> * **Block 1 ($3 \rightarrow 32$ channels):** *Extracts low-level, high-frequency spatial features like straight copper trace edges  and sharp corners.*
> * **Block 2 ($32 \rightarrow 64$ channels):** *Aggregates features into mid-level geometric patterns like copper trace curvatures, angles, and surface textures.*
> * **Block 3 ($64 \rightarrow 128$ channels):** *Identifies semantic structures like trace continuity and copper voids.*
> * **Block 4 ($128 \rightarrow 256$ channels):** *Extracts high-level defect-specific representations such as whether a void represents a localized pin-hole, an irregular mousebite notch, or a complete open circuit.*
>
> *By trading spatial resolution for channel depth, the network gains a larger receptive field: each neuron in Block 4 views an extensive region of the original image patch."*

---

#### C. Spatial Pooling: `nn.AdaptiveAvgPool2d((4, 4))`
> *"Following Block 4, the feature tensor is downsampled to $256 \times 14 \times 14$. Rather than immediately flattening $14 \times 14$ or using a standard max pool, we apply **Adaptive Average Pooling** targeted to a fixed $(4, 4)$ spatial grid:*
>
> * **What is it?** *Unlike standard pooling where you define kernel size and stride, adaptive pooling automatically calculates the required dynamic kernel size(size of the block that slides over the image) and stride(step size of how much the kernel moves by over the image) to force the output to exactly $4 \times 4$ across all 256 feature channels.*
> * **Why do we use it?**  
>   1. *It guarantees a strictly fixed dimension ($256 \times 4 \times 4 = 4,096$ values) before entering the fully connected layers, making the architecture robust even if input patch sizes deviate slightly.*
>   2. *Average pooling computes spatial averages over localized zones, summarizing regional defect activations rather than selecting isolated extreme noise spikes."*

---

#### D. The Classification Head: Why Drop 4,096 to 256?
> *"The flattened feature vector contains 4,096 values. In our classification head, we do not connect 4,096 directly to the 6 output classes. Instead, we introduce an intermediate hidden dense layer:*
>
> $$\text{Flatten}(4,096) \xrightarrow{\text{Linear}} 256 \xrightarrow{\text{ReLU}} \text{Dropout}(0.4) \xrightarrow{\text{Linear}} 6$$
>
> *Why is this intermediate 256-node layer essential?*
> 1. ***Non-Linear Feature Synthesis***: *A direct projection ($4096 \rightarrow 6$) would act as a purely linear classifier on top of the feature map. Adding the intermediate 256 hidden neurons with a ReLU activation creates a non-linear latent bottleneck that learns complex combinations and cross-correlations of spatial features.*
> 2. ***Preventing Parameter Explosion & Overfitting***: *If we connected 4,096 directly to 6 without intermediate representation, or went directly into a massive layer, we lose representational efficiency. Compressing 4,096 down to 256 acts as an **information bottleneck**, distilling the most salient discriminative signals.*
> 3. ***Dropout Regularization ($p=0.4$)***: *During training, 40% of the neurons are randomly set to zero on each forward pass. This forces the remaining neurons to learn independent, redundant defect cues rather than memorizing individual training samples.*
> 4. ***Final Linear Projection ($256 \rightarrow 6$)***: *Maps the 256 distilled features to 6 unnormalized log-probability scores (logits), corresponding to: `Open`, `Short`, `Mousebite`, `Spur`, `Copper`, and `Pin-hole`.*

---

### 3. Architecture Flow Diagram
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

## 🎯 Potential Evaluator Questions & Answers

**Q1: What are "channels" (often casually called layers) and how do they progress from 3 (RGB) up to 256?**  
> *"In deep learning, people sometimes casually say 'layers of the image', but their technical name is **Channels** (or **Feature Maps**). A layer is the mathematical operation (like `Conv2d`), whereas channels are the depth of information passing through it.*  
> 
> *Here is the easiest way to visualize them:*
> * ***Starting with 3 Channels (RGB):***  
>   *Think of the input image as a stack of **3 transparent sheets**: one sheet showing Red intensity, one showing Green, and one showing Blue. That is why the input tensor is shaped `3 × 224 × 224`.*
> * ***Progressing to 32, 64, 128, and 256 Channels:***  
>   *The CNN doesn't stay limited to RGB. Each `Conv2d` layer creates a new stack of specialized inspection sheets (feature maps):*  
>   * **Block 1 (32 channels):** *32 sheets, each acting like a specialized filter lighting up basic physical edges—horizontal copper lines, vertical lines, and 45° corners.*  
>   * **Block 2 (64 channels):** *Combines those lines into 64 sheets of geometric patterns—T-junctions, track bends, and circular via holes.*  
>   * **Block 3 (128 channels):** *128 sheets tracking continuous copper tracks, track isolation, and copper voids.*  
>   * **Block 4 (256 channels):** *256 specialized 'defect detector' sheets. For example, one channel lights up specifically for hairline breaks (Open), another for accidental bridges (Short), and another for edge nibbles (Mousebite).*  
> 
> *In short: we start with 3 basic color channels, and the network converts them into 256 specialized defect-inspection channels."*

---

**Q2: How exactly does ReLU add "non-linearity"?**  
> *"Convolution is purely linear mathematics—it only computes weighted sums ($y = W \cdot x + b$). If we stack multiple convolution layers without an activation function, the entire network collapses mathematically into just one single linear equation ($W_2(W_1 x) = W_{\text{combined}} x$). A purely linear network can only draw straight-line decision boundaries, making it impossible to classify real-world imagery.*  
> 
> *ReLU applies the piecewise rule: $f(x) = \max(0, x)$. It creates a sharp 'kink' or threshold at zero: it passes positive values with a slope of 1, and zeroes out all negative values with a slope of 0. Because it turns on some neurons and turns off others depending on the input, it bends and folds the mathematical decision space. This bending is what allows the network to learn complex non-linear shapes like curved copper tracks, circular via pads, and jagged defect contours."*

---

**Q3: In `MaxPool2d`, how does it make sure it doesn't filter out important defect data?**  
> *"It might seem like throwing away 3 out of 4 pixels in every $2\times2$ grid could discard crucial defect info, but it doesn't for three reasons:*  
> 1. ***Convolution has already highlighted the defect:*** *Before pooling, the `Conv2d` filters slide over the patch. Where a defect occurs (e.g., an abnormal copper bridge or a missing void), the filter produces a strong, high-magnitude activation spike. Normal, uniform copper or substrate produces low or near-zero activations.*  
> 2. ***Max pooling specifically selects the peak:*** *By taking the maximum value $\max(x_1, x_2, x_3, x_4)$, max pooling actively extracts the strongest anomaly signal in that neighborhood and discards the uninformative background.*  
> 3. ***Spatial redundancy:*** *Neighboring pixels in a $2\times2$ area are highly correlated (they belong to the same local edge). Max pooling preserves the presence of the feature while stripping away redundant pixel duplicates."*

---

**Q4: How does adding layers one by one up to 256 channels increase complexity?**  
> *"Adding layers and expanding channel depth ($3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$) increases the model's **representational capacity** and **receptive field**:*  
> 1. ***Receptive field expansion:*** *A neuron in Block 1 only sees a tiny $3\times3$ pixel window. By Block 4, through repeated pooling and convolutions, each neuron's receptive field spans across almost the entire $224\times224$ patch. This allows deeper layers to understand global context rather than isolated pixels.*  
> 2. ***Hierarchical abstraction:*** *Early layers (32 channels) can only represent simple primitives like vertical or horizontal contrast lines. As we add layers and channels, the model combines those lines into corners (64 channels), corners into continuous conductive tracks (128 channels), and tracks into full structural defect concepts (256 channels).*  
> 3. ***Multi-class separation capacity:*** *Distinguishing 6 nuanced defect categories (e.g., separating an irregular 'Mousebite' notch from an 'Open' trace break) requires looking at many different geometric traits simultaneously. Expanding to 256 channels gives the network 256 independent feature detectors to categorize complex anomalies accurately."*

---

**Q5: How does `AdaptiveAvgPool2d` capture defect info by averaging? Doesn't averaging blur or dilute the defect?**  
> *"Averaging would blur features if applied to the raw image, but here it is applied at the very end of Block 4 ($14\times14$ feature map), where it serves a completely different purpose:*  
> 1. ***Features are already detected:*** *By Block 4, the convolutions have already spotted the defect. Each of the 256 channels is an activation map indicating where and how strongly a specific defect pattern is present.*  
> 2. ***Regional summary over a $4\times4$ grid:*** *We don't average the whole image into a single number; we pool it down to a $4\times4$ grid (16 regional zones). Averaging within each zone measures the **density and area** of the defect activation. For example, a tiny pin-hole creates high activation in one small zone, whereas excess copper spans multiple zones.*  
> 3. ***Noise filtering:*** *Averaging prevents a single rogue outlier pixel from dominating the classification, giving a stable regional summary of the defect signature."*

---

**Q6: What exactly are "6 unnormalized log-probability scores (logits)" and how do they correspond to the output labels?**  
> *"The final dense layer `Linear(256, 6)` outputs 6 raw numbers—for example: `[1.2, -0.8, 5.4, -2.1, 0.3, -1.0]`.*  
> * ***What is a 'logit'?*** *It is an unconstrained, unnormalized score produced directly by the linear formula $y = Wx + b$. It can be any real number (positive, negative, or zero). It is called a 'log-probability' or logit because it represents the logarithm of unnormalized odds.*  
> * ***How it maps to defect classes:*** *The 6 positions correspond in strict alphabetical order to our 6 dataset defect classes:*  
>   * `Index 0` $\rightarrow$ **Copper**  
>   * `Index 1` $\rightarrow$ **Mousebite**  
>   * `Index 2` $\rightarrow$ **Open**  
>   * `Index 3` $\rightarrow$ **Pin-hole**  
>   * `Index 4` $\rightarrow$ **Short**  
>   * `Index 5` $\rightarrow$ **Spur**  
> * ***How it turns into a prediction:***  
>   1. *Applying the **Softmax function** turns these 6 raw scores into valid probabilities that add up to 100%: $P(\text{class } i) = \frac{e^{z_i}}{\sum e^{z_j}}$.*  
>   2. *During testing/inference, the model takes `argmax(logits)`—picking the index with the highest score. In the example above, Index 2 has the highest score ($5.4$), so the model predicts **Open**."*

---

**Q7: What happens if you remove the `AdaptiveAvgPool2d` and just flatten directly after Block 4?**  
> *"Directly flattening the $14 \times 14 \times 256$ tensor yields $50,176$ features. Connecting $50,176$ to 256 would require $12.8 \text{ million}$ parameters in just one dense layer—an 800% increase in model size—which would severely overfit our dataset and slow down inference. Adaptive pooling to $4 \times 4$ keeps the parameter count to a lightweight 1.43M."*

---

**Q8: Why use `Dropout(0.4)` instead of a higher rate like 0.7 or lower like 0.1?**  
> *"A dropout rate of 0.4 provides an optimal balance: it deactivates enough neurons to prevent co-adaptation without causing underfitting or destabilizing gradient flow during training."*

---

**Q9: Why did you use Batch Normalization before ReLU and not after?**  
> *"Placing BatchNorm directly after the linear convolution ensures that the activations entering the non-linear ReLU function are properly centered around zero with unit variance, maximizing the gradient throughput of ReLU and avoiding dead neuron regimes."*
