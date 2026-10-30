# PCB Defect Inspection — Presentation Guide: Part 1
**Speaker:** Vedant  
**Topic:** Problem Statement, Industrial Gap, Solution, Dataset Deep-Dive & Preprocessing Pipeline

---

## 🎙️ Spoken Presentation Script

### 1. Opening & Problem Statement
> *"Good morning, our project focuses on **Automated PCB Defect Inspection Using Deep Learning**.*
>
> *PCB is the backbone of modern electronics manufacturing. Today’s PCBs feature ultra-dense layouts with ultra-thin copper traces. Even a microscopic structural defect on a copper trace causes circuit failures or catastrophic short circuits.*
>
> *Currently, the industry faces a major bottleneck in quality control:*
> 1. *First, **Manual Visual Inspection**: Human inspectors working under microscopes suffer cognitive fatigue, leading to a defect escape rate of 15% to 30%.*
> 2. *Second, **Traditional Automated Optical Inspection (AOI)**: Conventional AOI systems rely on template matching and pixel subtraction. They are hypersensitive to slight lighting variations, causing high false alarm rates exceeding.*
>
> *To bridge this gap, our solution implements an end-to-end deep learning computer vision pipeline. Instead of pixel-subtraction, we treat defect detection as a visual feature learning problem. We ingest copper trace patches and classify them across six critical manufacturing defect categories."*

---

### 2. Dataset Overview
> *"We have utilized the **DeepPCB[pronounced as'Deep' + 'PCB'] Benchmark Dataset**, released by Peking University’s Intelligent Hardware Laboratory.*
>
> *DeepPCB featuers:*
> * *It contains **1,500 pairs of aligned high-resolution optical images** (approximately 640×640 pixels) captured using industrial-grade PCB scanners.*
> * *Each pair contains a **defect-free template** image (`*_temp.jpg`) and a **damaged test board** (`*_test.jpg`) containing genuine manufacturing anomalies.*
> * *The annotations provide bounding box coordinates `[x1, y1, x2, y2]` for the defects along with a class identifier for the six defect categories:*
>   1. ***Open***: *An unintended break or physical gap along a conductive trace.*
>   2. ***Short***: *An unintended conductive copper bridge spanning between adjacent traces.*
>   3. ***Mousebite***: *An irregular edge erosion or bite taken out of the copper boundary.*
>   4. ***Spur***: *A sharp, unwanted copper flare or protrusion extending from a trace.*
>   5. ***Copper (Spurious Copper)***: *An isolated island of unetched excess copper on the substrate.*
>   6. ***Pin-hole***: *A tiny void or hollow cavity internal to a copper track.*

---

### 3. Dataset Preprocessing Pipeline
> *"Raw full-board images cannot be fed directly into standard convolutional neural networks efficiently.
>
> 1. ***Defect Annotations***:  
>    *The dataset provides bounding box coordinates [x1, y1, x2, y2] for each defect in the image. We use these coordinates to crop the region of interest (ROI) from the test board.*
>
> 2. ***Contextual Bounding Box Expansion (15% Padding)***:  
>    *Raw annotation boxes tightly wrap only the defective pixels. If we crop strictly inside the box, the model loses all surrounding structural context. We implemented a 15% margin padding around every bounding box. This enables the network to see surrounding context.*
>
> 3. ***Board-Level Split Isolation (Preventing Data Leakage)***:  
>    *A critical mistake in image classification is splitting cropped defect patches randomly into train and test sets. Since multiple defects occur on the same physical PCB under identical lighting, random patch splitting causes test data leakage. We resolved this by performing `train_test_split` strictly at the **PCB Board level (80% Train, 20% Test)**.*
[The test set consists of 1,967 completely unseen defect patches from isolated boards.
Train Set: 8,046 images (~80.35%)
Test Set: 1,967 images (~19.65%)
Total Cropped Defect Patches: 10,013 images]
>
>
> 4. ***Standardization & Data Augmentation***:  
>    *All patches are resized to a uniform dimension of **224×224 pixels**.*  
>    *For the training split, we applied geometric data augmentations: `RandomHorizontalFlip(p=0.5)`, `RandomVerticalFlip(p=0.5)`, and `RandomRotation(degrees=10)`. This prevents model overfitting.*  
>    *Finally, images are converted to tensors and normalized using ImageNet channel statistics (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).*
>
> *Now, I will hand over to **Aryan**, who will walk you through our Baseline CNN architecture."*

---

## 🎯 Potential Evaluator Questions & Answers

**Q1: Why did you crop defect patches instead of passing the entire 640×640 board to the CNN?**  
> *"Passing full-board images directly to a classification CNN would cause tiny defects (often only 15×15 pixels) to get washed out during successive pooling operations. Cropping localized patches standardizes the receptive field on the actual anomaly while drastically saving computational memory."*

**Q2: What is 'data leakage' in this context, and how did you prevent it?**  
> *"If we crop all patches first and then use `train_test_split`, patches extracted from the exact same PCB board would end up in both training and test sets. Because boards have unique lighting, background etching color, and solder mask tint, the model would cheat by recognizing board-specific background artifacts. We prevented this by splitting at the board annotation level before cropping."*

**Q3: Why did you choose 15% padding specifically?**  
> *"A 15% contextual margin provides sufficient boundary continuity for adjacent traces and substrate edges without incorporating neighboring irrelevant defects into the same crop."*
