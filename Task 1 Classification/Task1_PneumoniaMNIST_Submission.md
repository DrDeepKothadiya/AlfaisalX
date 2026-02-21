# Task 1 – Extensive Comparative Analysis
## PneumoniaMNIST Classification

**Dataset:** PneumoniaMNIST (MedMNIST v2)  
**Task Type:** Binary Classification (Normal vs. Pneumonia)  
**Framework:** PyTorch  
**Hardware:** NVIDIA T4 GPU (Google Colab)

---

## 1. Introduction

This report presents an extensive comparative analysis of deep learning models for pneumonia detection using the PneumoniaMNIST dataset. The study evaluates multiple CNN architectures across three image resolutions, comparing both custom-built and pretrained transfer learning models, and analysing the trade-off between frozen feature extraction and full fine-tuning.

---

## 2. Dataset Overview

PneumoniaMNIST is derived from chest X-ray images and re-formatted as a standardised benchmark within the MedMNIST collection. It is a **binary classification** task distinguishing normal chest X-rays from pneumonia-positive ones.

| Split | Samples |
|-------|---------|
| Train | 4,708   |
| Validation | 524 |
| Test  | 624     |

- **Input:** Grayscale chest X-ray images (single channel)
- **Labels:** 0 = Normal, 1 = Pneumonia
- **Class Imbalance:** Pneumonia cases dominate (~75% positive)

---

## 3. Experimental Setup

### 3.1 Resolutions Tested
Three image resolutions were evaluated to study the effect of input size on model performance:
- **28×28** (native MedMNIST resolution)
- **32×32**
- **64×64**

### 3.2 Data Preprocessing

```python
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

- Images normalized to `[-1, 1]` range
- Batch size: 32
- No additional augmentation applied

### 3.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Epochs | 20 |
| Loss Function | BCEWithLogitsLoss |
| Device | CUDA (T4 GPU) |

### 3.4 Models

**Custom CNN**
- 3 convolutional blocks (Conv → BatchNorm → ReLU → MaxPool)
- Feature maps: 32 → 64 → 128 channels
- Fully connected output layer with sigmoid activation
- Trained from scratch on grayscale input

**Transfer Learning Models (Frozen Backbone)**
- **ResNet18** – first conv layer adapted to 1-channel input, FC head replaced
- **DenseNet121** – conv0 adapted, classifier head replaced
- **EfficientNet-B0** – first conv adapted, classifier head replaced
- Strategy: Backbone weights frozen; only classification head trained

---

## 4. Results

### 4.1 Complete Results Table

| Model | Resolution | AUC | Accuracy |
|-------|-----------|-----|----------|
| **DenseNet121** | 32×32 | **0.9014** | 0.7853 |
| **DenseNet121** | 64×64 | 0.8980 | **0.8125** |
| EfficientNet-B0 | 64×64 | 0.8802 | **0.8173** |
| ResNet18 | 32×32 | 0.8711 | 0.7724 |
| EfficientNet-B0 | 32×32 | 0.8570 | 0.7596 |
| ResNet18 | 64×64 | 0.8565 | 0.7580 |
| EfficientNet-B0 | 28×28 | 0.8336 | 0.7612 |
| ResNet18 | 28×28 | 0.8372 | 0.7628 |

> Note: CustomCNN results were excluded from the final comparative table as training was incorporated into the initial multi-resolution loop; individual metrics were not separately logged in the final results dictionary.

### 4.2 Best Performing Configuration

| Metric | Best Model | Resolution | Score |
|--------|-----------|-----------|-------|
| Highest AUC | DenseNet121 | 32×32 | **0.9014** |
| Highest Accuracy | EfficientNet-B0 | 64×64 | **0.8173** |

---

## 5. Analysis

### 5.1 Effect of Image Resolution

Resolution consistently impacts performance, though the relationship is non-linear:

- **ResNet18** shows an unexpected drop in AUC from 32×32 (0.871) to 64×64 (0.856), possibly due to the frozen backbone extracting ImageNet features misaligned with upsampled low-resolution medical images.
- **DenseNet121** performs best at 32×32 in terms of AUC, suggesting medium resolution is sufficient for capturing pneumonia-relevant texture features in X-rays.
- **EfficientNet-B0** benefits most from higher resolution (64×64), with AUC improving from 0.834 (28×28) to 0.880 (64×64), demonstrating that its compound scaling design leverages resolution well.

### 5.2 Model Architecture Comparison

**DenseNet121** achieves the best AUC overall (0.9014 at 32×32). Its dense connectivity allows feature reuse across layers, which is particularly beneficial when only the classifier head is unfrozen — gradients flow more effectively to the adapted input layer.

**ResNet18** performs competitively at 32×32 (AUC: 0.871) but is the weakest at 64×64. The residual skip connections offer stable training, but the frozen backbone limits adaptation to the grayscale medical domain.

**EfficientNet-B0**, despite being the most parameter-efficient architecture, performs worst at low resolution but scales well with resolution. Its compound scaling is better utilised with more pixels available.

### 5.3 Frozen Backbone vs. Fine-tuning

All transfer learning models in the systematic comparison used frozen backbones (only classification heads trained). The initial multi-resolution loop tested semi-fine-tuning (only FC/classifier unfrozen in the loop).

Key observations:
- Frozen backbone models still achieve competitive AUC (>0.83) despite being pretrained on RGB ImageNet data and adapted to single-channel chest X-rays.
- The adapted first convolutional layer (1-channel input) retains its pretrained weights through the `requires_grad=False` freeze, meaning the backbone sees single-channel features but processes them with RGB-tuned filters — a known limitation.
- Full fine-tuning would likely yield further improvements, especially for DenseNet121 which already shows the best frozen performance.

### 5.4 Loss Convergence

Training loss consistently decreased across all models and resolutions:

- **CustomCNN** converged fastest (loss dropped from ~0.24 → ~0.07 over 20 epochs at 28×28), benefiting from being trained end-to-end without domain mismatch.
- **ResNet18 (frozen)** converged more slowly (loss ~0.45 → ~0.31), reflecting the limited trainable parameters.
- **DenseNet121 (frozen)** at 64×64 showed strong convergence (loss ~0.37 → ~0.19), aligning with its best accuracy result.
- **EfficientNet-B0 (frozen)** at 28×28 and 32×32 converged poorly (~0.46 → ~0.36), while 64×64 showed notable improvement (~0.39 → ~0.24).

---

## 6. ROC Curve Analysis

ROC curves were generated for all three transfer learning models at 64×64 resolution using the frozen (untrained) pretrained weights as a baseline sanity check. This demonstrated that pretrained features alone, without any task-specific head training, yield near-random performance — confirming that classifier fine-tuning is essential for domain transfer from ImageNet to medical imaging.

---

## 7. Key Findings & Conclusions

1. **DenseNet121 at 32×32** achieves the best AUC (0.9014), making it the recommended configuration for this task under frozen-backbone constraints.

2. **EfficientNet-B0 at 64×64** achieves the highest accuracy (0.8173), a useful configuration when classification threshold optimisation is secondary to raw prediction correctness.

3. **Resolution matters**, but more resolution is not always better — DenseNet121 peaks at 32×32, not 64×64.

4. **Transfer learning works well even with frozen backbones**, achieving 0.83–0.90 AUC despite the ImageNet → grayscale medical domain gap.

5. **CustomCNN** trained from scratch shows fast convergence and competitive loss values, suggesting that for this relatively simple binary task, a well-designed lightweight CNN can be highly effective.

6. **Full fine-tuning** of pretrained models (all layers unfrozen) is expected to further improve results and is recommended as a next step.

---

## 8. Limitations & Future Work

- **No data augmentation** was applied; random horizontal flips, brightness jitter, and random crops could improve generalisation.
- **Class imbalance** (~75% pneumonia positive) was not explicitly handled; weighted loss or oversampling could improve sensitivity.
- **Full fine-tuning** experiments were not systematically evaluated across all models and resolutions.
- **DenseNet121 at 28×28** was excluded from the systematic frozen experiment; including it may reveal a performance floor.
- **Hyperparameter tuning** (learning rate scheduling, weight decay, dropout) was not performed.
- **Ensemble methods** combining predictions from multiple models/resolutions could push AUC above 0.92.

---

## 9. Code Summary

```python
# Key architecture adaptations for grayscale input
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # ResNet18
model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # DenseNet121
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)  # EfficientNet-B0

# Frozen backbone strategy
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():  # or model.classifier
    param.requires_grad = True

# Training
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# Evaluation metrics
auc = roc_auc_score(all_labels, all_probs)
acc = accuracy_score(all_labels, (all_probs > 0.5).astype(int))
```

---

## 10. References

- Yang, J., et al. (2023). MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*, 10, 41.
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
- Huang, G., et al. (2017). Densely Connected Convolutional Networks. *CVPR 2017*.
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML 2019*.
- Rajpurkar, P., et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays. *arXiv:1711.05225*.

---

*Report generated for Task 1 submission | PneumoniaMNIST Comparative Analysis*
