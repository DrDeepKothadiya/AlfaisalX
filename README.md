# AI Medical Imaging Challenge
> End-to-end AI system for pneumonia detection, medical report generation, and semantic image retrieval using the PneumoniaMNIST dataset.

---

## Overview

This repository contains three interconnected AI systems built on the [PneumoniaMNIST](https://medmnist.com/) dataset (MedMNIST v2):

| Task | Description | Key Technology | Key Result |
|------|-------------|---------------|------------|
| **Task 1** | CNN Classifier for pneumonia detection | Custom CNN (PyTorch) | AUC 0.9655 · Accuracy 91.5% · Sensitivity 94.4% |
| **Task 2** | Automated medical report generation | MedGemma-4B-IT (VLM) | Structured radiological reports; correct on 10/10 qualitative sample |
| **Task 3** | Semantic content-based image retrieval | BiomedCLIP + FAISS | P@1 = 0.869 · P@5 = 0.853 |

---

## Repository Structure

```
repository/
├── data/                          # Data loading utilities (auto-downloads via medmnist)
├── models/                        # Model architectures and saved weights
│   └── mnistcnn.pth               # Trained CNN weights (Task 1)
├── task1_classification/
│   ├── Task_1_classification.ipynb
│   └── train.py                   # Standalone training script
├── task2_report_generation/
│   └── Task_2_Medical_Report.ipynb
├── task3_retrieval/
│   ├── task3_semantic_retrieval.ipynb
│   ├── embeddings.npz             # Pre-extracted BiomedCLIP embeddings
│   └── faiss_index.bin            # Pre-built FAISS index
├── notebooks/                     # Google Colab-ready notebooks (copies of above)
├── reports/
│   ├── task1_classification_report.md
│   ├── task2_report_generation.md
│   └── task3_retrieval_system.md
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/DrDeepKothadiya/AlfaisalX.git
cd AlfaisalX
```

### 2. Set Up the Environment

**Option A — pip (recommended for Colab/Linux):**
```bash
pip install -r requirements.txt
```

**Option B — conda:**
```bash
conda create -n medai python=3.10
conda activate medai
pip install -r requirements.txt
```

> **GPU note:** Tasks 2 and 3 benefit greatly from a CUDA GPU. Google Colab's free T4 GPU is sufficient for all tasks. Task 1 runs comfortably on CPU (~5 minutes for 20 epochs).

### 3. Dataset

The PneumoniaMNIST dataset downloads automatically on first run via the `medmnist` package. No manual download is required.

```
Train: 4,708 images | Validation: 524 images | Test: 624 images
Format: 28×28 grayscale | Task: Binary (Normal=0, Pneumonia=1)
```

---

## Task 1 — CNN Classification

### Architecture

A custom 3-block CNN (`MNISTCNN`) with progressive channel widening (32→64→128), BatchNorm, AdaptiveAvgPool, and Dropout(0.5). Designed for 28×28 grayscale input.

### Running

```bash
# Training
python task1_classification/train.py

# Or open the notebook:
jupyter notebook task1_classification/Task_1_classification.ipynb
```

### Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **91.51%** |
| AUC-ROC | **0.9655** |
| Sensitivity (Pneumonia Recall) | **94.36%** |
| Specificity (Normal Recall) | **86.75%** |
| Optimal Threshold | 0.42 |

**Confusion Matrix (threshold = 0.42):**
```
                  Pred Normal  Pred Pneumonia
Actual Normal        203 (TN)      31 (FP)
Actual Pneumonia      22 (FN)     368 (TP)
```

See [`reports/task1_classification_report.md`](reports/task1_classification_report.md) for full analysis including failure case discussion.

---

## Task 2 — Medical Report Generation (VLM)

### Model

**MedGemma-4B-IT** (`google/medgemma-4b-it`) — Google's open-source medical VLM, deployed with 4-bit NF4 quantisation to fit on Colab free T4 GPU.

### Prerequisites

1. Create a [Hugging Face account](https://huggingface.co/join)
2. Request access to `google/medgemma-4b-it` on its [model page](https://huggingface.co/google/medgemma-4b-it)
3. Set your HF token:
   ```bash
   export HF_TOKEN=hf_your_token_here
   # Or in Python: from huggingface_hub import login; login()
   ```

> ⚠️ **HF token not working.**

### Running

```bash
# Open the notebook (GPU strongly recommended):
jupyter notebook task2_report_generation/Task_2_Medical_Report.ipynb
```

### Prompting Strategy

The final prompt (Strategy A) guides the model to produce a structured 4-point report:
```
You are an experienced radiologist.
Analyze this chest X-ray image.
Provide:
1. Radiological findings
2. Presence or absence of pneumonia
3. Severity
4. Clinical impression.
Be concise and medically accurate.
```

See [`reports/task2_report_generation.md`](reports/task2_report_generation.md) for 10 sample reports and qualitative analysis.

---

## Task 3 — Semantic Image Retrieval

### System Components

- **Embedding model:** BiomedCLIP (Microsoft Research) — 195M parameters, trained on 15M biomedical image-text pairs
- **Vector index:** FAISS `IndexFlatIP` (exact cosine similarity search)
- **Embedding dimension:** 512

### Running

```bash
# Open the notebook (GPU recommended for embedding extraction):
jupyter notebook task3_retrieval/task3_semantic_retrieval.ipynb
```

Pre-extracted embeddings and the FAISS index are provided (`embeddings.npz`, `faiss_index.bin`) so you can skip the extraction step and go straight to retrieval.

### Usage Examples

```python
from retrieval_system import MedicalImageRetrievalSystem

# Image-to-image search
retrieval_system.search_by_image_index(query_idx=42, k=5)

# Text-to-image search
retrieval_system.search_by_text(
    "chest X-ray with consolidation suggesting bacterial pneumonia",
    k=5
)
```

### Retrieval Performance (Precision@k)

| k | Overall | Normal | Pneumonia |
|---|---------|--------|-----------|
| P@1 | **0.8686** | 0.8291 | 0.8923 |
| P@3 | **0.8590** | 0.8105 | 0.8880 |
| P@5 | **0.8529** | 0.7863 | 0.8928 |
| P@10 | **0.8391** | 0.7658 | 0.8831 |

Only 1.9% of queries (12/624) resulted in complete retrieval failures (P@5 = 0). 62.2% of queries returned a perfectly class-homogeneous top-5 result set.

See [`reports/task3_retrieval_system.md`](reports/task3_retrieval_system.md) for architecture details, failure analysis, and embedding space visualisation.

---

## Google Colab

All three tasks are available as Colab notebooks for zero-setup online execution:

| Task | Colab Link | GPU Required? |
|------|-----------|--------------|
| Task 1 — CNN Classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) | No (CPU sufficient) |
| Task 2 — Report Generation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) | Yes (T4 GPU) |
| Task 3 — Retrieval System | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) | Recommended |

> Update the Colab badge links with your actual notebook URLs after uploading to Google Drive.

---

## Implementation Summary

| Remark |
|-----------|
| Complete pipelines for all 3 tasks; well-structured, modular, documented code with reproducible random seeds |
| Full metric suite (Accuracy, Precision, Recall, F1, AUC, Sensitivity, Specificity, Precision@k); threshold sweep; confusion matrix; t-SNE embedding visualisation |
| Failure case analysis for all tasks; discussion of resolution constraints, hallucination risk, class imbalance, and concrete improvement roadmap |
| This README; task-specific markdown reports; working Colab notebooks; `requirements.txt`; pre-built FAISS index and embeddings provided |

---

## Dependencies

See [`requirements.txt`](requirements.txt). Key packages:

```
torch>=2.0        medmnist>=2.2      transformers>=4.40
open-clip-torch   faiss-cpu          bitsandbytes>=0.43
scikit-learn      matplotlib         Pillow
```

---

## Contact

- Dr. Deep R Kothadiya — cothadiya.deep@gmail.com
