# Task 2: Medical Report Generation — Visual Language Model Report
**Dataset:** PneumoniaMNIST (MedMNIST v2) | **Model:** MedGemma-4B-IT (Google)

---

## 1. Model Selection Justification

**MedGemma-4B-IT** (`google/medgemma-4b-it`) was selected as the primary VLM for the following reasons:

- **Domain specificity:** MedGemma is explicitly trained on medical imaging data and clinical text, making it far more likely to produce clinically relevant radiological descriptions compared to general-purpose VLMs (e.g., standard LLaVA, GPT-4V).
- **Open-source and reproducible:** Available freely on Hugging Face under a permissive licence — consistent with open research principles and reproducibility requirements.
- **Instruction-tuned variant:** The `-it` suffix denotes instruction tuning, enabling structured prompting for report generation (numbered findings, severity grading, clinical impressions).
- **Efficient deployment:** With 4-bit quantisation (NF4 via BitsAndBytes), the 4B parameter model fits within Colab's free T4 GPU (15 GB VRAM), making the pipeline accessible without paid compute.
- **Alternatives considered:** LLaVA-Med was evaluated but is less actively maintained and produces less structured outputs for chest X-ray tasks. BioViL-T is primarily an encoder model without text generation capability.

### Model Configuration

```python
model_id = "google/medgemma-4b-it"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
# Images resized to 224×224 RGB before passing to processor
```

---

## 2. Prompting Strategies

Two prompting strategies were designed and evaluated. Both use the chat-template format expected by MedGemma's instruction tuning.

### Strategy A — Minimal Structured Prompt

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

**Rationale:** Short prompts reduce the risk of the model hallucinating verbose but inaccurate text. The numbered structure forces the model to address each clinical axis in a predictable order, making outputs easier to evaluate and compare across images.

### Strategy B — Verbose Diagnostic Prompt (tested, not adopted)

```
You are an experienced radiologist reviewing a chest radiograph.
Describe in detail: lung field clarity, presence of infiltrates, consolidation,
air bronchograms, pleural effusion, cardiomegaly, and any other abnormalities.
Then provide a structured diagnosis with severity (Mild/Moderate/Severe) and
recommended management plan.
```

**Why Strategy A was preferred:** Strategy B frequently produced over-generated outputs with hallucinated findings not present in the low-resolution 28×28 images (e.g., describing "bilateral hilar lymphadenopathy" which is unresolvable at this scale). Strategy A's brevity kept the model grounded in observable image features.

### Generation Parameters

| Parameter | Value | Reason |
|---|---|---|
| `max_new_tokens` | 200 | Sufficient for a 4-point clinical summary |
| `min_new_tokens` | 50 | Prevents truncated one-line outputs |
| `do_sample` | False | Greedy decoding for deterministic, reproducible outputs |
| `repetition_penalty` | 1.1 | Reduces repetitive phrase loops common in instruction-tuned models |

---

## 3. Sample Generated Reports

Ten representative images were selected: 4 confirmed normal cases, 4 confirmed pneumonia cases, and 2 images misclassified by the Task 1 CNN (one false positive and one false negative).

---

### Image 1 — Ground truth: Pneumonia | CNN: Pneumonia ✓

**MedGemma Report:**
> *Radiological findings: Increased opacity in the right lower lobe with ill-defined margins, consistent with consolidation. No pleural effusion identified.*
> *Pneumonia: Present.*
> *Severity: Moderate.*
> *Clinical impression: Right lower lobe pneumonia. Recommend clinical correlation and follow-up chest X-ray in 4–6 weeks.*

**Analysis:** Aligns correctly with ground truth. The model identifies consolidation and assigns moderate severity, which is consistent with typical MedMNIST pneumonia cases.

---

### Image 2 — Ground truth: Normal | CNN: Normal ✓

**MedGemma Report:**
> *Radiological findings: Lung fields appear clear and well-aerated bilaterally. No evidence of consolidation, infiltrates, or pleural effusion. Cardiac silhouette within normal limits.*
> *Pneumonia: Absent.*
> *Severity: N/A.*
> *Clinical impression: Normal chest radiograph.*

**Analysis:** Correctly identifies normal case. Output is appropriately concise and avoids over-reporting.

---

### Image 3 — Ground truth: Pneumonia | CNN: Pneumonia ✓

**MedGemma Report:**
> *Radiological findings: Patchy bilateral opacities predominantly in the lower lung zones. Air space disease present.*
> *Pneumonia: Present.*
> *Severity: Moderate to Severe.*
> *Clinical impression: Bilateral pneumonia with significant air space consolidation. Urgent clinical evaluation recommended.*

**Analysis:** Correct label. Bilateral involvement detection aligns with ground truth.

---

### Image 4 — Ground truth: Normal | CNN: Normal ✓

**MedGemma Report:**
> *Radiological findings: Clear lung fields. No infiltrates or consolidation identified. Normal vascular markings.*
> *Pneumonia: Absent.*
> *Severity: N/A.*
> *Clinical impression: No acute cardiopulmonary findings.*

**Analysis:** Correct. Clean, clinically appropriate report.

---

### Image 5 — Ground truth: Pneumonia | CNN: Pneumonia ✓

**MedGemma Report:**
> *Radiological findings: Hazy opacity in the left lower zone suggesting consolidation. Right lung relatively clear.*
> *Pneumonia: Present.*
> *Severity: Mild to Moderate.*
> *Clinical impression: Left lower zone pneumonia. Clinical correlation advised.*

**Analysis:** Correct label; lateralisation (left lower zone) is a clinically plausible description.

---

### Image 6 — Ground truth: Normal | CNN: Normal ✓

**MedGemma Report:**
> *Radiological findings: Both lung fields are clear. No abnormal density or opacity detected.*
> *Pneumonia: Absent.*
> *Severity: N/A.*
> *Clinical impression: Normal.*

**Analysis:** Correct and concise. No hallucination.

---

### Image 7 — Ground truth: Pneumonia | CNN: Pneumonia ✓

**MedGemma Report:**
> *Radiological findings: Dense consolidation in right middle and lower lobes with associated volume loss.*
> *Pneumonia: Present.*
> *Severity: Severe.*
> *Clinical impression: Severe right-sided pneumonia. Immediate medical attention recommended.*

**Analysis:** Correct. Severity escalation to "Severe" is plausible for dense consolidation patterns.

---

### Image 8 — Ground truth: Normal | CNN: Normal ✓

**MedGemma Report:**
> *Radiological findings: Lung fields clear. Cardiac size normal.*
> *Pneumonia: Absent.*
> *Severity: N/A.*
> *Clinical impression: No significant findings.*

**Analysis:** Correct.

---

### Image 9 — Ground truth: Pneumonia | CNN: Normal ✗ (CNN False Negative)

**MedGemma Report:**
> *Radiological findings: Subtle increased density in the right lower zone. Lung fields otherwise appear relatively clear.*
> *Pneumonia: Possible early pneumonia. Findings are subtle.*
> *Severity: Mild.*
> *Clinical impression: Cannot exclude early or mild pneumonia. Clinical correlation strongly recommended.*

**Analysis:** Interestingly, MedGemma flagged this as "possible early pneumonia" — hedging toward the correct label even though the CNN confidently predicted Normal. This demonstrates that the VLM's broader medical pretraining helps detect subtle patterns the CNN missed. This case represents the most clinically valuable VLM contribution.

---

### Image 10 — Ground truth: Normal | CNN: Pneumonia ✗ (CNN False Positive)

**MedGemma Report:**
> *Radiological findings: Increased vascular markings in bilateral lower zones. No frank consolidation identified. Possible mild atelectasis.*
> *Pneumonia: Unlikely. Vascular prominence may simulate early infiltrate.*
> *Severity: N/A.*
> *Clinical impression: No definitive pneumonia. Findings more consistent with vascular congestion.*

**Analysis:** MedGemma correctly identifies the normal case that the CNN misclassified, and provides a clinically meaningful explanation (vascular congestion mimicking infiltrates). This demonstrates the VLM's potential as a second-opinion mechanism.

---

## 4. Qualitative Analysis

### Alignment with Ground Truth

| Category | Images | MedGemma Correct | Notes |
|---|---|---|---|
| True Positives (CNN ✓, GT: Pneumonia) | 4 | 4/4 | Consistent pneumonia detection |
| True Negatives (CNN ✓, GT: Normal) | 4 | 4/4 | Clean normal reports |
| CNN False Negative (GT: Pneumonia) | 1 | 1/1 (hedged) | VLM hedged correctly toward pneumonia |
| CNN False Positive (GT: Normal) | 1 | 1/1 | VLM correctly overrode CNN |

MedGemma achieved 10/10 correct (or clinically appropriate hedged) classifications in this qualitative sample — **outperforming the CNN on the two difficult cases**.

### Key Observations

1. **Complementary to CNN:** The VLM shows particular strength on the edge cases where the CNN fails, suggesting a hybrid pipeline (CNN for speed, VLM for flagged uncertain cases) would be clinically valuable.
2. **Severity grading is reasonable but unverifiable:** MedGemma assigned severity labels (Mild/Moderate/Severe) consistently, but PneumoniaMNIST does not provide ground truth severity, so these cannot be validated quantitatively.
3. **Resolution awareness:** When prompting asked for highly specific findings (e.g., "describe air bronchograms"), the model occasionally hallucinated features. Strategy A's focus on high-level presence/absence and severity kept outputs grounded.
4. **Lateralisation:** The model sometimes specified laterality (e.g., "right lower lobe") for pneumonia cases. While plausible, this cannot be verified at 28×28 resolution and should be treated as uncertain.

---

## 5. Strengths and Limitations

**Strengths:**
- Produces structured, clinically coherent reports without fine-tuning.
- Shows complementary failure modes to the CNN — particularly useful on ambiguous cases.
- 4-bit quantised deployment enables free Colab T4 GPU execution.
- Deterministic outputs (greedy decoding) ensure reproducibility.

**Limitations:**
- **Input resolution mismatch:** MedGemma was trained on full-resolution radiographs; providing 28×28 images (upsampled to 224×224) introduces significant blurring artefacts that may confound the model.
- **Hallucination risk:** On ambiguous images, the model occasionally generates findings not observable at the available resolution.
- **No quantitative evaluation:** Metrics like BERTScore or BLEU against reference reports are not available without a radiologist-annotated report dataset.
- **Gated model access:** Requires a Hugging Face account with gated model approval, adding a setup barrier for reproducibility.
- **Clinical deployment gap:** Even state-of-the-art VLMs should not replace radiologist review — these outputs serve as decision support, not diagnosis.
