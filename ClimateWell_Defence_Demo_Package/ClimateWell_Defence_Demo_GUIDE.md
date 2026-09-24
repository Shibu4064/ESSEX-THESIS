# ClimateWell Defence Demo — Kaggle / Colab Run Guide

## What this package does

The notebook reconstructs and compares the two final candidate systems from the MSc dissertation:

- **BGE + TF-IDF classical hierarchical model**
- **ClimateWell-FuseNet V2**

It then creates a real Gradio demonstration using **genuine unseen records from the 10,176-record master corpus**.

The strict demo pool excludes:
- 1,720 labelled development records
- 500 audit records

This leaves **7,956 master records that are neither training nor audit examples**.

---

## Recommended platform: Kaggle

### 1. Create the notebook
Upload `ClimateWell_Defence_RealTime_Demo.ipynb` to Kaggle.

### 2. Enable hardware
In notebook settings:
- Accelerator: **GPU** (P100/T4 or better)
- Internet: **ON** for the first run

Internet is required to download:
- `BAAI/bge-small-en-v1.5`
- `allenai/specter`

After the model files are cached by Kaggle/Hugging Face, later runs are easier.

### 3. Add the three Excel files
Add:
- `Human labelled_DTU.xlsx`
- `Master file_10k papers.xlsx`
- `Test_sampled_data.xlsx`

The notebook searches `/kaggle/input/**` automatically, so the dataset folder name does not matter.

### 4. Run cells in order
Run from top to bottom.

The first important integrity output must show:

- Training rows = **1,720**
- Reject = **1,521**
- Accept = **199**
- Accepted rows with valid theme = **198**
- Master = **10,176**
- Audit = **500**
- Training ∩ Audit = **0**
- Unseen demo pool = **7,956**

If these numbers are different, stop and do not use the demo in the defence.

---

## Important: verify the audit columns once

The provided audit file contains these legacy columns:

- `pred_decision`
- `pred_theme`

There are no separately named human-label columns.

The final dissertation describes the audit as **450 Reject / 50 Accept**, and the uploaded workbook has exactly that distribution in `pred_decision`. However, the notebook deliberately does not assume that a column called `pred_decision` is human truth.

If you know that these are the manually verified audit labels, change:

```python
AUDIT_LABELS_VERIFIED = False
```

to:

```python
AUDIT_LABELS_VERIFIED = True
```

Then the notebook will calculate a true head-to-head external audit comparison for both models.

If you are not certain, leave it `False`. The prototype still works and uses the final dissertation policy:

**BGE + TF-IDF = primary**  
**ClimateWell-FuseNet V2 = challenger / uncertainty model**

This is the scientifically safer setting.

---

## Full run versus quick run

For the actual defence:

```python
QUICK_MODE = False
```

This uses five Stage-1 folds and the full FuseNet training schedule.

For debugging the interface only:

```python
QUICK_MODE = True
```

Do not quote QUICK_MODE scores in the defence.

---

## What takes the most time

The first run creates and caches:

- BGE embeddings for training, audit and master data
- SPECTER embeddings for training, audit and master data

After that, rerunning the notebook is much faster because `.npy` caches are reused.

FuseNet five-fold training is the second most expensive stage.

For the defence, run the full notebook once beforehand and keep the Kaggle/Colab session alive.

---

## What the model comparison means

The notebook reports internal cross-validation for both candidates.

If the audit labels are verified, it additionally reports:
- external Average Precision
- external Accept F1
- external binary macro-F1
- external theme macro-F1
- predicted audit Accept rate

The Gradio prototype never uses the audit to fit text vectorisers, model parameters or thresholds.

---

## What is saved

The notebook saves:

- `model_comparison.csv`
- `audit_model_predictions.csv`
- `unseen_master_predictions.csv`
- `defence_examples.csv`
- reusable classical model bundle
- FuseNet state dictionary
- FuseNet preprocessing objects
- deployment policy JSON
- run summary JSON

and packages them into:

`ClimateWell_Defence_Demo_Outputs.zip`

---

## How to demonstrate it during the viva

Use the Gradio dropdown in this order.

### Example 1 — High-confidence ACCEPT
Click:
**Load Unseen Master Record → Analyse Record**

Explain:
> “Stage 1 has accepted this genuinely unseen title–abstract record. Stage 2 is therefore activated and assigns a well-being theme.”

### Example 2 — High-confidence REJECT
Explain:
> “Because Stage 1 rejects the record, no theme is assigned. This preserves the conditional structure of the human annotation process.”

### Example 3 — Model disagreement
Explain:
> “Here the classical primary model and FuseNet challenger disagree. Instead of hiding this uncertainty, the system raises the record for human review.”

This is particularly useful for defending the human-in-the-loop contribution.

---

## Suggested wording if asked: “Which model is running?”

> “The primary deployment configuration is BGE plus TF-IDF unless a verified external head-to-head audit changes that selection. ClimateWell-FuseNet V2 runs in parallel as a challenger model and provides instance-sensitive modality gates and disagreement signals.”

---

## Suggested wording if asked: “Are these real examples?”

> “Yes. The demonstration pool is created programmatically from the 10,176-record master corpus after removing all 1,720 labelled development records and all 500 audit records, leaving 7,956 unseen records.”

---

## Before the defence

Do these four checks:
1. Run the notebook fully at least once.
2. Save/download `ClimateWell_Defence_Demo_Outputs.zip`.
3. Open the Gradio app and test all three showcase example types.
4. Keep screenshots of one Accept, one Reject and one disagreement case in case the live internet/share link fails.

The core prediction code still runs inside the notebook even if the public Gradio share link is unavailable.