# ML-Based Side-Channel Attack on AES-128

> Classical Machine Learning for Side-Channel Analysis: A Systematic Study of Feature Engineering Strategies with SHAP-Based Leakage Localization

**Author:** Krish Jeswal | Electronics & Telecommunication Engineering, RVCE (Batch 2024–2028)  
**GitHub:** [KrishJeswal](https://github.com/KrishJeswal) | **LinkedIn:** [krishjeswal](https://linkedin.com/in/krishjeswal)

---

## What This Project Is

This is an end-to-end machine learning pipeline that performs a **profiling side-channel attack on AES-128 encryption** using power traces from the ASCAD dataset. It has two parallel goals:

1. **Research paper** — A systematic comparison of classical ML classifiers and feature engineering strategies for side-channel analysis, evaluated using Guessing Entropy and SHAP-based leakage localization. Target venue: IEEE ISCAS 2026.

2. **Portfolio project** — A production-grade ML pipeline following modular software engineering principles (data ingestion → preprocessing → training → evaluation → deployment).

### The Core Idea

When a microcontroller runs AES-128 encryption, its power consumption leaks information about the secret key. By recording thousands of these power measurements (traces) and training ML models to detect patterns, an attacker can statistically recover the encryption key — without ever breaking the mathematics of AES.

---

## Novel Research Contributions

**Contribution 1 — Systematic POI × Model Grid:**
Full comparison of 6 classifiers × 3 POI extraction strategies × 4 values of k, evaluated by Guessing Entropy (not just accuracy).

**Contribution 2 — SHAP Leakage Localization:**
SHAP (SHapley Additive exPlanations) applied to identify which time samples in a power trace leak key information, producing an interpretable leakage map overlaid with the SNR plot.

**Contribution 3 — Leakage Model Comparison:**
Systematic comparison of Hamming Weight (9 classes) vs Identity (256 classes) labeling schemes across classifiers, with GE curves and NtD as evaluation metrics.

---

## Dataset — ASCAD (ANSSI)

| Property | Value |
|---|---|
| Source | French National Cybersecurity Agency (ANSSI) |
| Device | ATMega8515 microcontroller |
| Implementation | Masked AES-128 (first-order Boolean masking) |
| Profiling traces | 50,000 |
| Attack traces | 10,000 |
| Samples per trace | 700 (windowed around SubBytes for byte 2) |
| Target | Key byte at index 2 |
| Download | [data.gouv.fr](https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip) |

> **Note:** The zip contains raw traces (`ATMega8515_raw_traces.h5`, ~5.6 GB). Run `generate_ascad.py` after downloading to produce the processed file used by the pipeline.

---

## Project Structure

```
side-channel-ml/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py        # Load ASCAD_processed.h5, save splits
│   │   ├── data_transformation.py   # SNR, ANOVA, PCA feature engineering
│   │   └── model_trainer.py         # Train 6 models, save with joblib
│   ├── pipeline/
│   │   ├── train_pipeline.py        # Orchestrates full training flow
│   │   └── predict_pipeline.py      # Load model, predict HW from new trace
│   ├── utils.py                     # GE computation, SHAP helpers
│   ├── logger.py                    # File-based logging
│   └── exception.py                 # Custom exception handler
├── notebooks/
│   ├── 01_EDA.ipynb                 # Trace visualization, label distribution
│   ├── 02_feature_engineering.ipynb # SNR, ANOVA, PCA comparison
│   ├── 03_model_comparison.ipynb    # 6 models × 4 feature strategies
│   └── 04_novel_contribution.ipynb  # GE curves, SHAP, HW vs ID
├── artifacts/
│   ├── raw/                         # ASCAD_processed.h5 (not tracked in git)
│   ├── models/                      # Saved models (not tracked in git)
│   └── logs/                        # Runtime logs (not tracked in git)
├── generate_ascad.py                # One-time: raw traces → processed HDF5
├── verify.py                        # Sanity check: confirms data loads correctly
├── app.py                           # Streamlit deployment app
├── setup.py
└── requirements.txt
```

---

## ML Pipeline

```
ASCAD Raw Traces (5.6 GB)
        ↓  generate_ascad.py
ASCAD Processed (1.7 GB, 700 samples/trace)
        ↓  data_ingestion.py
Load traces + plaintext + key metadata
        ↓  Label generation
HW labels = HW(AES_SBOX[plaintext[2] XOR key[2]])
        ↓  data_transformation.py
POI Extraction: SNR / ANOVA / PCA  (700 → 50 features)
        ↓  model_trainer.py
Train: LR, DT, RF, XGBoost, SVM, MLP
Evaluate: Accuracy, Macro F1, 5-fold CV
        ↓  utils.py
Guessing Entropy computation → key recovery
        ↓  notebooks/04
SHAP leakage localization + HW vs ID comparison
        ↓  app.py
Streamlit: upload trace → predict HW class
```

---

## Models Compared

| Model | Type | Expected Role |
|---|---|---|
| Logistic Regression | Linear | Baseline |
| Decision Tree | Tree | Interpretable reference |
| Random Forest | Ensemble | Strong feature importance |
| XGBoost | Gradient Boosting | Expected best GE |
| SVM (RBF kernel) | Kernel method | SCA literature benchmark |
| MLP | Neural network | Bridge to deep learning |

---

## Feature Engineering Strategies

| Strategy | Method | Type |
|---|---|---|
| Raw traces | No reduction | Baseline |
| SNR-based POI | Signal-to-Noise Ratio | Domain-specific |
| ANOVA F-test | SelectKBest (sklearn) | Statistical |
| PCA | Principal Component Analysis | Unsupervised |

Parameter sweep: k ∈ {20, 50, 100, 200} POIs for SNR and ANOVA.

---

## Evaluation Metrics

| Metric | Used for |
|---|---|
| Accuracy | Standard ML (Table 1) |
| Macro F1 | Primary ML metric — handles class imbalance |
| 5-fold Stratified CV | Generalization check |
| Guessing Entropy (GE) | Primary SCA metric — does the attack recover the key? |
| Number of Traces to Disclosure (NtD) | How many traces needed for GE < 1 |

---

## Setup

### Prerequisites
- Python 3.10+
- ~8 GB free disk space during setup (reducible to ~2 GB after cleanup)

### Installation

```bash
git clone https://github.com/KrishJeswal/side-channel-ml.git
cd side-channel-ml

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -e .
pip install -r requirements.txt
```

### Dataset

```bash
# Download (~2.8 GB zip)
# Windows PowerShell
Invoke-WebRequest -Uri "https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip" -OutFile "artifacts/raw/ASCAD_data.zip"

# Mac/Linux
wget -O artifacts/raw/ASCAD_data.zip https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip

# Unzip
Expand-Archive -Path artifacts/raw/ASCAD_data.zip -DestinationPath artifacts/raw/   # Windows
# unzip artifacts/raw/ASCAD_data.zip -d artifacts/raw/                               # Mac/Linux

# Move the raw traces file (the actual dataset is ATMega8515_raw_traces.h5)
mv artifacts/raw/ASCAD_data/ASCAD_databases/ATMega8515_raw_traces.h5 artifacts/raw/ASCAD.h5
rm -rf artifacts/raw/ASCAD_data artifacts/raw/ASCAD_data.zip

# Generate the processed file (takes ~5 minutes)
python generate_ascad.py

# Verify everything loaded correctly
python verify.py
# Expected: ALL CHECKS PASSED
```

---

## Current Status

| Phase | Status |
|---|---|
| Project structure setup | Done |
| Dataset download + processing | Done |
| Data verification | Done |
| EDA notebook | Done |
| Feature engineering | Done |
| Model training | Done |
| GE evaluation | Done |
| SHAP analysis | Done |
| HW vs ID comparison | Done |
| Streamlit app | Pending |
| Research paper | In progress |

---

## Target Publication

**Primary:** IEEE Access 
**Backup:** INDOCRYPT 2025

---

## References

- Benadjila et al. (2020) — ASCAD database and deep learning for SCA
- Maghrebi et al. (2016) — Breaking cryptographic implementations using deep learning
- Picek et al. (2018) — The curse of class imbalance in SCA evaluations
- Lundberg & Lee (2017) — A unified approach to interpreting model predictions (SHAP)
- Standaert et al. (2009) — A unified framework for side-channel key recovery analysis