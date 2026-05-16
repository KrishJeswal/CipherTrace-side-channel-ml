# CipherTrace

**ML-based side-channel attack on masked AES-128.**  
Systematic evaluation of feature engineering strategies for profiling attacks, with SHAP leakage localization and a leakage model comparison.

> Paper submitted to IEEE Access, May 2026.

---

## What this is

A complete profiling attack pipeline against a first-order Boolean masked AES-128 implementation. The target is the ASCAD fixed-key dataset — 50,000 power traces from an ATMega8515 captured by ANSSI (French National Cybersecurity Agency).

The core question: given a power trace recorded during AES encryption, can a classical ML model recover the secret key? And if so, which feature engineering strategy and classifier combination works best — measured not by accuracy, but by Guessing Entropy.

Three novel findings:

- **Macro F1 does not predict GE.** Decision Tree had the best F1 (0.1094) and one of the worst GE values (148.86). MLP had mid-range F1 and the best GE (16.36 at k=50 with PCA).
- **ANOVA is actively harmful on masked AES.** It selects mask-correlated samples instead of key-correlated ones, pushing GE above the random baseline of 127.
- **Identity labels break the key. HW labels do not.** Replacing 9-class Hamming Weight labels with 256-class Identity labels reduces GE from 12.03 to 0.46 at 500 traces — near-complete key recovery.

---

## Results

### Classification Performance (SNR k=200)

| Classifier | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | **0.2713** | 0.0799 |
| XGBoost | 0.2675 | 0.0871 |
| MLP | 0.2612 | 0.0867 |
| Logistic Regression | 0.2633 | 0.0748 |
| Decision Tree | 0.2341 | **0.1094** |

Random baseline: accuracy = 0.111, Macro F1 ≈ 0.111.

### Guessing Entropy at 500 Traces (k=50)

| Classifier | SNR | ANOVA | PCA |
|---|---|---|---|
| MLP | 53.49 | 138.23 | **16.36 ★** |
| XGBoost | 110.06 | 146.19 | 102.07 |
| Decision Tree | 148.86 | 113.29 | 100.75 |
| Random Forest | 121.86 | 146.66 | 134.45 |
| SVM | 126.63 | — | — |
| Logistic Regression | 151.76 | 149.91 | 153.82 |

Random baseline GE = 127.

### HW vs. Identity (MLP + PCA k=100)

| Leakage Model | Classes | GE@100 | GE@500 | NtD |
|---|---|---|---|---|
| Hamming Weight | 9 | 51.34 | 12.03 | Not reached |
| Identity | 256 | 9.81 | **0.46** | ~500 traces |

---

## Dataset

**ASCAD fixed-key** — download from [data.gouv.fr](https://www.data.gouv.fr/datasets/ascad)

| Property | Value |
|---|---|
| Profiling traces | 50,000 |
| Attack traces | 10,000 |
| Samples per trace | 700 |
| Target byte index | 2 |
| Correct key byte | 0xe0 (224) |
| Countermeasure | First-order Boolean masking |
| Max SNR | 0.0032 |

Place the downloaded `ASCAD.h5` file at `data/ASCAD.h5` before running anything.

---

## Project Structure

```
side-channel-ml/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py        # Load ASCAD.h5, split profiling/attack sets
│   │   ├── data_transformation.py   # SNR, ANOVA, PCA feature extraction pipeline
│   │   └── model_trainer.py         # Train 6 classifiers, 5-fold CV, joblib save
│   ├── pipeline/
│   │   ├── train_pipeline.py        # Orchestrates full training flow
│   │   └── predict_pipeline.py      # Load model, compute GE on attack traces
│   ├── utils.py                     # GE computation, SNR, SHAP helpers
│   ├── logger.py
│   └── exception.py
├── notebooks/
│   ├── 01_EDA.ipynb                 # Trace visualization, SNR, label distribution
│   ├── 02_feature_engineering.ipynb # POI strategy comparison, PCA variance
│   ├── 03_model_comparison.ipynb    # 72-run grid, GE heatmap, k sweep
│   └── 04_novel_contributions.ipynb # SHAP analysis, HW vs. Identity GE curves
├── artifacts/                       # Saved models and processed arrays
├── app.py                           # Streamlit: upload trace → predict key byte
├── requirements.txt
├── setup.py
└── README.md
```

---

## Setup

```bash
git clone https://github.com/KrishJeswal/side-channel-ml
cd side-channel-ml

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Place `ASCAD.h5` at `data/ASCAD.h5`, then verify the dataset:

```bash
python verify.py
# Expected: profiling_traces (50000, 700), attack_traces (10000, 700)
```

---

## Running the Pipeline

**Full training run:**
```bash
python src/pipeline/train_pipeline.py
```

Trains all 6 classifiers across all 3 POI strategies and 4 values of k. Saves models to `artifacts/`.

**GE evaluation:**
```bash
python src/pipeline/predict_pipeline.py
```

**Streamlit app:**
```bash
streamlit run app.py
```

---

## Tech Stack

- **Data** — h5py, numpy, pandas
- **ML** — scikit-learn, XGBoost
- **Explainability** — SHAP
- **Visualization** — matplotlib, seaborn
- **App** — Streamlit

---

## Paper

**Systematic Evaluation of Machine Learning Classifiers and Feature Engineering Strategies for Side-Channel Attacks on Masked AES-128**  
Krish Jeswal — RV College of Engineering, Bangalore  
*Submitted to IEEE Access, May 2026*

---

## Author

**Krish Jeswal**  
Electronics and Communication Engineering, RVCE (Batch 2024–2028)  
[GitHub](https://github.com/KrishJeswal) · [LinkedIn](https://linkedin.com/in/krishjeswal)