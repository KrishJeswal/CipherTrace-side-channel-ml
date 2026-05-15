import os
import sys
import numpy as np
import streamlit as st
import joblib
import h5py

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.utils import compute_snr, compute_ge, AES_SBOX

# ── Constants ────────────────────────────────────────────────────────────────

CORRECT_KEY_BYTE = 0xe0
TARGET_BYTE      = 2
N_TRACES_RANGE   = [10, 25, 50, 100, 200, 500]

MODEL_LABELS = {
    "mlp":               "MLP (Neural Network)",
    "xgboost":           "XGBoost",
    "random_forest":     "Random Forest",
    "decision_tree":     "Decision Tree",
    "logistic_regression": "Logistic Regression",
    "svm":               "SVM (RBF kernel)",
}
STRATEGY_LABELS = {
    "pca": "PCA (Principal Component Analysis)",
    "snr": "SNR-based POI Selection",
    "anova": "ANOVA F-test POI Selection",
}

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Side-Channel Attack Demo",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* ── FORCE POINTER CURSOR ON INTERACTIVE ELEMENTS ── */

/* Targets the dropdown box and the internal text area */
div[data-baseweb="select"], 
div[data-baseweb="select"] > div, 
div[data-testid="stSelectbox"] [role="button"],
div[data-testid="stSelectbox"] input {
    cursor: pointer !important;
}

/* Targets the Slider track and the handle */
div[data-testid="stSlider"] > div,
div[data-testid="stSlider"] [role="slider"] {
    cursor: pointer !important;
}

/* Targets the sidebar buttons and main buttons */
.stButton button {
    background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    width: 100%;
    cursor: pointer !important; /* Ensure pointer on buttons */
}
.stButton button:hover { background: linear-gradient(135deg, #0284c7, #0369a1) !important; }

/* ── REST OF YOUR STYLES ── */

.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.main-header h1 {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 2rem;
    color: #f1f5f9;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}

.main-header p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}

.accent { color: #38bdf8; }

.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.metric-card .label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #38bdf8;
    margin: 0.2rem 0;
}

.metric-card .sub {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 0.2rem;
}

.result-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.badge-success { background: #064e3b; color: #34d399; border: 1px solid #065f46; }
.badge-warn    { background: #451a03; color: #fb923c; border: 1px solid #7c2d12; }
.badge-info    { background: #0c4a6e; color: #38bdf8; border: 1px solid #0369a1; }

.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #cbd5e1;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 1.5rem 0 0.8rem 0;
    border-left: 3px solid #38bdf8;
    padding-left: 0.8rem;
}

.info-box {
    background: #0c1930;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.95rem;
    color: #93c5fd;
    margin: 0.8rem 0;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.7;
}

.extraction-card {
    background: #0f172a;
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
}

.extraction-card h3 {
    margin: 0;
    font-size: 1.1rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
}

.extraction-card .rank-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    margin: 0.5rem 0;
}

.extraction-card .rank-text {
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.extraction-card .desc {
    color: #cbd5e1;
    font-size: 1rem;
    line-height: 1.6;
    max-width: 800px;
    margin: 0 auto;
}

hr { border-color: #1e293b; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_attack_data():
    path = os.path.join("artifacts", "raw", "ASCAD_processed.h5")
    with h5py.File(path, "r") as f:
        X_prof  = f["Profiling_traces/traces"][:].astype(np.float32)
        pt_prof = f["Profiling_traces/metadata/plaintext"][:]
        X_atk   = f["Attack_traces/traces"][:].astype(np.float32)
        pt_atk  = f["Attack_traces/metadata/plaintext"][:]
    y_prof = np.array([
        bin(AES_SBOX[int(pt[TARGET_BYTE]) ^ CORRECT_KEY_BYTE]).count('1')
        for pt in pt_prof
    ], dtype=np.int32)
    snr = compute_snr(X_prof, y_prof)
    return X_prof, y_prof, pt_prof, X_atk, pt_atk, snr

def load_model_and_transformer(model_name, strategy, k):
    tag = f"{strategy}_k{k}" if strategy != "pca" else f"{strategy}_k{k}"
    model_path = os.path.join("artifacts", "models", f"{model_name}_{strategy}_k{k}.joblib")
    transformer_key = f"pca_n{k}" if strategy == "pca" else f"{strategy}_k{k}"
    transformer_path = os.path.join("artifacts", "models", f"transformer_{transformer_key}.joblib")

    if not os.path.exists(model_path):
        return None, None, f"Model file not found: {model_path}"
    if not os.path.exists(transformer_path):
        return None, None, f"Transformer file not found: {transformer_path}"

    model       = joblib.load(model_path)
    transformer = joblib.load(transformer_path)
    return model, transformer, None

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <h1>🔐 Side-Channel Attack <span class="accent">Demo</span></h1>
  <p>ML-based profiling attack on AES-128 · ASCAD Dataset · Krish Jeswal, RVCE</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Attack Configuration")
    
    st.markdown("""
    **Configure your attack variables here:**
    * **Classifier**: The Machine Learning algorithm used to predict the key byte.
    * **POI Strategy**: How the algorithm selects the most important features (Points of Interest) from the raw data.
    * **k**: The number of features extracted from the raw power samples.
    * **GE experiments**: The number of attack iterations averaged to calculate a reliable Guessing Entropy.
    """)

    model_name = st.selectbox(
        "Classifier",
        options=list(MODEL_LABELS.keys()),
        format_func=lambda x: MODEL_LABELS[x],
        index=0
    )

    strategy = st.selectbox(
        "POI Strategy",
        options=list(STRATEGY_LABELS.keys()),
        format_func=lambda x: STRATEGY_LABELS[x],
        index=0
    )

    k = st.select_slider(
        "k (POIs / PCA components)",
        options=[20, 50, 100, 200],
        value=100
    )

    n_experiments = st.slider("GE experiments", min_value=10, max_value=200, value=50, step=10)

    st.markdown("---")
    st.markdown("### 📊 Quick Reference Guide")
    st.markdown("""
    <div class="info-box">
    <strong>Best Discovered Config:</strong><br>
    MLP + PCA + k=100<br><br>
    <strong>Target Benchmarks:</strong><br>
    Random Guessing ≈ 127 GE<br>
    Successful Attack: &lt; 5 GE
    </div>
    """, unsafe_allow_html=True)

    run_attack = st.button("▶ Run Attack", use_container_width=True)

# ── Load data ─────────────────────────────────────────────────────────────────

with st.spinner("Loading ASCAD dataset..."):
    try:
        X_prof, y_prof, pt_prof, X_atk, pt_atk, snr = load_attack_data()
        data_ok = True
    except Exception as e:
        st.error(f"Failed to load ASCAD_processed.h5: {e}")
        data_ok = False

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🎯 Run Attack", "📈 Dataset Guide", "📚 Technical About"])

# ── TAB 1: ATTACK ────────────────────────────────────────────────────────────

with tab1:
    if not data_ok:
        st.warning("Dataset not loaded. Please ensure the ASCAD dataset is downloaded and present in `artifacts/raw/ASCAD_processed.h5`.")
        st.stop()

    st.markdown("""
    ### ⚙️ What Happens When You Run an Attack?
    When you click **Run Attack**, the application executes a series of steps designed to extract the secret AES key without brute-forcing the cryptography itself:
    
    1. **Feature Extraction**: The raw power measurements of the target chip are transformed using your chosen strategy (PCA, SNR, or ANOVA) down to `k` features. This isolates the "leaky" moments.
    2. **Machine Learning Prediction**: The selected classifier looks at those features and outputs a probability score for each of the 9 Hamming Weight (HW) classes. It estimates how many binary '1's were processed inside the chip during encryption.
    3. **Log-Likelihood Accumulation**: We loop through all 256 possible byte values (hypotheses) for the secret key. We check what the HW *would* be for each hypothesis, and score it using the probabilities generated by our ML model.
    4. **Key Ranking (Guessing Entropy)**: Finally, we sort all 256 hypotheses from highest score to lowest. If our attack is working, the correct secret key floats to the top.
    """)

    if run_attack:
        model, transformer, err = load_model_and_transformer(model_name, strategy, k)

        if err:
            st.error(f"⚠️ {err}\n\nRun the training pipeline first:\n`python -m src.pipeline.train_pipeline --strategy {strategy} --k {k}`")
        else:
            with st.spinner(f"Running evaluation metrics over {n_experiments} independent experiments..."):
                X_atk_t = transformer.transform(X_atk)
                probs    = model.predict_proba(X_atk_t)

                ge_values = compute_ge(
                    probs, pt_atk, CORRECT_KEY_BYTE,
                    n_traces_range=N_TRACES_RANGE,
                    n_experiments=n_experiments
                )

                # Single trace prediction
                hw_pred, hw_probs = int(np.argmax(probs[0])), probs[0]
                ntd = next((n for n, ge in zip(N_TRACES_RANGE, ge_values) if ge < 1), None)

            st.markdown("---")
            st.markdown('<div class="section-title">Attack Results</div>', unsafe_allow_html=True)
            
            ge_500 = ge_values[-1]
            badge_class = "badge-success" if ge_500 < 5 else ("badge-warn" if ge_500 < 50 else "badge-info")
            badge_text  = "KEY BROKEN" if ge_500 < 5 else ("PARTIAL SUCCESS" if ge_500 < 50 else "FAILED TO BREAK")

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">GE @ 500 traces</div>
                  <div class="value">{ge_500:.1f}</div>
                  <div class="sub"><span class="result-badge {badge_class}">{badge_text}</span></div>
                </div>""", unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">GE @ 100 traces</div>
                  <div class="value">{ge_values[N_TRACES_RANGE.index(100)]:.1f}</div>
                  <div class="sub">rank of correct key</div>
                </div>""", unsafe_allow_html=True)

            with m3:
                ntd_display = str(ntd) if ntd else ">500"
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">NtD (Traces to GE&lt;1)</div>
                  <div class="value">{ntd_display}</div>
                  <div class="sub">number of traces to disclose</div>
                </div>""", unsafe_allow_html=True)

            with m4:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">Predicted HW (Trace 0)</div>
                  <div class="value">{hw_pred}</div>
                  <div class="sub">Hamming Weight class</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("""
            **Understanding Your Results:**
            * **GE @ 500 traces (Guessing Entropy)**: This represents the average rank of the correct secret key out of 256 possibilities after the model has analyzed 500 traces. A lower number is better. 
            * **GE @ 100 traces**: The rank of the key after only 100 traces. This shows how fast your selected ML algorithm is learning.
            * **NtD (Number of Traces to Disclose)**: The exact number of traces required for the Guessing Entropy to drop below 1 (meaning the correct key is securely locked in as the #1 prediction).
            """)

            # Calculate scores to find the correct key's rank
            scores = np.zeros(256)
            sample_idx = np.random.choice(len(pt_atk), 500, replace=False)
            for k_hyp in range(256):
                hw_hyp = [bin(AES_SBOX[int(pt_atk[i, TARGET_BYTE]) ^ k_hyp]).count('1')
                          for i in sample_idx]
                scores[k_hyp] = sum(
                    np.log(probs[i, hw_hyp[j]] + 1e-9)
                    for j, i in enumerate(sample_idx)
                )
            
            correct_key_rank = int(np.where(np.argsort(scores)[::-1] == CORRECT_KEY_BYTE)[0][0]) + 1

            # Determine visual theme for extraction result
            if correct_key_rank == 1:
                card_border = "border: 2px solid #34d399;"
                card_color = "#34d399" # Green
                rank_title = "Key Successfully Extracted!"
                rank_desc = "The model perfectly identified the secret byte (`0xe0`) from 256 possibilities. No brute-forcing is required. The side-channel vulnerability is critical."
            elif correct_key_rank <= 50:
                card_border = "border: 2px solid #fb923c;"
                card_color = "#fb923c" # Orange
                rank_title = "Partial Success / Narrowed Search"
                rank_desc = "The model significantly narrowed down the search space. While the exact key wasn't ranked #1, a highly reduced computational brute-force attack would succeed rapidly."
            else:
                card_border = "border: 2px solid #ef4444;"
                card_color = "#ef4444" # Red
                rank_title = "Attack Failed"
                rank_desc = "The model essentially failed to learn the leakage patterns from the hardware masking defense. It is either guessing randomly or was actively misled by the strategy."

            st.markdown('<div class="section-title">Attack Performance & Analysis</div>', unsafe_allow_html=True)
            
            # Key Extraction Visual Card
            st.markdown(f"""
            <div class="extraction-card" style="{card_border}">
                <h3>Final Key Prediction Rank</h3>
                <div class="rank-number" style="color: {card_color};">#{correct_key_rank}</div>
                <div class="rank-text" style="color: {card_color};">{rank_title}</div>
                <div class="desc">
                    Out of 256 possible byte hypotheses, your selected configuration placed the correct secret key (`0xe0`) at <b>Rank {correct_key_rank}</b>. <br><br>
                    {rank_desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Dynamic feedback based on selections
            model_feedback = ""
            if model_name == "mlp":
                model_feedback = "You chose **MLP (Neural Network)**, which natively produces continuous, well-calibrated sigmoid outputs perfectly suited for log-likelihood accumulation."
            elif model_name in ["xgboost", "random_forest", "decision_tree"]:
                model_feedback = f"You chose a **Tree-based model ({MODEL_LABELS[model_name]})**. Tree models produce poorly calibrated, overconfident probabilities (near 0 or 1). This causes them to perform poorly on Guessing Entropy metrics despite potentially high classification accuracy."
            else:
                model_feedback = f"You chose **{MODEL_LABELS[model_name]}**. While it provides a baseline, neural networks (MLPs) significantly outperform it on this complex masking defense."

            strategy_feedback = ""
            if strategy == "pca":
                strategy_feedback = "Your choice of **PCA** is highly effective here. It allows the model to exploit non-linear, distributed leakage patterns that univariate metrics cannot detect."
            elif strategy == "snr":
                strategy_feedback = "Your choice of **SNR** captures basic first-order univariate leakages. However, because the target chip uses a Boolean masking defense, the max SNR is an extremely low 0.0032. It misses complex feature interactions."
            elif strategy == "anova":
                strategy_feedback = "Your choice of **ANOVA** is actively harmful on masked ASCAD. It selects statistically significant samples that are not correlated with the HW model, causing the model to learn spurious features."

            st.markdown(f'''
            ### 🔍 Technical Breakdown & Benchmark Comparison
            
            **1. Your Classifier Choice**: {model_feedback}
            
            **2. Your Strategy Choice**: {strategy_feedback}
            
            **3. How does this compare to the project benchmark?** The absolute best result discovered in this project used an **Identity Leakage Model (256 classes) + MLP + PCA (k=100)**, which completely broke the key with a GE of **0.46** at 500 traces. Under the current 9-class Hamming Weight model being tested, the best baseline is **HW + MLP + PCA (k=50)**, hitting a GE of **16.36**. 
            ''')

    else:
        st.info("Configure the attack parameters in the left sidebar and click **▶ Run Attack** to start analyzing the data.")

# ── TAB 2: EDA ───────────────────────────────────────────────────────────────

with tab2:
    if not data_ok:
        st.warning("Dataset not loaded. Please ensure the ASCAD dataset is available.")
        st.stop()

    st.markdown("""
    <div class="section-title">Dataset Characteristics & Target Device</div>

    The ASCAD dataset utilized in this project was sourced from the **French National Cybersecurity Agency (ANSSI)**. It consists of roughly 1.7 GB of processed data capturing the operation of an **ATMega8515 microcontroller executing a masked version of AES-128 encryption**.

    These measurements act as an unintentional "side channel" that leaks information about the secrets inside the chip.
    * **Data Properties**: The traces are stored as `float32` and possess a value range falling between `-66.00` and `47.00`. 
    * **Dimensions**: There are 50,000 power traces designated for ML profiling (training), and 10,000 traces designated for the attack (testing). Every single trace is composed of exactly 700 individual power samples.

    **The Hardware Defense: First-Order Boolean Masking**
    This specific dataset is challenging because the target device utilizes a defense mechanism called **first-order Boolean masking**. This defense is designed specifically to thwart basic statistical attacks by randomly splitting internal variables, thereby hiding first-order leakages.
    * Because of this masking, the maximum Signal-to-Noise Ratio (SNR) achievable across the entire 700-sample timeline is an extremely weak **0.0032**, peaking at sample index 567.
    * When mapping the mean power trace per class, there is no visible divergence or separation, proving the masking successfully obscures standard data leakage.

    **Feature Overlap Constraints:**
    If we compare simple univariate feature selection metrics, we observe significant divergence. Out of 50 chosen Points of Interest (POIs), the traditional SNR metric and the ANOVA F-test metric only agree on exactly 13 samples. On masked ASCAD, ANOVA is actively harmful; it selects samples that appear statistically significant but hold zero correlation with the underlying Hamming Weight leakage model.

    <div class="section-title">Hamming Weight (HW) Class Distribution</div>

    Our primary task is an ML 9-class classification problem. The target is to predict the **Hamming Weight**—the total number of binary '1's processed by the chip's internal S-Box matrix. Since a byte has 8 bits, there are 9 possible Hamming Weights (0 through 8).

    The distribution of the labels within the dataset perfectly matches the mathematically expected Binomial distribution `B(8, 0.5)`. Because of this binomial nature, a heavy class imbalance is entirely expected and unavoidable.

    | HW Class | Total Count | Proportion | Note |
    |---|---|---|---|
    | Class **0** | 166 traces | 0.3% | Extremely rare occurrence |
    | Class **1** | 1,568 traces | 3.1% | |
    | Class **2** | 5,500 traces | 11.0% | |
    | Class **3** | 10,848 traces | 21.7% | |
    | Class **4** | 13,747 traces | 27.5% | The most common HW value |
    | Class **5** | 11,023 traces | 22.0% | |
    | Class **6** | 5,416 traces | 10.8% | |
    | Class **7** | 1,532 traces | 3.1% | |
    | Class **8** | 200 traces | 0.4% | Extremely rare occurrence |
    """, unsafe_allow_html=True)


# ── TAB 3: ABOUT ─────────────────────────────────────────────────────────────

with tab3:
    st.markdown("""
    <div class="section-title">1. What is a Profiling Side-Channel Attack?</div>

    A **profiling side-channel attack** is an advanced supervised Machine Learning methodology that exploits the unintended physical information leaked by a cryptographic device. 
    
    When a microchip runs an algorithm, it requires electricity. Its power consumption subtly fluctuates depending on the precise zeroes and ones it is processing internally. By systematically capturing these microscopic power fluctuations ("traces") and attaching them to known mathematical labels, we can train ML models. Once trained, the ML model can analyze *unknown* traces and mathematically reconstruct the 128-bit secret key without ever having to break the actual cryptographic mathematics of AES.

    <div class="section-title">2. Technical Findings & Novel Contributions</div>

    This application is a demonstration of findings extracted from the ASCAD dataset, highlighting several critical research observations:

    **A. Feature Engineering & PCA**
    Principal Component Analysis (PCA) consistently outperforms simple feature extractors like SNR and ANOVA on this masked dataset. Specifically, reducing the traces to 50 PCA components is enough to explain approximately 95% of the variance. 

    **B. SHAP Leakage Localization**
    When running SHAP (SHapley Additive exPlanations) on our best Multi-Layer Perceptron (MLP) model, we find that the top 10 most important features identified by SHAP have almost no overlap with the top 10 features identified by traditional SNR (only a 2/10 overlap). 
    * This low overlap is not a failure—it is the finding itself. SNR only identifies *first-order univariate* leakages.
    * SHAP captures feature interactions, providing evidence that the trained MLP exploits distributed, non-linear leakage patterns that SNR cannot detect. This explains why MLPs outperform models that rely solely on SNR for feature selection.

    **C. The Superiority of Neural Networks over Trees**
    A core finding is that traditional evaluation metrics like "Macro F1" do not reliably predict an attack's actual cryptographic success. For example, Decision Tree models achieved the highest overall Macro F1 score (0.1094), but had one of the worst Guessing Entropy values.
    
    The critical metric is **Guessing Entropy (GE)**, which relies on adding together logarithmic probabilities. Tree models (like Random Forest and XGBoost) produce poorly calibrated, overconfident probabilities near 0 or 1 that ruin log-likelihood accumulation. Post-hoc calibration (using CalibratedClassifierCV) actually made XGBoost's GE worse (dropping from 119.53 to 135.67). MLPs natively produce continuous, well-calibrated sigmoid outputs perfectly suited for this task.

    **D. Identity vs. Hamming Weight Paradigm Shift**
    Standard side-channel attacks reduce the 256 possible byte values down to 9 Hamming Weight classes, under the assumption that it simplifies the classification problem. This project demonstrates that against masked ASCAD, an **Identity leakage model** (which forces the MLP to classify all 256 individual byte values) is vastly superior. 
    
    While the standard HW model achieved a GE of 12.03, the 256-class Identity model combined with MLP and PCA achieved an astounding GE of **0.46** at 500 traces. This means the exact correct secret key was confidently ranked #1 in ~54% of the experiments. This proves that aggregating values into 9 bins aggressively destroys critical, fine-grained probability distinctions that MLPs are highly capable of exploiting.

    <div class="section-title">3. Project Information</div>

    * **Author**: Krish Jeswal
    * **Institute**: RVCE Bengaluru | EC Batch 2024–2028
    * **GitHub**: KrishJeswal
    * **Target Venue**: IEEE ISCAS 2026
    """, unsafe_allow_html=True)