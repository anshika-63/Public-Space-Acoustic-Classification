"""
Public-Space Acoustic Event Classification Application
T254 - Streamlit Frontend for Environmental Audio Recognition

A production-grade, user-friendly interface for classifying environmental
and public-space acoustic events using trained Phase 3 Machine Learning models.
"""

import io
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference import (
    CLASSES,
    CLASS_METADATA,
    classify_audio,
    extract_features,
    generate_demo_audio,
    get_audio_metadata,
    load_model,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Public-Space Acoustic Event Classifier",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, high-contrast, professional UI
st.markdown(
    """
    <style>
    /* Global Container Spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        color: #f8fafc;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 6px;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #cbd5e1;
        margin-bottom: 14px;
        line-height: 1.5;
    }
    .badge-tag {
        display: inline-block;
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-tag-alt {
        display: inline-block;
        background-color: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }

    /* Result Cards */
    .prediction-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #38bdf8;
        border-radius: 12px;
        padding: 22px;
        margin-top: 16px;
        margin-bottom: 20px;
        color: #ffffff;
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.12);
    }
    .prediction-icon {
        font-size: 3.2rem;
        line-height: 1;
        margin-bottom: 8px;
    }
    .prediction-name {
        font-size: 1.9rem;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 4px;
    }
    .prediction-category {
        font-size: 0.95rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .prediction-desc {
        font-size: 0.98rem;
        color: #e2e8f0;
        line-height: 1.5;
        margin-bottom: 14px;
    }
    .stat-pill {
        display: inline-block;
        background: #0f172a;
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 8px 14px;
        margin-right: 10px;
        margin-bottom: 6px;
    }
    .stat-pill-label {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        display: block;
    }
    .stat-pill-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
    }

    /* Top-3 Mini Cards */
    .runner-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #f8fafc;
    }

    /* Disclaimers & Callouts */
    .honesty-callout {
        background-color: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 14px;
        margin-bottom: 18px;
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Cached Resource & Data Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading acoustic model weights into memory...")
def get_cached_model(model_key: str) -> Tuple[Any, Dict[str, Any]]:
    """Cached wrapper around load_model to avoid reloading heavy joblib files."""
    return load_model(model_key)


@st.cache_data
def get_phase3_evaluation_summary() -> Optional[Dict[str, Any]]:
    """Load Phase 3 benchmark metrics JSON if present."""
    metrics_path = "models/phase3/evaluation_metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# 3. Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Acoustic Classifier")
    st.caption("T254 • Public-Space Acoustic Event Recognition")

    # Model Selection
    st.markdown("#### 1. Select Model Architecture")
    model_choice = st.radio(
        label="Inference Model:",
        options=[
            "Random Forest (Recommended)",
            "SVM Pipeline (RBF Kernel)",
            "Best Deployable Model (CV Winner)",
        ],
        index=0,
        help=(
            "Random Forest provides ensemble tree voting probability estimates.\n"
            "SVM Pipeline provides raw signed decision margins to the separating hyperplanes."
        ),
    )

    if "Random Forest" in model_choice:
        selected_model_key = "random_forest"
        model_display_name = "Random Forest (300 Trees)"
    elif "SVM" in model_choice:
        selected_model_key = "svm"
        model_display_name = "SVM Pipeline (StandardScaler + SVC)"
    else:
        selected_model_key = "best"
        model_display_name = "Best Deployable Acoustic Model"

    # Audio Input Source
    st.markdown("---")
    st.markdown("#### 2. Audio Input Source")
    input_mode = st.radio(
        label="Input Method:",
        options=["Upload Audio File", "Built-in Demo Clip"],
        index=1,
        help="Upload your own audio recording or select an interactive synthetic demonstration sound.",
    )

    # Benchmark Quick View
    st.markdown("---")
    st.markdown("#### 📊 Phase 3 Benchmark")
    eval_data = get_phase3_evaluation_summary()
    if eval_data:
        rf_acc = eval_data.get("random_forest", {}).get("accuracy", 0.6683) * 100
        rf_f1 = eval_data.get("random_forest", {}).get("macro_f1", 0.6868) * 100
        svm_acc = eval_data.get("svm", {}).get("accuracy", 0.6703) * 100
        svm_f1 = eval_data.get("svm", {}).get("macro_f1", 0.6858) * 100

        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            st.metric("RF Macro F1", f"{rf_f1:.1f}%")
            st.metric("RF Accuracy", f"{rf_acc:.1f}%")
        with col_sb2:
            st.metric("SVM Macro F1", f"{svm_f1:.1f}%")
            st.metric("SVM Accuracy", f"{svm_acc:.1f}%")
        st.caption("10-Fold Leakage-Free CV on 8,732 clips.")
    else:
        st.info("Evaluation metrics file available in `models/phase3/`.")

    st.markdown("---")
    st.caption("Google DeepMind Project T254 • Powered by Streamlit & Librosa")


# -----------------------------------------------------------------------------
# 4. Main Page Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🎧 Public-Space Acoustic Event Classification</div>
        <div class="hero-subtitle">
            Automated environmental sound recognition for smart city infrastructure, surveillance monitoring,
            and municipal noise compliance. Trained and validated on the 10-class UrbanSound8K dataset.
        </div>
        <div>
            <span class="badge-tag">80 MFCC Features</span>
            <span class="badge-tag">22,050 Hz Standardized</span>
            <span class="badge-tag">10 Official Folds CV</span>
            <span class="badge-tag-alt">Strict Score Honesty</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 5. Audio Selection & Acquisition
# -----------------------------------------------------------------------------
audio_bytes: Optional[bytes] = None
audio_filename: str = ""

st.markdown("### 1. Acoustic Input Signal")

if input_mode == "Upload Audio File":
    uploaded_file = st.file_uploader(
        "Upload an environmental sound clip (WAV, MP3, FLAC, or OGG):",
        type=["wav", "mp3", "flac", "ogg"],
        help="Upload public space or ambient sound recordings. Clips between 1 and 4 seconds yield optimal representation.",
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        audio_filename = uploaded_file.name

else:
    demo_col1, demo_col2 = st.columns([2, 1])
    with demo_col1:
        demo_sound = st.selectbox(
            "Select an interactive demonstration sound pattern:",
            options=[
                ("siren", "🚨 Siren — Emergency frequency-swept warning tone"),
                ("car_horn", "🚗 Car Horn — Dual-frequency warning chord"),
                ("engine_idling", "🚛 Engine Idling — Low-frequency multi-harmonic rumble"),
                ("drilling", "🔩 Drilling — Rotary percussion vibration"),
                ("gun_shot", "💥 Gun Shot — Fast supersonic acoustic shockwave"),
            ],
            format_func=lambda x: x[1],
            index=0,
        )
    with demo_col2:
        demo_duration = st.slider("Synthetic Clip Duration (seconds):", 1.0, 3.5, 2.0, 0.5)

    if demo_sound:
        demo_key = demo_sound[0]
        audio_bytes = generate_demo_audio(event_type=demo_key, duration=demo_duration)
        audio_filename = f"synthetic_{demo_key}_{demo_duration:.1f}s.wav"


# -----------------------------------------------------------------------------
# 6. Audio Player and Signal Metrics Inspection
# -----------------------------------------------------------------------------
if audio_bytes is not None and len(audio_bytes) > 0:
    st.markdown("#### 🎧 Audio Playback & Signal Properties")
    player_col, metrics_col = st.columns([1, 1])

    with player_col:
        st.audio(audio_bytes, format="audio/wav")
        st.caption(f"Active clip: **{audio_filename}** ({len(audio_bytes):,} bytes)")

    with metrics_col:
        try:
            audio_info = get_audio_metadata(audio_bytes)
            m1, m2, m3 = st.columns(3)
            m1.metric("Duration", f"{audio_info['duration_seconds']:.2f} s")
            m2.metric("Sample Rate", f"{audio_info['sample_rate']:,} Hz")
            m3.metric("Channels", audio_info["channel_mode"])
            st.caption(
                f"Container format: **{audio_info['format']}** ({audio_info['subtype']}). "
                "Resampled to **22,050 Hz mono** during feature extraction."
            )
        except Exception as ex:
            st.warning(f"Could not read metadata tags: {ex}")

    # -------------------------------------------------------------------------
    # 7. Classification Action & Inference Execution
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 2. Event Classification & Analysis")

    run_classify = st.button(
        "🔍 Classify Acoustic Event",
        type="primary",
        use_container_width=True,
    )

    if run_classify or st.session_state.get("auto_run", False):
        st.session_state["auto_run"] = True

        with st.spinner(f"Running inference with {model_display_name}..."):
            try:
                # Load model
                model, metadata = get_cached_model(selected_model_key)

                # Execute classification
                result = classify_audio(
                    audio_input=audio_bytes,
                    model=model,
                    metadata=metadata,
                    model_name=model_display_name,
                )

                # Hero Result Banner
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div class="prediction-icon">{result['predicted_icon']}</div>
                        <div class="prediction-name">{result['predicted_label']}</div>
                        <div class="prediction-category">{result['predicted_category']} &bull; UrbanSound8K Class: <code>{result['predicted_class']}</code></div>
                        <div class="prediction-desc">{result['predicted_description']}</div>
                        <div>
                            <div class="stat-pill">
                                <span class="stat-pill-label">Primary Metric ({result['score_unit'].upper()})</span>
                                <span class="stat-pill-value">{result['top_display_score']}</span>
                            </div>
                            <div class="stat-pill">
                                <span class="stat-pill-label">Metric Type</span>
                                <span class="stat-pill-value">{result['score_type'].capitalize()}</span>
                            </div>
                            <div class="stat-pill">
                                <span class="stat-pill-label">Inference Latency</span>
                                <span class="stat-pill-value">{result['latency_ms']:.1f} ms</span>
                            </div>
                            <div class="stat-pill">
                                <span class="stat-pill-label">Model Architecture</span>
                                <span class="stat-pill-value">{result['model_name']}</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Top-3 Predictions Section
                st.markdown("#### 🥇 Top-3 Ranked Acoustic Hypotheses")
                r_col1, r_col2, r_col3 = st.columns(3)
                top_3 = result["top_3"]

                for col, rank, rank_badge in zip(
                    [r_col1, r_col2, r_col3],
                    top_3,
                    ["Rank 1 (Winner)", "Rank 2 (Runner-Up)", "Rank 3 (Alternative)"],
                ):
                    with col:
                        st.markdown(
                            f"""
                            <div class="runner-card">
                                <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">{rank_badge}</div>
                                <div style="font-size: 1.3rem; font-weight: 700; margin-top: 4px; color: #f8fafc;">
                                    {rank['icon']} {rank['label']}
                                </div>
                                <div style="font-size: 1.1rem; color: #38bdf8; font-weight: 700; margin-top: 4px;">
                                    {rank['display_score']}
                                </div>
                                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">
                                    Class ID: <code>{rank['class']}</code>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Full 10-Class Score Distribution Chart
                st.markdown(f"#### 📊 {result['score_label']}")

                scores_df = pd.DataFrame(result["all_scores"])

                if result["score_type"] == "probability":
                    # Altair horizontal bar chart for Random Forest
                    chart = (
                        alt.Chart(scores_df)
                        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                        .encode(
                            x=alt.X(
                                "score_pct:Q",
                                title="Probability Estimate (%)",
                                scale=alt.Scale(domain=[0, 100]),
                            ),
                            y=alt.Y("label:N", sort="-x", title="Acoustic Event Class"),
                            color=alt.condition(
                                alt.datum["class"] == result["predicted_class"],
                                alt.value("#38bdf8"),  # Highlight winner
                                alt.value("#475569"),
                            ),
                            tooltip=[
                                alt.Tooltip("label:N", title="Class"),
                                alt.Tooltip("display_score:N", title="Probability"),
                            ],
                        )
                        .properties(height=350)
                    )
                    st.altair_chart(chart, use_container_width=True)

                else:
                    # Altair horizontal bar chart for SVM Decision Margins (can be negative or positive)
                    chart = (
                        alt.Chart(scores_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("score:Q", title="Signed Distance to Separating Hyperplane"),
                            y=alt.Y("label:N", sort="-x", title="Acoustic Event Class"),
                            color=alt.condition(
                                alt.datum["class"] == result["predicted_class"],
                                alt.value("#38bdf8"),
                                alt.condition(
                                    alt.datum.score > 0,
                                    alt.value("#64748b"),
                                    alt.value("#334155"),
                                ),
                            ),
                            tooltip=[
                                alt.Tooltip("label:N", title="Class"),
                                alt.Tooltip("display_score:N", title="Decision Margin"),
                            ],
                        )
                        .properties(height=350)
                    )
                    st.altair_chart(chart, use_container_width=True)

                # Strict Score Honesty Callout
                st.markdown(
                    f"""
                    <div class="honesty-callout">
                        <strong>⚠️ Score Interpretation & ML Transparency:</strong><br/>
                        {result['disclaimer']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # -------------------------------------------------------------
                # 8. Feature Vector Inspector (Expander)
                # -------------------------------------------------------------
                with st.expander("🔬 View Extracted 80-Dimensional MFCC Feature Vector"):
                    st.markdown(
                        "The model processes **80 normalized acoustic features**: 40 temporal MFCC means "
                        "representing the average spectral distribution, followed by 40 temporal MFCC standard deviations "
                        "representing dynamic temporal modulation."
                    )

                    feat_names = result["feature_names"]
                    feat_vals = result["features"]

                    means = feat_vals[:40]
                    stds = feat_vals[40:]

                    feat_cols1, feat_cols2 = st.columns(2)
                    with feat_cols1:
                        st.markdown("**MFCC 1–40 Temporal Means (Spectral Envelope)**")
                        mean_df = pd.DataFrame({
                            "Coefficient": [f"MFCC {i+1}" for i in range(40)],
                            "Mean": means,
                        })
                        st.line_chart(mean_df.set_index("Coefficient"), height=220)

                    with feat_cols2:
                        st.markdown("**MFCC 1–40 Temporal StDevs (Modulation Energy)**")
                        std_df = pd.DataFrame({
                            "Coefficient": [f"MFCC {i+1}" for i in range(40)],
                            "StdDev": stds,
                        })
                        st.line_chart(std_df.set_index("Coefficient"), height=220)

                    # Top 10 extreme features
                    st.markdown("**Extreme Magnitude Coefficients in Input Vector:**")
                    all_feats_df = pd.DataFrame({
                        "Feature Column": feat_names,
                        "Value": feat_vals,
                        "Abs Value": np.abs(feat_vals),
                    }).sort_values(by="Abs Value", ascending=False)
                    st.dataframe(
                        all_feats_df[["Feature Column", "Value"]].head(10),
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Error during audio classification: {e}")
                st.exception(e)
else:
    st.info("👆 Please upload an audio file or select a demonstration sound above to begin analysis.")


# -----------------------------------------------------------------------------
# 9. Technical Reference, Taxonomy & Edge Cases (Expandable Tabs)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📚 Technical Documentation & System Reference")

tab_taxonomy, tab_pipeline, tab_benchmarks, tab_constraints = st.tabs([
    "🏷️ UrbanSound8K Taxonomy",
    "⚙️ Audio Preprocessing",
    "📈 Phase 3 Benchmarks",
    "⚠️ Real-World Limitations",
])

with tab_taxonomy:
    st.markdown("#### UrbanSound8K 10 Acoustic Classes")
    tax_rows = []
    for cls_name, info in CLASS_METADATA.items():
        tax_rows.append({
            "Icon": info["icon"],
            "Class Name": info["label"],
            "Dataset Identifier": cls_name,
            "Category": info["category"],
            "Acoustic Characteristics": info["description"],
        })
    st.table(pd.DataFrame(tax_rows))

with tab_pipeline:
    st.markdown("#### Feature Extraction Architecture")
    st.markdown(
        """
        The preprocessing module standardizes raw audio into an 80-dimensional numeric representation:
        1. **Mono Downmixing**: Multi-channel signals (stereo, 5.1) are averaged across channels using `librosa.to_mono`.
        2. **Sampling Rate Standardization**: Converted from native rate (e.g. 44.1 kHz, 48 kHz) to **22,050 Hz** using Kaiser window interpolation.
        3. **Short Signal Handling**: Clips with fewer than 2,048 samples are padded with zeros to ensure complete window framing.
        4. **Short-Time Fourier Transform (STFT)**: Frame length $N = 2048$, Hop size $H = 512$ samples (~23 ms resolution).
        5. **Mel-Frequency Filterbanks**: 40 triangular filters mapped along the perceptually motivated nonlinear Mel scale.
        6. **Statistical Aggregation**:
           - **40 Temporal Means**: $\\mu_k = \\frac{1}{T} \\sum_{t=1}^T M_{k,t}$ (Spectral shape).
           - **40 Temporal Standard Deviations**: $\\sigma_k = \\sqrt{\\frac{1}{T} \\sum_{t=1}^T (M_{k,t} - \\mu_k)^2}$ (Temporal variation).
        7. **Finite Sanitization**: Every coefficient is checked with `np.nan_to_num` ensuring strictly finite float32 arrays.
        """
    )

with tab_benchmarks:
    st.markdown("#### 10-Fold Leakage-Free Cross-Validation Results")
    st.markdown(
        """
        Models were trained strictly across the 10 official UrbanSound8K folds. In each fold, standardizers
        were fitted only on the 9 training folds, preventing feature leakage into the evaluation fold.
        """
    )
    bench_df = pd.DataFrame([
        {
            "Model Architecture": "Random Forest (300 Trees)",
            "Macro F1": "68.68%",
            "Accuracy": "66.83%",
            "Macro Precision": "70.77%",
            "Macro Recall": "67.84%",
            "Fold Std Dev": "± 5.86%",
            "Selection Rationale": "Superior Precision & Leaf Robustness",
        },
        {
            "Model Architecture": "SVM Pipeline (RBF C=10)",
            "Macro F1": "68.58%",
            "Accuracy": "67.03%",
            "Macro Precision": "68.46%",
            "Macro Recall": "69.50%",
            "Fold Std Dev": "± 5.76%",
            "Selection Rationale": "Consistent Global Hyperplane Margin",
        },
    ])
    st.table(bench_df)
    st.caption(
        "Paired t-test between fold Macro F1 scores yielded p = 0.9669 (no statistically significant difference). "
        "Random Forest was chosen as deployable winner due to higher precision on rare events."
    )

with tab_constraints:
    st.markdown("#### Real-World Deployment Considerations & Acoustic Edge Cases")
    st.markdown(
        """
        - **Monophonic Assumption**: UrbanSound8K models are single-label classifiers. In live municipal environments, multiple acoustic events (e.g. siren + car horn + drilling) often occur simultaneously.
        - **Microphone Distance & Attenuation**: Sound energy decays following the inverse-square law ($1/r^2$). Distant events exhibit diminished high frequencies due to atmospheric absorption.
        - **Acoustic Reverberation**: Enclosed spaces, urban canyons, and glass facades introduce multipath reflections, which can blur temporal standard deviations.
        - **Digital Clipping**: Excessively loud audio (e.g. close-range gunshots or sirens) may clip the A/D converter, introducing spurious high-frequency harmonics.
        """
    )
