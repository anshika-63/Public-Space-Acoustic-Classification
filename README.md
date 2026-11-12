# Public-Space Acoustic Event Classification (T254)

An end-to-end, reproducible machine learning system and interactive web application for automated detection and classification of environmental and public-space acoustic events, built on the 10-class **UrbanSound8K** dataset.

[![Python 3.9](https://img.shields.io/badge/python-3.9.6-blue.svg)](https://www.python.org/downloads/release/python-396/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg)](https://scikit-learn.org/)
[![Librosa](https://img.shields.io/badge/librosa-0.11.0-orange.svg)](https://librosa.org/)
[![Tests](https://img.shields.io/badge/pytest-26%20passed-brightgreen.svg)](https://pytest.org)

---

## 1. Project Overview

Municipal authorities, urban planners, and public security systems require automated solutions to monitor environmental soundscapes, detect safety hazards, and enforce noise ordinances. The **Public-Space Acoustic Event Classification** project implements a complete, leak-free machine learning workflow:
* High-fidelity audio preprocessing and standardized **80-dimensional MFCC feature extraction**.
* Full dataset processing across all **8,732 audio clips** of the official UrbanSound8K benchmark.
* **Leakage-free 10-fold cross-validation** using the official predefined folds.
* Comparative evaluation between an ensemble **Random Forest Classifier** and a **Support Vector Machine (SVM) Pipeline**.
* A responsive, high-contrast **Streamlit web application** providing real-time inference, container metadata inspection, top-3 ranked hypotheses, full 10-class score charts, and strict score honesty.

---

## 2. Problem Statement & Motivation

Public spaces—such as transit hubs, public parks, street corridors, and pedestrian zones—generate complex, dynamic acoustic environments. Conventional surveillance predominantly relies on optical cameras, which suffer from line-of-sight obstructions, lighting dependencies, and severe privacy concerns.

Acoustic event classification offers continuous, non-visual situational awareness:
1. **Public Safety**: Immediate detection of gunshots, car horns, and emergency sirens.
2. **Noise Ordinance Compliance**: Monitoring industrial drilling, jackhammers, and loud HVAC air conditioning units.
3. **Smart City Operations**: Mapping urban sound patterns (traffic idling, children playing, street buskers) without capturing identifiable personal conversations.

---

## 3. Project Objectives

1. **Standardized Preprocessing**: Ingest variable-duration, multi-channel environmental audio and standardize to single-channel 22,050 Hz PCM float32 arrays with NaN/Inf sanitization.
2. **Acoustic Feature Extraction**: Compute 40 Mel-Frequency Cepstral Coefficients (MFCCs) aggregated across time frames via mean and standard deviation to yield an 80-dimensional feature vector.
3. **Leakage-Free Cross-Validation**: Adhere strictly to the 10 predefined UrbanSound8K folds to evaluate generalization to unseen recording sessions and microphone setups.
4. **Model Comparison & Rigorous Statistical Testing**: Evaluate Random Forest against an RBF-kernel SVM using paired $t$-tests, Wilcoxon signed-rank tests, and McNemar tests.
5. **Interactive Frontend with Score Honesty**: Deliver an intuitive Streamlit dashboard displaying honest metrics (probability estimates for Random Forest, raw signed hyperplane distances for SVM) without misleading probability distortions.

---

## 4. Dataset Details: UrbanSound8K

The system is trained and evaluated on the official **UrbanSound8K** benchmark:
* **Total Samples**: 8,732 labeled audio excerpts ($\le 4.0$ seconds each).
* **Audio Length**: Over 8.7 hours of urban sound recordings.
* **Total Classes**: 10 distinct environmental sound categories.
* **Official Folds**: Folds 1 through 10, pre-allocated to avoid placing slices from the same original recording into both training and evaluation sets.
* **Feature Schema**: 84 columns (4 metadata columns: `slice_file_name`, `fold`, `classID`, `class` + 80 numeric feature columns).

### The 10 Acoustic Classes

| Class ID | Class Label | Category | Acoustic Characteristics |
| :---: | :--- | :--- | :--- |
| `0` | **air_conditioner** | Mechanical / HVAC | Steady low-frequency compressor hum and broadband airflow. |
| `1` | **car_horn** | Urban Traffic | Dual-tone harmonic blasts with sudden onset. |
| `2` | **children_playing** | Social / Human | Dynamic, high-frequency vocal bursts, shouts, and laughter. |
| `3` | **dog_bark** | Biological / Animal | Repetitive impulsive acoustic transients with harmonic overtones. |
| `4` | **drilling** | Industrial / Tool | Rapid mechanical percussion with high-frequency friction noise. |
| `5` | **engine_idling** | Traffic / Automotive | Low-frequency periodic combustion cycles from stationary vehicles. |
| `6` | **gun_shot** | Safety / Hazard | Supersonic impulse wave with steep onset and rapid exponential decay. |
| `7` | **jackhammer** | Construction | High-energy, low-frequency pneumatic hammer impacts. |
| `8` | **siren** | Emergency Services | Frequency-modulated sinusoidal sweeps (wail/yelp). |
| `9` | **street_music** | Cultural / Social | Multi-instrumental harmonic and rhythmic melodies. |

---

## 5. Technology Stack

* **Programming Language**: Python 3.9.6
* **Audio DSP & Feature Extraction**: `librosa` (v0.11.0), `soundfile` (v0.13.1), `scipy` (v1.13.1)
* **Machine Learning & Evaluation**: `scikit-learn` (v1.6.1), `joblib` (v1.5.3), `pandas` (v2.3.3), `numpy` (v2.0.2)
* **Dataset Ingestion**: `datasets` (v4.5.0), `pyarrow` (v21.0.0), `tqdm` (v4.70.1)
* **Frontend Web Dashboard**: `streamlit` (v1.50.0), `altair` (v5.5.0), `matplotlib` (v3.9.4)
* **Testing & Quality Assurance**: `pytest` (v8.4.2), `streamlit.testing.v1.AppTest`

---

## 6. Project Directory Structure

```text
Public-Space-Acoustic-Classification/
├── .streamlit/
│   └── config.toml               # Streamlit server and headless configuration
├── data/
│   ├── processed_features.csv    # 100-sample pilot dataset (Phase 1 validation)
│   └── urbansound8k_features_80.csv # Full 8,732-sample 80-feature dataset
├── models/
│   └── phase3/                   # Serialized deployable models and CV metrics
│       ├── best_acoustic_model.joblib
│       ├── random_forest_model.joblib
│       ├── svm_pipeline.joblib
│       ├── evaluation_metrics.json
│       ├── model_comparison.csv
│       ├── fold_metrics.csv
│       ├── oof_predictions_random_forest.csv
│       └── oof_predictions_svm.csv
├── src/
│   ├── __init__.py
│   ├── dataset_builder.py        # Hugging Face extraction and checkpointing
│   ├── inference.py              # Modular inference engine and demo synthesis
│   ├── model_evaluation.py       # Metrics calculation and statistical tests
│   ├── model_training.py         # 10-fold CV and final model refitting
│   ├── preprocessing.py          # 80 MFCC feature extraction pipeline
│   └── run_phase3.py             # Phase 3 orchestration script
├── test/
│   ├── test_app.py               # Inference and Streamlit AppTest unit tests
│   ├── test_model_training.py    # Leakage-free CV and model integrity tests
│   └── test_preprocessing.py     # Signal DSP and feature vector tests
├── app.py                        # Streamlit web application frontend
├── pytest.ini                    # Pytest configuration (pythonpath = .)
├── requirements.txt              # Verified pinned dependencies
└── README.md                     # Comprehensive project documentation
```

---

## 7. Audio Preprocessing Pipeline

The preprocessing workflow in `src/preprocessing.py` converts raw audio streams into clean numerical arrays:
1. **Container Parsing**: Decodes WAV, MP3, FLAC, and OGG streams into float32 waveforms using `soundfile` and `librosa`.
2. **Channel Downmixing**: Multi-channel (stereo, 5.1) audio signals are downmixed to monophonic via `librosa.to_mono`.
3. **Short Signal Handling**: Clips shorter than 2,048 samples (~93 ms) are padded with zero-value reflection padding to ensure valid STFT window framing.
4. **Sampling Rate Standardization**: Any source sampling rate (e.g. 44.1 kHz, 48 kHz) is resampled to **22,050 Hz** using high-quality Kaiser window interpolation (`librosa.resample`).

---

## 8. Feature Extraction: 80 MFCC Representation

For every clip, the Short-Time Fourier Transform (STFT) is calculated with an FFT window of $N = 2048$ samples and a hop size of $H = 512$ samples (~23 ms resolution). Spectral energies are passed through 40 triangular Mel-scale filterbanks to extract 40 Mel-Frequency Cepstral Coefficients ($k \in [1, 40]$) across $T$ temporal frames:

$$\mu_k = \frac{1}{T} \sum_{t=1}^T M_{k,t} \quad \text{(40 Temporal Means)}$$

$$\sigma_k = \sqrt{\frac{1}{T} \sum_{t=1}^T \left(M_{k,t} - \mu_k\right)^2} \quad \text{(40 Temporal Standard Deviations)}$$

* **Feature Vector Length**: Exactly **80 numeric features** (`mfcc_mean_1` .. `mfcc_mean_40`, `mfcc_std_1` .. `mfcc_std_40`).
* **Sanitization**: All non-finite coefficients are converted to finite numeric zero with `np.nan_to_num`.

---

## 9. Machine Learning Models

### 1. Random Forest Classifier
* **Hyperparameters**: 300 estimators, unconstrained maximum tree depth, `class_weight='balanced'`, random seed 42.
* **Tree Aggregation**: 882,230 total decision nodes across the ensemble.
* **Scoring Mechanism**: Leaf-level empirical voting fractions (`predict_proba()`), reflecting tree agreement.

### 2. Support Vector Machine (SVM) Pipeline
* **Pipeline Structure**: `Pipeline([('scaler', StandardScaler()), ('svc', SVC(...))])`.
* **Hyperparameters**: Radial Basis Function (RBF) kernel, regularization parameter $C = 10.0$, `gamma='scale'`, `class_weight='balanced'`, `probability=False`.
* **Scoring Mechanism**: Raw signed geometric distance from the sample to the separating hyperplane (`decision_function()`).

---

## 10. 10-Fold Cross-Validation Methodology

Evaluating environmental audio classifiers via standard random train/test splits leads to severe over-optimistic performance (data leakage) because slices from the same original recording sound nearly identical.

To prevent leakage, this project strictly utilizes the **10 predefined UrbanSound8K folds**:
* In each iteration $k \in \{1, \dots, 10\}$, fold $k$ is held out as the unseen test set, while the remaining 9 folds form the training set.
* For the SVM pipeline, `StandardScaler` is fitted **exclusively on the 9 training folds** and applied to transform fold $k$, preventing distribution leakage.
* Every clip receives exactly one out-of-fold (OOF) prediction, ensuring an unbiased full-dataset evaluation ($N = 8,732$).

---

## 11. Model Evaluation & Benchmark Results

### 11.1 Full Dataset 10-Fold OOF Performance

| Evaluation Metric | Random Forest (300 Trees) | SVM Pipeline (RBF C=10) | Difference (RF - SVM) |
| :--- | :---: | :---: | :---: |
| **Macro F1-Score** | **68.68%** | 68.58% | +0.10% |
| **Overall Accuracy** | 66.83% | **67.03%** | -0.20% |
| **Weighted F1-Score** | **66.85%** | 66.46% | +0.39% |
| **Macro Precision** | **70.77%** | 68.46% | **+2.31%** |
| **Macro Recall** | 67.84% | **69.50%** | -1.66% |
| **Fold Mean Accuracy** | 66.80% $\pm$ 5.18% | 67.03% $\pm$ 5.08% | -0.23% |
| **Fold Mean Macro F1** | 68.12% $\pm$ 4.92% | 68.16% $\pm$ 4.89% | -0.04% |

### 11.2 Statistical Hypothesis Testing

* **Paired $t$-test (Fold Macro F1)**: $t = -0.0426$, $p = 0.9669$ (No statistically significant difference at $\alpha = 0.05$).
* **Paired $t$-test (Fold Accuracy)**: $t = -0.2630$, $p = 0.7985$ (No statistically significant difference).
* **Wilcoxon Signed-Rank Test**: $W = 26.0$, $p = 0.9219$.
* **Sample-Level McNemar Test**: $\chi^2 = 0.1523$, $p = 0.6964$ ($b = 942$ discordant RF-correct / SVM-incorrect vs. $c = 959$ discordant SVM-correct / RF-incorrect).

### 11.3 Model Selection Decision
**Winner:** **Random Forest Classifier** was selected as the deployable model. While overall accuracy is neck-and-neck, Random Forest delivers noticeably higher Macro Precision (+2.31%), reducing false alarms on critical transient sounds (gunshots, car horns), while offering leaf-level robustness against noisy outdoor acoustic fluctuations.

---

## 12. Streamlit Dashboard Features

The web frontend (`app.py`) provides an interactive interface for environmental audio inspection:
1. **Model Selection**: Switch dynamically between Random Forest (Recommended), SVM Pipeline, and Deployable Model.
2. **Audio Input**:
   * Upload custom audio files (`.wav`, `.mp3`, `.flac`, `.ogg`).
   * Built-in interactive sound synthesizer for immediate demonstration without external files (Siren, Car Horn, Engine Idling, Drilling, Gun Shot).
3. **Signal Properties Inspector**: Displays clip duration, native sampling rate, audio channels, and container format.
4. **Hero Prediction Banner**: Highlights the winning acoustic event, category badge, description, latency in milliseconds, and primary score.
5. **Top-3 Ranked Hypotheses**: Clean comparison cards showing the top three candidates.
6. **Strict Score Honesty Chart**:
   * **Random Forest**: Displays tree-voting probability estimates (0–100%) with explicit disclaimer that they are empirical voting fractions across 300 trees, not calibrated Bayesian probabilities.
   * **SVM**: Displays raw signed decision margins (distance to hyperplane), with an explicit note that they are uncalibrated geometric margins and have not been artificially squeezed through softmax functions.
7. **MFCC Feature Inspector**: Expandable visualization displaying the 40 spectral means and 40 modulation standard deviations for the analyzed audio clip.

---

## 13. Supported Audio Formats

Audio decoding is powered by `libsndfile` via `soundfile` and `librosa`:

* **WAV**: Uncompressed linear PCM (8-bit, 16-bit, 24-bit, 32-bit float).
* **FLAC**: Lossless compressed audio.
* **OGG**: Ogg Vorbis compressed audio streams.
* **MP3**: MPEG-1/2 Audio Layer III compressed audio.

*All supported formats are automatically downmixed to mono and resampled to 22,050 Hz during preprocessing.*

---

## 14. Installation & Setup

### Prerequisites
* Operating System: macOS, Linux, or Windows (WSL recommended).
* Python: **3.9.6** (or compatible 3.9.x environment).

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd Public-Space-Acoustic-Classification
```

### Step 2: Set Up Virtual Environment
```bash
python3.9 -m venv .venv
```

### Step 3: Activate Virtual Environment
* **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```
* **Windows (Command Prompt / PowerShell)**:
  ```cmd
  .venv\Scripts\activate
  ```

### Step 4: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Download Trained Model Artifacts

Because large model binaries (> 100 MB) are intentionally excluded from normal Git commits to keep the repository lightweight, trained models are provided via GitHub Releases. The Streamlit dashboard requires these model files inside `models/phase3/` before launching:

1. Open the repository's [GitHub Releases page](https://github.com/anshika-63/Public-Space-Acoustic-Classification/releases).
2. Download these two assets from the **v1.0.0** Release:
   * `random_forest_model.joblib`
   * `svm_pipeline.joblib`
3. Ensure the destination directory exists and place both files inside:
   ```bash
   mkdir -p models/phase3
   # Place downloaded random_forest_model.joblib and svm_pipeline.joblib in models/phase3/
   ```
4. Create `best_acoustic_model.joblib` (which is identical to `random_forest_model.joblib`) using:
   ```bash
   cp models/phase3/random_forest_model.joblib models/phase3/best_acoustic_model.joblib
   ```

---

## 15. Launching the Dashboard

Ensure the virtual environment is activated, then launch the Streamlit server:

```bash
.venv/bin/streamlit run app.py
```

The application will start and provide a local browser URL:

```text
Local URL: http://localhost:8501
```

Open `http://localhost:8501` in any modern web browser (Chrome, Firefox, Safari, Edge).

### Stopping the Server
To terminate the local Streamlit development server, press `Ctrl + C` in the active terminal session.

---

## 16. Running the Automated Test Suite

The project includes 26 automated unit and integration tests covering signal processing, model cross-validation, inference logic, and Streamlit UI workflows:

```bash
.venv/bin/pytest test/ -v
```

Expected output:
```text
============================== 26 passed in 2.0s ===============================
```

### Test Suite Structure
* `test/test_preprocessing.py`: Verifies mono downmixing, 22.05 kHz resampling, silence padding, and 80-feature extraction.
* `test/test_model_training.py`: Verifies fold isolation, absence of feature leakage, and scaler independence.
* `test/test_app.py`: Verifies model loading, score honesty (RF tree-votes vs SVM decision margins), audio metadata extraction, and Streamlit AppTest headless execution.

---

## 17. Limitations & Future Improvements

1. **Monophonic Assumption**: UrbanSound8K is single-label annotated. In real-world public spaces, acoustic events overlap (e.g. sirens sounding while a car horn honks and an engine idles). Future iterations should incorporate polyphonic sound event detection (SED) architectures.
2. **Environmental Noise Floor**: Severe background traffic noise or wind shear can mask subtle high-frequency acoustic signatures (e.g. distant gunshots or children playing).
3. **Acoustic Reverberation**: Urban street canyons and enclosed spaces cause multipath reflections that alter temporal standard deviations. Adding data augmentation with simulated room impulse responses (RIR) will improve acoustic robustness.
4. **Deep Learning Benchmarking**: Future work can contrast this 80-feature MFCC classical ML baseline with pre-trained audio transformers such as PANNs (CNN14) or Audio Spectrogram Transformer (AST).

---

## 18. Academic & Research Disclaimer

This software and its associated models are developed for academic research, education, and algorithmic demonstration purposes. The models are **not certified for mission-critical, life-safety, or automated law enforcement deployment** without independent sensor calibration, ambient acoustic testing, and human-in-the-loop oversight.
