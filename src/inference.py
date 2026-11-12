"""
Inference and Prediction Pipeline Module
T254 - Public-Space Acoustic Event Classification

Provides modular inference helpers for loading trained Phase 3 models,
extracting acoustic metadata, computing 80 MFCC features, and generating
honest, leakage-free class predictions for the Streamlit web application.
"""

import io
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import soundfile as sf

# Ensure project root is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import get_feature_names, preprocess_audio

MODELS_DIR = "models/phase3"

# Official 10 UrbanSound8K classes
CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]

# Human-readable class descriptions and icons
CLASS_METADATA = {
    "air_conditioner": {
        "label": "Air Conditioner",
        "icon": "❄️",
        "category": "Mechanical / Ambient",
        "description": "Continuous low-to-mid frequency hum from HVAC compressor and fan units.",
    },
    "car_horn": {
        "label": "Car Horn",
        "icon": "🚗",
        "category": "Urban Traffic",
        "description": "Sharp harmonic multi-tone warning blast produced by vehicle horn diaphragms.",
    },
    "children_playing": {
        "label": "Children Playing",
        "icon": "🏃",
        "category": "Social / Human",
        "description": "Dynamic high-frequency vocalizations, laughter, shouting, and footsteps.",
    },
    "dog_bark": {
        "label": "Dog Bark",
        "icon": "🐕",
        "category": "Biological / Animal",
        "description": "Repetitive, high-energy acoustic impulses with rich harmonic overtones.",
    },
    "drilling": {
        "label": "Drilling",
        "icon": "🔩",
        "category": "Construction / Industrial",
        "description": "High-speed rotary percussion vibration with broadband mechanical friction.",
    },
    "engine_idling": {
        "label": "Engine Idling",
        "icon": "🚛",
        "category": "Urban Traffic",
        "description": "Rhythmic low-frequency combustion cycles from stationary trucks or cars.",
    },
    "gun_shot": {
        "label": "Gun Shot",
        "icon": "💥",
        "category": "Public Safety / Hazard",
        "description": "Extremely fast supersonic acoustic shockwave impulse followed by rapid decay.",
    },
    "jackhammer": {
        "label": "Jackhammer",
        "icon": "🔨",
        "category": "Construction / Industrial",
        "description": "Pneumatic reciprocating hammer impacts with heavy low-frequency shock vibrations.",
    },
    "siren": {
        "label": "Siren",
        "icon": "🚨",
        "category": "Emergency / Public Safety",
        "description": "Continuous pitch-modulated acoustic warning tones (wail, yelp, or two-tone).",
    },
    "street_music": {
        "label": "Street Music",
        "icon": "🎷",
        "category": "Social / Cultural",
        "description": "Structured melodic and rhythmic acoustic soundscapes from buskers and instruments.",
    },
}


def load_model(
    model_name: str = "random_forest",
    models_dir: str = MODELS_DIR,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a serialized Phase 3 model and its associated training metadata.

    Parameters:
        model_name: Identifier for the model:
                    - 'random_forest' or 'rf' -> random_forest_model.joblib
                    - 'svm' or 'svc' -> svm_pipeline.joblib
                    - 'best' or 'best_acoustic_model' -> best_acoustic_model.joblib
        models_dir: Directory where serialized joblib artifacts are stored.

    Returns:
        Tuple of (model_object, metadata_dict).
    """
    norm_name = model_name.lower().strip()
    if norm_name in ["rf", "random_forest", "random-forest"]:
        filename = "random_forest_model.joblib"
    elif norm_name in ["svm", "svc", "svm_pipeline"]:
        filename = "svm_pipeline.joblib"
    elif norm_name in ["best", "best_model", "best_acoustic_model"]:
        filename = "best_acoustic_model.joblib"
    else:
        raise ValueError(
            f"Unknown model identifier '{model_name}'. "
            "Supported values: 'random_forest', 'svm', 'best_acoustic_model'."
        )

    file_path = os.path.join(models_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model artifact not found at: {file_path}")

    artifact = joblib.load(file_path)
    if not isinstance(artifact, dict):
        raise ValueError(f"Corrupted artifact in {file_path}: expected dictionary package.")

    # RF and best_acoustic_model store under 'model', SVM pipeline stores under 'pipeline'
    model = artifact.get("model") or artifact.get("pipeline")
    if model is None:
        raise KeyError(f"Neither 'model' nor 'pipeline' key found in {file_path}.")

    metadata = artifact.get("metadata", {})
    return model, metadata


def get_audio_metadata(audio_bytes: bytes) -> Dict[str, Union[float, int, str]]:
    """
    Inspect raw audio bytes and extract container/format metadata.

    Parameters:
        audio_bytes: Binary audio data (WAV, MP3, FLAC, OGG, etc.)

    Returns:
        Dict with keys: duration_seconds, sample_rate, channels, frames, format, subtype.
    """
    if not audio_bytes or len(audio_bytes) == 0:
        raise ValueError("Audio input buffer is empty.")

    try:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as sfile:
            sample_rate = int(sfile.samplerate)
            channels = int(sfile.channels)
            frames = int(sfile.frames)
            duration = float(frames) / float(sample_rate) if sample_rate > 0 else 0.0
            fmt = str(sfile.format)
            subtype = str(sfile.subtype)

            return {
                "duration_seconds": round(duration, 3),
                "sample_rate": sample_rate,
                "channels": channels,
                "channel_mode": "Mono" if channels == 1 else "Stereo" if channels == 2 else f"{channels}-ch",
                "frames": frames,
                "format": fmt,
                "subtype": subtype,
                "file_size_bytes": len(audio_bytes),
            }
    except Exception as e:
        raise RuntimeError(f"Could not parse audio format with soundfile: {e}") from e


def extract_features(
    audio_input: Union[bytes, bytearray, io.BytesIO, Tuple[np.ndarray, int]],
    n_mfcc: int = 40,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract 80 MFCC features (40 means + 40 standard deviations) matching Phase 1/2 schema.

    Parameters:
        audio_input: Raw audio bytes or (array, sr) tuple.
        n_mfcc: Number of MFCC coefficients (default 40 -> 80 features).

    Returns:
        Tuple of (feature_matrix_2d with shape (1, 80), feature_names_list).
    """
    feature_vec = preprocess_audio(audio_input, n_mfcc=n_mfcc)
    expected_len = 2 * n_mfcc

    if feature_vec.shape != (expected_len,):
        raise ValueError(
            f"Feature extraction failed: expected shape ({expected_len},), got {feature_vec.shape}"
        )

    if np.isnan(feature_vec).any() or np.isinf(feature_vec).any():
        raise ValueError("Feature vector contains non-finite values (NaN or Inf).")

    feature_names = get_feature_names(n_mfcc=n_mfcc)
    feature_matrix = feature_vec.reshape(1, -1)
    return feature_matrix, feature_names


def classify_audio(
    audio_input: Union[bytes, bytearray, io.BytesIO, Tuple[np.ndarray, int]],
    model: Any,
    metadata: Optional[Dict[str, Any]] = None,
    model_name: str = "random_forest",
) -> Dict[str, Any]:
    """
    Perform end-to-end classification with honest score reporting.

    For Random Forest:
        Scores are extracted via `predict_proba()`. Labeled clearly as
        'Random Forest Class Probability Estimates' (tree-voting proportions).

    For Support Vector Machine (SVM):
        Trained with probability=False. Scores are extracted via `decision_function()`.
        Labeled clearly as 'SVM Decision Margins' (raw signed distance to hyperplane).
        Never mislabeled or softmax-distorted into artificial probabilities.

    Parameters:
        audio_input: Raw audio bytes or audio tuple.
        model: Fitted scikit-learn model or Pipeline.
        metadata: Model metadata dictionary.
        model_name: Descriptive name of the model being used.

    Returns:
        Structured prediction dictionary with honest scores and Top-3 rankings.
    """
    t0 = time.perf_counter()

    # 1. Extract 80 MFCC features
    feats_2d, feat_names = extract_features(audio_input, n_mfcc=40)

    # 2. Extract model classes
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    elif metadata and "classes" in metadata:
        classes = list(metadata["classes"])
    else:
        classes = CLASSES

    # 3. Predict primary class label
    predicted_class = str(model.predict(feats_2d)[0])

    # 4. Extract honest scores
    class_scores = []
    if hasattr(model, "predict_proba"):
        # Random Forest: tree voting estimates
        probs = model.predict_proba(feats_2d)[0]
        score_type = "probability"
        score_unit = "%"
        score_label = "Random Forest Class Probability Estimates"
        disclaimer = (
            "Probability estimates represent empirical voting fractions across 300 decision trees. "
            "They are tree-vote proportions, not formally calibrated posterior probabilities."
        )
        for cls_name, prob in zip(classes, probs):
            class_scores.append({
                "class": cls_name,
                "label": CLASS_METADATA.get(cls_name, {}).get("label", cls_name),
                "icon": CLASS_METADATA.get(cls_name, {}).get("icon", "🔊"),
                "score": float(prob),
                "score_pct": float(prob * 100.0),
                "display_score": f"{prob * 100.0:.1f}%",
            })
    elif hasattr(model, "decision_function"):
        # SVM: raw signed distances to separating hyperplane
        margins = model.decision_function(feats_2d)[0]
        score_type = "margin"
        score_unit = "margin"
        score_label = "SVM Decision Margins (Signed Hyperplane Distance)"
        disclaimer = (
            "Scores represent raw signed distances to the separating hyperplanes. "
            "Higher positive values indicate stronger geometric separation support for that class. "
            "These are uncalibrated decision margins and have NOT been distorted with artificial softmax scaling."
        )
        for cls_name, margin in zip(classes, margins):
            class_scores.append({
                "class": cls_name,
                "label": CLASS_METADATA.get(cls_name, {}).get("label", cls_name),
                "icon": CLASS_METADATA.get(cls_name, {}).get("icon", "🔊"),
                "score": float(margin),
                "score_pct": None,
                "display_score": f"{margin:+.2f}",
            })
    else:
        raise AttributeError(
            f"Model {type(model)} does not support either predict_proba or decision_function."
        )

    # Sort descending by score
    class_scores.sort(key=lambda item: item["score"], reverse=True)
    top_3 = class_scores[:3]

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_METADATA.get(predicted_class, {}).get("label", predicted_class),
        "predicted_icon": CLASS_METADATA.get(predicted_class, {}).get("icon", "🔊"),
        "predicted_category": CLASS_METADATA.get(predicted_class, {}).get("category", "General"),
        "predicted_description": CLASS_METADATA.get(predicted_class, {}).get("description", ""),
        "top_score": top_3[0]["score"],
        "top_display_score": top_3[0]["display_score"],
        "score_type": score_type,
        "score_unit": score_unit,
        "score_label": score_label,
        "disclaimer": disclaimer,
        "top_3": top_3,
        "all_scores": class_scores,
        "features": feats_2d[0],
        "feature_names": feat_names,
        "latency_ms": round(latency_ms, 2),
        "model_name": model_name,
    }


def generate_demo_audio(
    event_type: str = "siren",
    duration: float = 2.0,
    sr: int = 22050,
) -> bytes:
    """
    Generate clean synthetic audio waveforms simulating characteristic urban acoustic events.
    Useful for immediate interactive testing in the UI when no upload file is available.

    Supported event types:
        - 'siren': Two-tone pitch-modulated frequency sweep
        - 'car_horn': Dual-tone harmonic horn chord with envelope decay
        - 'engine_idling': Low-frequency multi-harmonic rumble
        - 'gun_shot': Sharp acoustic transient impulse with exponential decay
        - 'drilling': High-frequency vibratory pulses with friction texture
    """
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)

    event_type = event_type.lower().strip()

    if event_type == "siren":
        # Sweeping siren tone oscillating between 600 Hz and 1100 Hz at 1.5 Hz
        mod_freq = 1.5
        inst_freq = 800.0 + 300.0 * np.sin(2 * np.pi * mod_freq * t)
        phase = 2 * np.pi * np.cumsum(inst_freq) / sr
        audio = 0.6 * np.sin(phase)

    elif event_type == "car_horn":
        # Dual-frequency vehicle horn (F4 349.2 Hz + A4 440 Hz)
        tone1 = np.sin(2 * np.pi * 349.2 * t)
        tone2 = 0.8 * np.sin(2 * np.pi * 440.0 * t)
        tone3 = 0.3 * np.sin(2 * np.pi * 880.0 * t)
        audio = (tone1 + tone2 + tone3) / 2.1
        # Envelope: brief attack and continuous sustain with slight tremor
        envelope = np.minimum(t / 0.05, 1.0) * (0.9 + 0.1 * np.sin(2 * np.pi * 12.0 * t))
        audio = audio * envelope * 0.7

    elif event_type == "engine_idling":
        # Low frequency rumble (45 Hz fundamental with harmonics and noise)
        f0 = 45.0
        audio = (
            0.5 * np.sin(2 * np.pi * f0 * t)
            + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
            + 0.15 * np.sin(2 * np.pi * 4 * f0 * t)
        )
        noise = np.random.normal(0, 0.1, num_samples).astype(np.float32)
        audio = (audio + noise) * 0.6

    elif event_type == "gun_shot":
        # Sudden acoustic shock impulse (0.01s rise) followed by fast exponential decay
        decay = np.exp(-12.0 * t)
        noise_burst = np.random.normal(0, 1.0, num_samples).astype(np.float32)
        low_thud = np.sin(2 * np.pi * 80.0 * t) * np.exp(-6.0 * t)
        audio = (0.7 * noise_burst + 0.5 * low_thud) * decay
        audio = np.clip(audio, -1.0, 1.0)

    elif event_type == "drilling":
        # Rapid periodic mechanical impacts at 35 Hz + broadband resonance
        impact_rate = 35.0
        buzz = np.sin(2 * np.pi * 520.0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * impact_rate * t))
        hiss = 0.2 * np.random.normal(0, 1.0, num_samples).astype(np.float32)
        audio = (buzz + hiss) * 0.6

    else:
        # Default pure 440 Hz test tone
        audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    # Normalize audio signal to prevent digital clipping
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.85

    # Encode to WAV format in memory
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
