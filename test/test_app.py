"""
Unit and Integration Tests for Inference Pipeline and Application Logic
T254 - Public-Space Acoustic Event Classification
"""

import io
import numpy as np
import pytest
import soundfile as sf

from src.inference import (
    CLASSES,
    CLASS_METADATA,
    classify_audio,
    extract_features,
    generate_demo_audio,
    get_audio_metadata,
    load_model,
)


def test_classes_and_metadata_consistency():
    """Verify that all 10 UrbanSound8K classes have complete UI metadata."""
    assert len(CLASSES) == 10
    for cls_name in CLASSES:
        assert cls_name in CLASS_METADATA
        meta = CLASS_METADATA[cls_name]
        assert "label" in meta
        assert "icon" in meta
        assert "category" in meta
        assert "description" in meta


def test_load_model_rf():
    """Verify loading Random Forest model artifact and metadata."""
    model, metadata = load_model("random_forest")
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert model.n_features_in_ == 80
    assert len(model.classes_) == 10
    assert metadata["model_type"] == "random_forest"
    assert metadata["n_features"] == 80


def test_load_model_svm():
    """Verify loading SVM pipeline artifact and verify it lacks predict_proba."""
    pipeline, metadata = load_model("svm")
    assert pipeline is not None
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "decision_function")
    assert not hasattr(pipeline, "predict_proba")  # probability=False
    assert "scaler" in pipeline.named_steps
    assert "svc" in pipeline.named_steps
    assert metadata["model_type"] == "svm"


def test_load_model_best():
    """Verify loading deployable best_acoustic_model artifact."""
    model, metadata = load_model("best")
    assert model is not None
    assert hasattr(model, "predict")
    assert metadata["n_samples"] == 8732


def test_load_model_invalid():
    """Verify that unsupported model identifiers raise ValueError."""
    with pytest.raises(ValueError, match="Unknown model identifier"):
        load_model("neural_network")


def test_get_audio_metadata():
    """Verify metadata inspection of valid synthesized WAV bytes."""
    demo_wav = generate_demo_audio("siren", duration=1.5, sr=22050)
    meta = get_audio_metadata(demo_wav)

    assert meta["format"] == "WAV"
    assert meta["sample_rate"] == 22050
    assert meta["channels"] == 1
    assert meta["channel_mode"] == "Mono"
    assert 1.4 <= meta["duration_seconds"] <= 1.6
    assert meta["file_size_bytes"] > 0


def test_get_audio_metadata_errors():
    """Verify error handling on empty or corrupted audio bytes."""
    with pytest.raises(ValueError, match="empty"):
        get_audio_metadata(b"")

    with pytest.raises(RuntimeError, match="Could not parse audio"):
        get_audio_metadata(b"CORRUPTED_NOT_AUDIO_BYTES_12345")


def test_extract_features():
    """Verify 80-feature MFCC vector extraction from audio bytes."""
    demo_wav = generate_demo_audio("car_horn", duration=1.0, sr=22050)
    feats_2d, feat_names = extract_features(demo_wav, n_mfcc=40)

    assert feats_2d.shape == (1, 80)
    assert len(feat_names) == 80
    assert feat_names[0] == "mfcc_mean_1"
    assert feat_names[40] == "mfcc_std_1"
    assert not np.isnan(feats_2d).any()
    assert not np.isinf(feats_2d).any()


def test_classify_audio_random_forest_honesty():
    """Verify Random Forest classification returns honest probability estimates."""
    model, metadata = load_model("random_forest")
    demo_wav = generate_demo_audio("siren", duration=1.5, sr=22050)

    result = classify_audio(demo_wav, model, metadata, model_name="Random Forest")

    assert result["predicted_class"] in CLASSES
    assert result["score_type"] == "probability"
    assert result["score_unit"] == "%"
    assert "Probability Estimates" in result["score_label"]
    assert "tree-vote proportions" in result["disclaimer"].lower()

    # Probability sum should be 1.0 ± 1e-4
    total_prob = sum(s["score"] for s in result["all_scores"])
    assert abs(total_prob - 1.0) < 1e-4

    # Top-3 should be sorted descending
    top_3 = result["top_3"]
    assert len(top_3) == 3
    assert top_3[0]["score"] >= top_3[1]["score"] >= top_3[2]["score"]
    assert top_3[0]["class"] == result["predicted_class"]
    assert result["latency_ms"] > 0.0


def test_classify_audio_svm_honesty():
    """Verify SVM classification returns honest decision margins and never fake probabilities."""
    model, metadata = load_model("svm")
    demo_wav = generate_demo_audio("engine_idling", duration=1.5, sr=22050)

    result = classify_audio(demo_wav, model, metadata, model_name="SVM")

    assert result["predicted_class"] in CLASSES
    assert result["score_type"] == "margin"
    assert result["score_unit"] == "margin"
    assert "SVM Decision Margins" in result["score_label"]
    assert "hyperplane" in result["disclaimer"].lower()
    assert "not been distorted" in result["disclaimer"].lower()

    # Scores should not sum to 1.0 (they are raw hyperplane distances)
    scores = [s["score"] for s in result["all_scores"]]
    assert len(scores) == 10
    assert abs(sum(scores) - 1.0) > 0.1  # Confirms uncalibrated margins

    # Top-3 should be sorted descending
    top_3 = result["top_3"]
    assert len(top_3) == 3
    assert top_3[0]["score"] >= top_3[1]["score"] >= top_3[2]["score"]
    assert top_3[0]["class"] == result["predicted_class"]


def test_generate_demo_audio_all_types():
    """Verify synthetic waveforms are correctly generated for all supported demo events."""
    event_types = ["siren", "car_horn", "engine_idling", "gun_shot", "drilling"]
    for event in event_types:
        wav_bytes = generate_demo_audio(event, duration=1.0, sr=22050)
        assert len(wav_bytes) > 1000
        data, sr = sf.read(io.BytesIO(wav_bytes))
        assert sr == 22050
        assert len(data) == 22050
        assert not np.isnan(data).any()


def test_streamlit_app_ui_execution():
    """Verify end-to-end rendering and classification workflow in Streamlit AppTest."""
    import os
    os.environ["HOME"] = "/tmp"
    os.environ["MPLCONFIGDIR"] = "/tmp"
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Initial render failed with exception: {at.exception}"

    # Click classify button in demo mode and verify execution
    if len(at.button) > 0:
        at.button[0].click().run()
        assert not at.exception, f"Classification cycle failed with exception: {at.exception}"
