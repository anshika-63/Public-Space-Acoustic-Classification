"""
Unit tests for Audio Preprocessing Module
T254 - Public-Space Acoustic Event Classification
"""

import io
import numpy as np
import pytest
import soundfile as sf

from src.preprocessing import get_feature_names, preprocess_audio


def generate_sine_wave(freq=440.0, duration=1.0, sr=44100):
    """Helper to generate a clean sinusoidal audio signal."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_feature_names_count_and_format():
    """Verify that get_feature_names produces exactly 80 unique, properly formatted columns."""
    names = get_feature_names(n_mfcc=40)
    assert len(names) == 80
    assert len(set(names)) == 80  # All unique
    assert names[0] == "mfcc_mean_1"
    assert names[39] == "mfcc_mean_40"
    assert names[40] == "mfcc_std_1"
    assert names[79] == "mfcc_std_40"


def test_mono_audio_array():
    """Test feature extraction from a mono audio array."""
    sr = 44100
    audio = generate_sine_wave(freq=440.0, duration=1.0, sr=sr)
    features = preprocess_audio((audio, sr), n_mfcc=40)

    assert isinstance(features, np.ndarray)
    assert features.shape == (80,)
    assert features.dtype == np.float32
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


def test_stereo_audio_array():
    """Test stereo downmix and feature extraction."""
    sr = 44100
    channel1 = generate_sine_wave(freq=440.0, duration=1.0, sr=sr)
    channel2 = generate_sine_wave(freq=880.0, duration=1.0, sr=sr)

    # Standard (samples, channels)
    stereo = np.column_stack([channel1, channel2])
    features = preprocess_audio((stereo, sr), n_mfcc=40)
    assert features.shape == (80,)
    assert not np.isnan(features).any()

    # Transposed (channels, samples)
    stereo_transposed = stereo.T
    features_t = preprocess_audio((stereo_transposed, sr), n_mfcc=40)
    assert features_t.shape == (80,)
    assert not np.isnan(features_t).any()


def test_audio_bytes_input():
    """Test feature extraction from raw audio bytes (as from file uploads or streaming datasets)."""
    sr = 22050
    audio = generate_sine_wave(freq=500.0, duration=1.0, sr=sr)

    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    audio_bytes = buffer.getvalue()

    features = preprocess_audio(audio_bytes, n_mfcc=40)
    assert features.shape == (80,)
    assert not np.isnan(features).any()


def test_resampling_rates():
    """Test that various common sampling rates resample seamlessly to 22050 Hz."""
    for test_sr in [16000, 32000, 44100, 48000]:
        audio = generate_sine_wave(freq=300.0, duration=0.5, sr=test_sr)
        features = preprocess_audio((audio, test_sr), n_mfcc=40)
        assert features.shape == (80,)
        assert not np.isnan(features).any()


def test_silence_and_noise_signals():
    """Verify numeric stability: silence and noise should not produce NaN or Inf."""
    sr = 22050

    # Total silence
    silence = np.zeros(sr, dtype=np.float32)
    feats_silence = preprocess_audio((silence, sr), n_mfcc=40)
    assert feats_silence.shape == (80,)
    assert not np.isnan(feats_silence).any()
    assert not np.isinf(feats_silence).any()

    # Random noise
    np.random.seed(42)
    noise = np.random.uniform(-1.0, 1.0, sr).astype(np.float32)
    feats_noise = preprocess_audio((noise, sr), n_mfcc=40)
    assert feats_noise.shape == (80,)
    assert not np.isnan(feats_noise).any()
    assert not np.isinf(feats_noise).any()


def test_short_audio_padding():
    """Verify that signals shorter than n_fft (2048 samples) are padded and processed safely."""
    sr = 22050
    very_short_audio = np.ones(500, dtype=np.float32) * 0.1
    features = preprocess_audio((very_short_audio, sr), n_mfcc=40)
    assert features.shape == (80,)
    assert not np.isnan(features).any()


def test_unsupported_input_type():
    """Verify TypeError is raised when unsupported object type is passed."""
    with pytest.raises(TypeError):
        preprocess_audio(12345)
