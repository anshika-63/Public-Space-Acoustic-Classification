"""
Audio Preprocessing and Feature Extraction Module
T254 - Public-Space Acoustic Event Classification

Extracts 80 acoustic features (40 MFCC temporal means + 40 MFCC temporal standard deviations)
from raw audio bytes or NumPy audio arrays.
"""

import io
from typing import List, Optional, Tuple, Union
import librosa
import numpy as np
import soundfile as sf


def get_feature_names(n_mfcc: int = 40) -> List[str]:
    """
    Generate standard column names for the extracted MFCC features.

    Parameters:
        n_mfcc (int): Number of MFCC coefficients. Default 40.

    Returns:
        List[str]: List of 2 * n_mfcc column names (means followed by standard deviations).
    """
    mean_cols = [f"mfcc_mean_{i + 1}" for i in range(n_mfcc)]
    std_cols = [f"mfcc_std_{i + 1}" for i in range(n_mfcc)]
    return mean_cols + std_cols


def preprocess_audio(
    audio_input: Union[bytes, bytearray, io.BytesIO, Tuple[np.ndarray, int], np.ndarray],
    sr: Optional[int] = None,
    n_mfcc: int = 40,
    target_sr: int = 22050,
) -> np.ndarray:
    """
    Extract 80 robust MFCC features (40 means + 40 standard deviations) from audio.

    Supports:
        - Raw audio bytes / bytearray (e.g. from dataset streaming or file upload)
        - io.BytesIO stream
        - Tuple of (audio_array, sample_rate)
        - NumPy audio array with sr specified

    Parameters:
        audio_input: Audio source in any of the supported formats.
        sr (int, optional): Source sampling rate if audio_input is a NumPy array.
        n_mfcc (int): Number of MFCC coefficients to compute (default 40 -> 80 features).
        target_sr (int): Target sampling rate to standardize to (default 22,050 Hz).

    Returns:
        np.ndarray: 1D array of length (2 * n_mfcc) with dtype float32.
    """
    # 1. Parse input format into audio array and sampling rate
    if isinstance(audio_input, (bytes, bytearray)):
        audio, orig_sr = sf.read(io.BytesIO(audio_input))
    elif isinstance(audio_input, io.BytesIO):
        audio, orig_sr = sf.read(audio_input)
    elif isinstance(audio_input, tuple) and len(audio_input) == 2:
        audio, orig_sr = audio_input
    elif isinstance(audio_input, np.ndarray):
        audio = audio_input
        orig_sr = sr if sr is not None else target_sr
    else:
        raise TypeError(
            f"Unsupported audio input type: {type(audio_input)}. "
            "Expected bytes, BytesIO, tuple (audio, sr), or np.ndarray."
        )

    audio = np.asarray(audio, dtype=np.float32)

    # 2. Convert multi-channel / stereo audio to mono
    if audio.ndim > 1:
        # Standard soundfile shape is (samples, channels)
        if audio.shape[0] > audio.shape[1]:
            audio = librosa.to_mono(audio.T)
        else:
            audio = librosa.to_mono(audio)

    # 3. Handle edge cases (empty or extremely short audio clips)
    if len(audio) == 0:
        audio = np.zeros(target_sr, dtype=np.float32)
    elif len(audio) < 2048:
        audio = np.pad(audio, (0, 2048 - len(audio)), mode="constant")

    # 4. Standardize sampling rate to target_sr (22,050 Hz)
    if orig_sr != target_sr:
        audio = librosa.resample(
            y=audio,
            orig_sr=orig_sr,
            target_sr=target_sr,
        )

    # 5. Extract Mel-Frequency Cepstral Coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=target_sr,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512,
    )

    # 6. Compute temporal statistics across time frames: mean and standard deviation
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 7. Concatenate to form a 2 * n_mfcc feature vector (default: 40 + 40 = 80)
    features = np.concatenate([mfcc_mean, mfcc_std])

    # 8. Clean any NaN / Inf values for robust downstream ML model compatibility
    cleaned_features = np.nan_to_num(
        features, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    return cleaned_features


if __name__ == "__main__":
    print("Testing src.preprocessing module...")
    feature_names = get_feature_names(n_mfcc=40)
    print(f"Generated {len(feature_names)} feature names.")
    print("Sample feature names:", feature_names[:3], "...", feature_names[-3:])

    # Synthetic audio test
    sr_test = 44100
    t = np.linspace(0, 1.0, sr_test, endpoint=False)
    synthetic_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    feats = preprocess_audio((synthetic_audio, sr_test))
    print(f"Extraction test successful! Output shape: {feats.shape}, dtype: {feats.dtype}")
    assert feats.shape == (80,), f"Expected shape (80,), got {feats.shape}"
    assert not np.isnan(feats).any(), "NaN found in features"
    assert not np.isinf(feats).any(), "Inf found in features"
    print("Self-test completed with 100% pass.")
