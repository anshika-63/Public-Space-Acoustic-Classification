"""
Model Training and 10-Fold Cross-Validation Module
T254 - Public-Space Acoustic Event Classification

Implements leakage-free 10-fold cross-validation using official UrbanSound8K folds
for Random Forest and Support Vector Machine (SVM) classifiers.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import get_feature_names

DATASET_PATH = "data/urbansound8k_features_80.csv"
OUTPUT_DIR = "models/phase3"

METADATA_COLS = ["slice_file_name", "fold", "classID", "class"]
EXPECTED_CLASSES = [
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


def load_and_validate_dataset(csv_path: str = DATASET_PATH) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load extracted UrbanSound8K CSV and perform strict schema and numeric validation.

    Returns:
        Tuple[pd.DataFrame, List[str], List[str]]: (dataframe, metadata_columns, feature_columns)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Shape validation: 8,732 rows x 84 columns
    if df.shape != (8732, 84):
        raise ValueError(f"Expected shape (8732, 84), got {df.shape}")

    # 2. Metadata columns verification
    actual_meta = list(df.columns[:4])
    if actual_meta != METADATA_COLS:
        raise ValueError(f"Expected metadata columns {METADATA_COLS}, got {actual_meta}")

    # 3. Feature columns verification (exactly 80 MFCC features)
    feature_cols = list(df.columns[4:])
    expected_feature_cols = get_feature_names(n_mfcc=40)
    if feature_cols != expected_feature_cols:
        raise ValueError("Feature columns do not match expected get_feature_names(40)")

    # 4. Folds validation: official folds 1 to 10
    actual_folds = sorted(df["fold"].unique().tolist())
    if actual_folds != list(range(1, 11)):
        raise ValueError(f"Expected folds 1-10, got {actual_folds}")

    # 5. Class validation: exactly 10 classes
    actual_classes = sorted(df["class"].unique().tolist())
    if actual_classes != sorted(EXPECTED_CLASSES):
        raise ValueError(f"Classes mismatch: expected {EXPECTED_CLASSES}, got {actual_classes}")

    # 6. Numeric & finite checks on all feature columns
    feat_matrix = df[feature_cols].to_numpy()
    if np.isnan(feat_matrix).any():
        raise ValueError("NaN values detected in feature columns")
    if np.isinf(feat_matrix).any():
        raise ValueError("Inf values detected in feature columns")

    return df, METADATA_COLS, feature_cols


def build_random_forest() -> RandomForestClassifier:
    """Instantiate baseline Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def build_svm_pipeline() -> Pipeline:
    """Instantiate baseline SVM pipeline with fold-isolated StandardScaler."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            random_state=42,
        )),
    ])


def run_10_fold_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str = "random_forest",
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Perform official 10-fold cross-validation without data leakage.

    Parameters:
        df: Full dataset DataFrame.
        feature_cols: List of 80 feature column names.
        model_name: 'random_forest' or 'svm'.

    Returns:
        Tuple of (fold_metrics_list, oof_predictions_df).
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    fold_metrics = []
    oof_rows = []

    print(f"\n--- Starting 10-Fold Cross-Validation: {model_name.upper()} ---")

    for test_fold in range(1, 11):
        t0 = time.time()

        train_mask = df["fold"] != test_fold
        test_mask = df["fold"] == test_fold

        X_train = df.loc[train_mask, feature_cols].to_numpy(dtype=np.float32)
        y_train = df.loc[train_mask, "class"].to_numpy()

        X_test = df.loc[test_mask, feature_cols].to_numpy(dtype=np.float32)
        y_test = df.loc[test_mask, "class"].to_numpy()

        # Instantiate fresh model for this fold
        if model_name.lower() in ["rf", "random_forest"]:
            model = build_random_forest()
        elif model_name.lower() in ["svm", "svc"]:
            model = build_svm_pipeline()
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        # Fit exclusively on training folds
        model.fit(X_train, y_train)

        # Predict on unseen test fold
        y_pred = model.predict(X_test)
        fold_time = time.time() - t0

        # Compute fold-level metrics
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        fold_metrics.append({
            "model": model_name,
            "test_fold": test_fold,
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "accuracy": float(acc),
            "macro_precision": float(macro_prec),
            "macro_recall": float(macro_rec),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "train_time_sec": float(fold_time),
        })

        # Record out-of-fold predictions
        test_df = df.loc[test_mask, METADATA_COLS].copy()
        test_df["actual_label"] = y_test
        test_df["predicted_label"] = y_pred
        test_df["model"] = model_name
        oof_rows.append(test_df)

        print(
            f"  Fold {test_fold:2d}/10 | Test Samples: {len(y_test):4d} | "
            f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Time: {fold_time:.1f}s"
        )

    oof_df = pd.concat(oof_rows, ignore_index=True)
    return fold_metrics, oof_df


def fit_final_deployable_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str = "random_forest",
) -> Tuple[object, Dict]:
    """
    Train final deployable model on complete 8,732 dataset using fixed hyperparameters.

    Returns:
        Tuple of (fitted_model, metadata_dict).
    """
    print(f"\nTraining final deployable {model_name.upper()} model on all 8,732 samples...")
    X_full = df[feature_cols].to_numpy(dtype=np.float32)
    y_full = df["class"].to_numpy()

    t0 = time.time()
    if model_name.lower() in ["rf", "random_forest"]:
        model = build_random_forest()
    elif model_name.lower() in ["svm", "svc"]:
        model = build_svm_pipeline()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.fit(X_full, y_full)
    fit_time = time.time() - t0
    print(f"Final {model_name.upper()} model training completed in {fit_time:.2f}s")

    metadata = {
        "model_type": model_name,
        "n_samples": int(len(df)),
        "n_features": int(len(feature_cols)),
        "feature_names": feature_cols,
        "classes": sorted(list(model.classes_)),
        "fit_time_seconds": float(fit_time),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return model, metadata
