"""
Unit tests for Machine Learning Model Training & Evaluation
T254 - Public-Space Acoustic Event Classification
"""

import io
import os
import tempfile
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model_evaluation import evaluate_oof_predictions
from src.model_training import (
    EXPECTED_CLASSES,
    METADATA_COLS,
    build_random_forest,
    build_svm_pipeline,
    load_and_validate_dataset,
)
from src.preprocessing import get_feature_names


@pytest.fixture
def synthetic_dataset():
    """Create a small, valid mock dataset for fast testing (100 samples, 10 folds, 10 classes)."""
    np.random.seed(42)
    n_samples = 100
    folds = np.tile(np.arange(1, 11), n_samples // 10)
    classes = np.tile(EXPECTED_CLASSES, n_samples // 10)
    class_ids = np.tile(np.arange(10), n_samples // 10)

    feature_cols = get_feature_names(n_mfcc=40)
    feat_data = np.random.randn(n_samples, 80).astype(np.float32)

    df_meta = pd.DataFrame({
        "slice_file_name": [f"mock_{i}.wav" for i in range(n_samples)],
        "fold": folds,
        "classID": class_ids,
        "class": classes,
    })
    df_feats = pd.DataFrame(feat_data, columns=feature_cols)
    return pd.concat([df_meta, df_feats], axis=1)


def test_schema_validation_error_on_bad_shape(tmp_path):
    """Verify ValueError is raised if dataset shape differs from 8,732 x 84."""
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        load_and_validate_dataset(str(bad_csv))


def test_feature_metadata_separation(synthetic_dataset):
    """Verify that metadata columns are isolated from model input features."""
    meta_cols = list(synthetic_dataset.columns[:4])
    feat_cols = list(synthetic_dataset.columns[4:])

    assert meta_cols == METADATA_COLS
    assert len(feat_cols) == 80
    assert not any(col in feat_cols for col in METADATA_COLS)


def test_10_fold_split_no_overlap(synthetic_dataset):
    """Verify that train and test fold masks have zero overlap for all 10 folds."""
    for test_fold in range(1, 11):
        train_mask = synthetic_dataset["fold"] != test_fold
        test_mask = synthetic_dataset["fold"] == test_fold

        train_indices = set(synthetic_dataset.index[train_mask])
        test_indices = set(synthetic_dataset.index[test_mask])

        # Mutually exclusive and collectively exhaustive
        assert len(train_indices.intersection(test_indices)) == 0
        assert len(train_indices.union(test_indices)) == len(synthetic_dataset)


def test_svm_scaler_leakage_free(synthetic_dataset):
    """Verify that SVM Pipeline scales strictly based on training split statistics."""
    feature_cols = list(synthetic_dataset.columns[4:])
    train_mask = synthetic_dataset["fold"] != 1
    test_mask = synthetic_dataset["fold"] == 1

    X_train = synthetic_dataset.loc[train_mask, feature_cols].to_numpy()
    y_train = synthetic_dataset.loc[train_mask, "class"].to_numpy()
    X_test = synthetic_dataset.loc[test_mask, feature_cols].to_numpy()

    pipeline = build_svm_pipeline()
    pipeline.fit(X_train, y_train)

    scaler = pipeline.named_steps["scaler"]
    # Check that scaler means match training data mean, NOT test data mean
    np.testing.assert_allclose(scaler.mean_, np.mean(X_train, axis=0), rtol=1e-5)
    assert not np.allclose(scaler.mean_, np.mean(X_test, axis=0))

    # Ensure predict operates cleanly
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)


def test_metrics_and_confusion_matrix_shape(synthetic_dataset):
    """Verify metrics calculation and 10x10 confusion matrix output."""
    mock_oof = synthetic_dataset[METADATA_COLS].copy()
    mock_oof["actual_label"] = synthetic_dataset["class"]
    # Introduce mock predictions
    mock_oof["predicted_label"] = synthetic_dataset["class"]
    mock_oof.iloc[0, mock_oof.columns.get_loc("predicted_label")] = EXPECTED_CLASSES[(EXPECTED_CLASSES.index(mock_oof.iloc[0]["actual_label"]) + 1) % 10]

    overall, class_rep, cm = evaluate_oof_predictions(mock_oof, model_name="mock_model")

    assert "accuracy" in overall
    assert "macro_f1" in overall
    assert overall["total_samples"] == len(synthetic_dataset)
    assert cm.shape == (10, 10)
    assert len(class_rep) == 10
    assert list(cm.index) == EXPECTED_CLASSES
    assert list(cm.columns) == EXPECTED_CLASSES


def test_model_serialization_and_reload(tmp_path):
    """Verify model artifact saving and loading with joblib."""
    pipeline = build_svm_pipeline()
    X = np.random.randn(20, 80).astype(np.float32)
    y = np.array([EXPECTED_CLASSES[i % 10] for i in range(20)])
    pipeline.fit(X, y)

    artifact_path = tmp_path / "test_model.joblib"
    joblib.dump({"pipeline": pipeline, "classes": EXPECTED_CLASSES}, artifact_path)

    loaded = joblib.load(artifact_path)
    assert "pipeline" in loaded
    loaded_pipe = loaded["pipeline"]
    preds = loaded_pipe.predict(X)
    assert len(preds) == 20
    assert all(p in EXPECTED_CLASSES for p in preds)
