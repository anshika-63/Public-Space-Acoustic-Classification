"""
Model Evaluation and Comparison Module
T254 - Public-Space Acoustic Event Classification

Computes comprehensive metrics, per-class evaluation, confusion matrices,
statistical significance testing, and artifact serialization.
"""

import json
import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_training import EXPECTED_CLASSES, OUTPUT_DIR


def evaluate_oof_predictions(
    oof_df: pd.DataFrame,
    model_name: str,
    class_order: List[str] = EXPECTED_CLASSES,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Calculate full evaluation metrics from out-of-fold predictions.

    Returns:
        Tuple of (overall_metrics_dict, class_report_df, confusion_matrix_df).
    """
    y_true = oof_df["actual_label"].to_numpy()
    y_pred = oof_df["predicted_label"].to_numpy()

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    overall_metrics = {
        "model": model_name,
        "total_samples": int(len(y_true)),
        "accuracy": float(acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }

    # Per-class classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=class_order,
        output_dict=True,
        zero_division=0,
    )
    class_rows = []
    for cls_name in class_order:
        class_rows.append({
            "class": cls_name,
            "precision": report_dict[cls_name]["precision"],
            "recall": report_dict[cls_name]["recall"],
            "f1-score": report_dict[cls_name]["f1-score"],
            "support": int(report_dict[cls_name]["support"]),
        })
    class_report_df = pd.DataFrame(class_rows)

    # 10x10 Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_order)
    cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)

    return overall_metrics, class_report_df, cm_df


def compare_models(
    fold_metrics_rf: List[Dict],
    fold_metrics_svm: List[Dict],
    overall_rf: Dict,
    overall_svm: Dict,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform statistical and tabular comparison between Random Forest and SVM across the 10 folds.
    """
    df_rf = pd.DataFrame(fold_metrics_rf)
    df_svm = pd.DataFrame(fold_metrics_svm)

    rf_f1 = df_rf["macro_f1"].to_numpy()
    svm_f1 = df_svm["macro_f1"].to_numpy()
    rf_acc = df_rf["accuracy"].to_numpy()
    svm_acc = df_svm["accuracy"].to_numpy()

    # Paired t-test across 10 folds
    t_stat_f1, p_val_f1 = stats.ttest_rel(rf_f1, svm_f1)
    t_stat_acc, p_val_acc = stats.ttest_rel(rf_acc, svm_acc)

    # Wilcoxon signed-rank test
    try:
        w_stat_f1, w_pval_f1 = stats.wilcoxon(rf_f1, svm_f1)
    except Exception:
        w_stat_f1, w_pval_f1 = np.nan, np.nan

    comparison_rows = [
        {
            "metric": "Overall Accuracy",
            "Random_Forest": overall_rf["accuracy"],
            "SVM": overall_svm["accuracy"],
            "Difference (RF - SVM)": overall_rf["accuracy"] - overall_svm["accuracy"],
        },
        {
            "metric": "Macro F1-Score",
            "Random_Forest": overall_rf["macro_f1"],
            "SVM": overall_svm["macro_f1"],
            "Difference (RF - SVM)": overall_rf["macro_f1"] - overall_svm["macro_f1"],
        },
        {
            "metric": "Weighted F1-Score",
            "Random_Forest": overall_rf["weighted_f1"],
            "SVM": overall_svm["weighted_f1"],
            "Difference (RF - SVM)": overall_rf["weighted_f1"] - overall_svm["weighted_f1"],
        },
        {
            "metric": "Macro Precision",
            "Random_Forest": overall_rf["macro_precision"],
            "SVM": overall_svm["macro_precision"],
            "Difference (RF - SVM)": overall_rf["macro_precision"] - overall_svm["macro_precision"],
        },
        {
            "metric": "Macro Recall",
            "Random_Forest": overall_rf["macro_recall"],
            "SVM": overall_svm["macro_recall"],
            "Difference (RF - SVM)": overall_rf["macro_recall"] - overall_svm["macro_recall"],
        },
        {
            "metric": "Fold Mean Accuracy ± Std",
            "Random_Forest": f"{rf_acc.mean():.4f} ± {rf_acc.std():.4f}",
            "SVM": f"{svm_acc.mean():.4f} ± {svm_acc.std():.4f}",
            "Difference (RF - SVM)": f"{rf_acc.mean() - svm_acc.mean():.4f}",
        },
        {
            "metric": "Fold Mean Macro F1 ± Std",
            "Random_Forest": f"{rf_f1.mean():.4f} ± {rf_f1.std():.4f}",
            "SVM": f"{svm_f1.mean():.4f} ± {svm_f1.std():.4f}",
            "Difference (RF - SVM)": f"{rf_f1.mean() - svm_f1.mean():.4f}",
        },
    ]
    comparison_df = pd.DataFrame(comparison_rows)

    # Determine best model
    if overall_rf["macro_f1"] > overall_svm["macro_f1"]:
        best_model = "Random Forest"
        winner_key = "random_forest"
    else:
        best_model = "SVM"
        winner_key = "svm"

    test_summary = {
        "best_model": best_model,
        "winner_key": winner_key,
        "paired_ttest_macro_f1": {
            "t_statistic": float(t_stat_f1),
            "p_value": float(p_val_f1),
            "statistically_significant_p05": bool(p_val_f1 < 0.05),
        },
        "paired_ttest_accuracy": {
            "t_statistic": float(t_stat_acc),
            "p_value": float(p_val_acc),
            "statistically_significant_p05": bool(p_val_acc < 0.05),
        },
        "wilcoxon_macro_f1": {
            "statistic": float(w_stat_f1) if not np.isnan(w_stat_f1) else None,
            "p_value": float(w_pval_f1) if not np.isnan(w_pval_f1) else None,
        },
    }

    return comparison_df, test_summary


def save_phase3_artifacts(
    rf_final_model,
    rf_metadata: Dict,
    svm_final_pipeline,
    svm_metadata: Dict,
    oof_rf: pd.DataFrame,
    oof_svm: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    class_report_rf: pd.DataFrame,
    class_report_svm: pd.DataFrame,
    cm_rf: pd.DataFrame,
    cm_svm: pd.DataFrame,
    comparison_df: pd.DataFrame,
    overall_metrics_dict: Dict,
    output_dir: str = OUTPUT_DIR,
) -> Dict[str, str]:
    """
    Save all Phase 3 models, metrics, predictions, and reports to disk safely.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}

    # 1. Models
    rf_path = os.path.join(output_dir, "random_forest_model.joblib")
    joblib.dump({"model": rf_final_model, "metadata": rf_metadata}, rf_path)
    saved_files["random_forest_model"] = rf_path

    svm_path = os.path.join(output_dir, "svm_pipeline.joblib")
    joblib.dump({"pipeline": svm_final_pipeline, "metadata": svm_metadata}, svm_path)
    saved_files["svm_pipeline"] = svm_path

    # Best model deployable link / copy
    best_key = overall_metrics_dict["statistical_comparison"]["winner_key"]
    best_model_obj = rf_final_model if best_key == "random_forest" else svm_final_pipeline
    best_meta = rf_metadata if best_key == "random_forest" else svm_metadata
    best_path = os.path.join(output_dir, "best_acoustic_model.joblib")
    joblib.dump({"model": best_model_obj, "metadata": best_meta}, best_path)
    saved_files["best_model"] = best_path

    # 2. Predictions
    oof_rf_path = os.path.join(output_dir, "oof_predictions_random_forest.csv")
    oof_rf.to_csv(oof_rf_path, index=False)
    saved_files["oof_predictions_random_forest"] = oof_rf_path

    oof_svm_path = os.path.join(output_dir, "oof_predictions_svm.csv")
    oof_svm.to_csv(oof_svm_path, index=False)
    saved_files["oof_predictions_svm"] = oof_svm_path

    # 3. Reports
    rep_rf_path = os.path.join(output_dir, "classification_report_random_forest.csv")
    class_report_rf.to_csv(rep_rf_path, index=False)
    saved_files["classification_report_random_forest"] = rep_rf_path

    rep_svm_path = os.path.join(output_dir, "classification_report_svm.csv")
    class_report_svm.to_csv(rep_svm_path, index=False)
    saved_files["classification_report_svm"] = rep_svm_path

    # 4. Confusion Matrices
    cm_rf_path = os.path.join(output_dir, "confusion_matrix_random_forest.csv")
    cm_rf.to_csv(cm_rf_path)
    saved_files["confusion_matrix_random_forest"] = cm_rf_path

    cm_svm_path = os.path.join(output_dir, "confusion_matrix_svm.csv")
    cm_svm.to_csv(cm_svm_path)
    saved_files["confusion_matrix_svm"] = cm_svm_path

    # 5. Fold metrics & Comparison
    fold_path = os.path.join(output_dir, "fold_metrics.csv")
    fold_metrics_df.to_csv(fold_path, index=False)
    saved_files["fold_metrics"] = fold_path

    comp_path = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(comp_path, index=False)
    saved_files["model_comparison"] = comp_path

    # 6. Comprehensive JSON metrics
    json_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(overall_metrics_dict, f, indent=2)
    saved_files["evaluation_metrics"] = json_path

    return saved_files
