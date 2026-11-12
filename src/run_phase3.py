"""
Phase 3 Execution Pipeline: 10-Fold CV, Comparison, Final Fitting & Artifact Saving
T254 - Public-Space Acoustic Event Classification
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.model_evaluation import (
    compare_models,
    evaluate_oof_predictions,
    save_phase3_artifacts,
)
from src.model_training import (
    DATASET_PATH,
    OUTPUT_DIR,
    fit_final_deployable_model,
    load_and_validate_dataset,
    run_10_fold_cv,
)


def run_phase3_pipeline():
    start_total_time = time.time()
    print("=" * 70)
    print("PHASE 3: MACHINE LEARNING MODEL TRAINING & 10-FOLD EVALUATION")
    print("=" * 70)

    # 1. Load and validate full dataset
    print(f"Loading and validating dataset from: {DATASET_PATH}")
    df, meta_cols, feat_cols = load_and_validate_dataset(DATASET_PATH)
    print(f"✓ Dataset verified: {len(df)} samples, {len(feat_cols)} features, 10 folds, 10 classes.")

    # Check if models/phase3/ already exists and warn/create
    if os.path.exists(OUTPUT_DIR) and len(os.listdir(OUTPUT_DIR)) > 0:
        print(f"Notice: Destination directory '{OUTPUT_DIR}' contains existing files. Artifacts will be updated safely.")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 2. Run 10-fold CV for Random Forest
    rf_start = time.time()
    fold_metrics_rf, oof_rf = run_10_fold_cv(df, feat_cols, model_name="random_forest")
    rf_total_cv_time = time.time() - rf_start
    print(f"✓ Random Forest 10-fold CV completed in {rf_total_cv_time:.2f}s")

    # 3. Run 10-fold CV for SVM
    svm_start = time.time()
    fold_metrics_svm, oof_svm = run_10_fold_cv(df, feat_cols, model_name="svm")
    svm_total_cv_time = time.time() - svm_start
    print(f"✓ SVM 10-fold CV completed in {svm_total_cv_time:.2f}s")

    # 4. Evaluate OOF predictions
    print("\nCalculating full out-of-fold evaluation metrics...")
    overall_rf, class_rep_rf, cm_rf = evaluate_oof_predictions(oof_rf, model_name="Random Forest")
    overall_svm, class_rep_svm, cm_svm = evaluate_oof_predictions(oof_svm, model_name="SVM")

    # 5. Statistical and tabular model comparison
    comparison_df, test_summary = compare_models(
        fold_metrics_rf, fold_metrics_svm, overall_rf, overall_svm
    )

    # 6. Fit final deployable models on the complete dataset (8,732 samples)
    rf_final, rf_meta = fit_final_deployable_model(df, feat_cols, model_name="random_forest")
    svm_final, svm_meta = fit_final_deployable_model(df, feat_cols, model_name="svm")

    # Combine all fold metrics for tabular output
    df_fold_rf = pd.DataFrame(fold_metrics_rf)
    df_fold_svm = pd.DataFrame(fold_metrics_svm)
    combined_fold_metrics = pd.concat([df_fold_rf, df_fold_svm], ignore_index=True)

    # Package full evaluation dictionary
    overall_metrics_dict = {
        "execution_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_dataset_samples": int(len(df)),
        "n_features": int(len(feat_cols)),
        "random_forest": overall_rf,
        "svm": overall_svm,
        "statistical_comparison": test_summary,
        "fold_metrics_random_forest": fold_metrics_rf,
        "fold_metrics_svm": fold_metrics_svm,
    }

    # 7. Save all artifacts
    print(f"\nSaving model artifacts, predictions, and reports to: {OUTPUT_DIR}/")
    saved_files = save_phase3_artifacts(
        rf_final_model=rf_final,
        rf_metadata=rf_meta,
        svm_final_pipeline=svm_final,
        svm_metadata=svm_meta,
        oof_rf=oof_rf,
        oof_svm=oof_svm,
        fold_metrics_df=combined_fold_metrics,
        class_report_rf=class_rep_rf,
        class_report_svm=class_rep_svm,
        cm_rf=cm_rf,
        cm_svm=cm_svm,
        comparison_df=comparison_df,
        overall_metrics_dict=overall_metrics_dict,
        output_dir=OUTPUT_DIR,
    )

    total_time = time.time() - start_total_time
    print(f"\n✓ Phase 3 pipeline completed successfully in {total_time:.2f} seconds!")
    print("\nSaved Artifacts:")
    for k, v in saved_files.items():
        size_kb = os.path.getsize(v) / 1024
        print(f"  - {k:30s}: {v} ({size_kb:.1f} KB)")

    print("\n" + "=" * 70)
    print("PHASE 3 SUMMARY TABLE")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print("\nStatistical Winner:", test_summary["best_model"])
    print("=" * 70)

    return overall_metrics_dict, comparison_df, saved_files


if __name__ == "__main__":
    run_phase3_pipeline()
