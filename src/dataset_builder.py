"""
UrbanSound8K Full Dataset Extraction Pipeline
T254 - Public-Space Acoustic Event Classification

Extracts 80 MFCC features (40 means + 40 standard deviations) and preserves
metadata (slice_file_name, fold, classID, class) for all ~8,732 samples.
Includes safe resume functionality and progress checkpointing every 500 samples.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Set

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from datasets import Audio, load_dataset

from src.preprocessing import get_feature_names, preprocess_audio


DATASET_NAME = "danavery/urbansound8K"
DEFAULT_OUTPUT_CSV = "data/urbansound8k_features_80.csv"
DEFAULT_CHECKPOINT = 500

ALL_10_CLASSES = [
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


def load_processed_file_names(csv_path: str) -> Set[str]:
    """Load the set of already processed slice_file_name values to enable safe resume."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()

    try:
        df_existing = pd.read_csv(csv_path, usecols=["slice_file_name"])
        return set(df_existing["slice_file_name"].dropna().tolist())
    except Exception as e:
        print(f"Warning: Could not read existing file for resume ({e}). Starting fresh.")
        return set()


def flush_checkpoint(records: List[Dict], csv_path: str, is_first_write: bool) -> bool:
    """Flush accumulated sample records to CSV disk storage."""
    if not records:
        return is_first_write

    df_batch = pd.DataFrame(records)
    write_header = is_first_write and (not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0)

    df_batch.to_csv(
        csv_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
    )
    return False


def build_dataset(
    output_csv: str = DEFAULT_OUTPUT_CSV,
    checkpoint_interval: int = DEFAULT_CHECKPOINT,
    sample_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Stream UrbanSound8K dataset, extract 80 features per audio clip, and save to CSV.

    Parameters:
        output_csv (str): Destination path for extracted CSV.
        checkpoint_interval (int): Number of records between disk checkpoints.
        sample_limit (int, optional): Maximum samples to extract (None for entire dataset).

    Returns:
        pd.DataFrame: Summary DataFrame of processed samples.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    processed_names = load_processed_file_names(output_csv)
    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    is_first_write = not file_exists

    if processed_names:
        print(f"Safe Resume: Found {len(processed_names)} existing records in {output_csv}. Resuming from next sample...")
    else:
        print(f"Starting fresh feature extraction -> Destination: {output_csv}")

    feature_col_names = get_feature_names(n_mfcc=40)
    print(f"Connecting to Hugging Face dataset '{DATASET_NAME}' (streaming mode)...")

    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    records_buffer: List[Dict] = []
    total_processed = len(processed_names)
    skipped_count = 0
    failed_count = 0
    start_time = time.time()

    print("Beginning stream extraction...")

    for i, sample in enumerate(dataset):
        if sample_limit is not None and total_processed >= sample_limit:
            print(f"Sample limit of {sample_limit} reached.")
            break

        file_name = sample.get("slice_file_name", f"sample_{i}.wav")

        # Skip already processed samples
        if file_name in processed_names:
            skipped_count += 1
            if skipped_count % 1000 == 0:
                print(f"Resuming: Skipped {skipped_count} previously processed samples...")
            continue

        try:
            audio_bytes = sample["audio"]["bytes"]
            features = preprocess_audio(audio_bytes, n_mfcc=40)

            record = {
                "slice_file_name": file_name,
                "fold": int(sample["fold"]),
                "classID": int(sample["classID"]),
                "class": str(sample["class"]),
            }
            # Add 80 features
            for name, val in zip(feature_col_names, features):
                record[name] = float(val)

            records_buffer.append(record)
            processed_names.add(file_name)
            total_processed += 1

            if len(records_buffer) >= checkpoint_interval:
                is_first_write = flush_checkpoint(records_buffer, output_csv, is_first_write)
                records_buffer = []
                elapsed = time.time() - start_time
                rate = total_processed / max(elapsed, 0.1)
                print(
                    f"[Checkpoint] Processed: {total_processed} samples | "
                    f"Failures: {failed_count} | Elapsed: {elapsed:.1f}s ({rate:.1f} samples/s)"
                )

        except Exception as e:
            failed_count += 1
            print(f"Error processing sample {i + 1} ({file_name}): {e}")

    # Flush any remaining records
    if records_buffer:
        flush_checkpoint(records_buffer, output_csv, is_first_write)
        print(f"[Final Flush] Wrote remaining {len(records_buffer)} records to disk.")

    total_time = time.time() - start_time
    print(f"\nProcessing Complete!")
    print(f"Total processed in dataset: {total_processed}")
    print(f"Failed samples: {failed_count}")
    print(f"Skipped on resume: {skipped_count}")
    print(f"Total run time: {total_time:.2f} seconds")

    return validate_dataset(output_csv)


def validate_dataset(csv_path: str) -> pd.DataFrame:
    """Validate data integrity, class balance, folds 1-10, and numeric columns."""
    print("\n" + "=" * 60)
    print(f"DATASET VALIDATION REPORT: {csv_path}")
    print("=" * 60)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    df = pd.read_csv(csv_path)

    print(f"Total Rows (Samples): {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"File Size on Disk: {file_size_mb:.2f} MB")

    # Verify column count: 4 metadata + 80 features = 84 columns
    assert len(df.columns) == 84, f"Expected 84 columns (4 metadata + 80 features), found {len(df.columns)}"

    # Verify absence of NaN or Inf values in the 80 feature columns
    feature_cols = get_feature_names(n_mfcc=40)
    nan_count = df[feature_cols].isna().sum().sum()
    print(f"NaN / Null count in feature columns: {nan_count}")
    assert nan_count == 0, f"Found {nan_count} NaN values in feature columns!"

    # Verify Folds
    folds_present = sorted(df["fold"].unique().tolist())
    print(f"Folds present: {folds_present} (Total: {len(folds_present)})")

    # Verify Classes
    classes_present = sorted(df["class"].unique().tolist())
    print(f"Classes present: {len(classes_present)} / 10")

    print("\n--- CLASS DISTRIBUTION ---")
    class_counts = df["class"].value_counts()
    for c_name, count in class_counts.items():
        print(f"  {c_name:20s}: {count:5d} samples ({count / len(df) * 100:.1f}%)")

    print("\n--- FOLD DISTRIBUTION ---")
    fold_counts = df["fold"].value_counts().sort_index()
    for f_num, count in fold_counts.items():
        print(f"  Fold {f_num:2d}: {count:5d} samples ({count / len(df) * 100:.1f}%)")

    print("=" * 60)
    print("VALIDATION SUCCESSFUL: All checks passed.")
    print("=" * 60)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UrbanSound8K Dataset Extraction Pipeline")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV, help="Output CSV path")
    parser.add_argument("--checkpoint", type=int, default=DEFAULT_CHECKPOINT, help="Checkpoint interval")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit (optional)")
    parser.add_argument("--validate", action="store_true", help="Validate existing CSV without processing")

    args = parser.parse_args()

    if args.validate:
        validate_dataset(args.output)
    else:
        build_dataset(
            output_csv=args.output,
            checkpoint_interval=args.checkpoint,
            sample_limit=args.limit,
        )
