"""
Train ML classifier on full 100K synthetic dataset
"""

import os
import subprocess
import sys

# Make library importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd

# ------------------------------------------------------------------
# Path constants
# ------------------------------------------------------------------
REAL_FEATURES = os.path.join('data', 'processed', 'features_and_labels.csv')
SYNTH_FEATURES = os.path.join('data', 'processed', 'features_and_labels_synth.csv')
# Path to the 100K-row synthetic dataset provided by the user
SYNTH_RAW = os.path.join('data', 'processed', 'synthetic_metadiscourse_dataset.csv')
COMBINED = 'combined_training_data.csv'


def ensure_synthetic_features():
    """Generate feature file for synthetic dataset if not present."""
    if os.path.exists(SYNTH_FEATURES):
        # Quick sanity-check on row count – we expect ~100k rows
        with open(SYNTH_FEATURES, "r", encoding="utf-8") as fh:
            row_count = sum(1 for _ in fh) - 1  # subtract header

        if row_count >= 100000:
            print(f"Synthetic feature file already exists ✔  (rows: {row_count})")
            return

        print(
            f"Existing synthetic feature file only has {row_count} rows – regenerating from full dataset …"
        )
        os.remove(SYNTH_FEATURES)

    print("Synthetic feature file not found – generating features from 100 K dataset …")
    # Directly write the extracted features to the desired location
    os.makedirs(os.path.dirname(SYNTH_FEATURES), exist_ok=True)
    subprocess.run([
        'python',
        'tools/feature_extractor.py',
        '--input', SYNTH_RAW,
        '--output', SYNTH_FEATURES
    ], check=True)

    if not os.path.exists(SYNTH_FEATURES):
        raise FileNotFoundError(f"Failed to create {SYNTH_FEATURES}")
    else:
        print(f"Synthetic features generated → {SYNTH_FEATURES}")


def build_combined_dataset():
    """Create combined_training_data.csv by merging real and synthetic feature sets."""
    ensure_synthetic_features()

    df_real = pd.read_csv(REAL_FEATURES)
    df_synth = pd.read_csv(SYNTH_FEATURES)

    expected_rows = len(df_real) + len(df_synth)

    if os.path.exists(COMBINED):
        print("Combined dataset already exists – validating size …")
        with open(COMBINED, "r", encoding="utf-8") as fh:
            current_rows = sum(1 for _ in fh) - 1

        if current_rows == expected_rows:
            print("Combined dataset is up-to-date ✔")
            return

        print("Combined dataset is stale (row mismatch) – rebuilding …")

    print("Building combined training dataset …")
    merged = pd.concat([df_real, df_synth], join="outer", ignore_index=True).fillna(0)
    merged.to_csv(COMBINED, index=False)
    print(f"Combined dataset saved → {COMBINED}  (rows: {len(merged)})")


def main():
    # ------------------------------------------------------------------
    # Prepare dataset (merge real + synthetic)
    # ------------------------------------------------------------------
    build_combined_dataset()

    print("Loading combined training data …")
    df = pd.read_csv(COMBINED)

    print(f"Dataset shape: {df.shape}")
    print(f"Positive examples: {(df['is_metadiscourse_label'] == True).sum()}")
    print(f"Negative examples: {(df['is_metadiscourse_label'] == False).sum()}")

    # ------------------------------------------------------------------
    # Train models using the feature-based pipeline (scripts/train_model.py)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Delegating model training to scripts/train_model.py …")
    print("=" * 60)

    subprocess.run(['python', os.path.join('scripts', 'train_model.py')], check=True)

    print("\nFull-model training pipeline completed ✔")

    return None, None

if __name__ == "__main__":
    main()