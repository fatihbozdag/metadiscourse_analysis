# Standard libs
import os
import sys

# Third-party
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from metalinguistics.utils.preprocessing import clean_features

def train_metadiscourse_model():
    # Prefer combined dataset if available
    input_csv = "combined_training_data.csv" if os.path.exists("combined_training_data.csv") else "features_and_labels.csv"
    model_output_path = "metadiscourse_model.joblib"

    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} rows from {input_csv}")
    except FileNotFoundError:
        print(f"Error: {input_csv} not found. Please run feature_extractor.py first.")
        return

    # Separate features (X) and labels (y)
    # We will train two models: one for is_metadiscourse and one for marker_category
    X = clean_features(df.drop(columns=['is_metadiscourse_label', 'marker_category_label']))

    # Final safeguard: convert anything left to numeric, coerce errors to NaN then fill 0.
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    y_is_metadiscourse = df['is_metadiscourse_label']
    y_marker_category = df['marker_category_label']

    # --- Model 1: Predict is_metadiscourse ---
    print("\n--- Training model for is_metadiscourse ---")
    X_train_is, X_test_is, y_train_is, y_test_is = train_test_split(X, y_is_metadiscourse, test_size=0.2, random_state=42, stratify=y_is_metadiscourse)

    model_is_metadiscourse = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model_is_metadiscourse.fit(X_train_is, y_train_is)

    y_pred_is = model_is_metadiscourse.predict(X_test_is)
    print("Classification Report (is_metadiscourse):")
    print(classification_report(y_test_is, y_pred_is))
    print(f"Accuracy (is_metadiscourse): {accuracy_score(y_test_is, y_pred_is):.4f}")

    # --- Model 2: Predict marker_category (only for true metadiscourse instances) ---
    # Filter data to include only true metadiscourse instances for category prediction
    true_metadiscourse_df = df[df['is_metadiscourse_label'] == True].copy()
    if not true_metadiscourse_df.empty:
        X_category = clean_features(
            true_metadiscourse_df.drop(columns=['is_metadiscourse_label', 'marker_category_label'])
        )
        y_category = true_metadiscourse_df['marker_category_label']

        # Encode string labels → integers
        le = LabelEncoder()
        y_cat_encoded = le.fit_transform(y_category)

        print("\n--- Training model for marker_category ---")
        X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
            X_category, y_cat_encoded, test_size=0.2, random_state=42, stratify=y_cat_encoded
        )

        model_marker_category = RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight='balanced'
        )
        model_marker_category.fit(X_train_cat, y_train_cat)

        y_pred_cat = model_marker_category.predict(X_test_cat)
        print("Classification Report (marker_category):")
        print(classification_report(y_test_cat, y_pred_cat, target_names=le.classes_))
        print(f"Accuracy (marker_category): {accuracy_score(y_test_cat, y_pred_cat):.4f}")

        # Save artefacts
        is_path = model_output_path.replace('.joblib', '_is_metadiscourse.joblib')
        cat_path = model_output_path.replace('.joblib', '_marker_category.joblib')
        enc_path = model_output_path.replace('.joblib', '_marker_label_encoder.joblib')

        joblib.dump(model_is_metadiscourse, is_path)
        joblib.dump(model_marker_category, cat_path)
        joblib.dump(le, enc_path)

        print(f"\nModels saved to {is_path}, {cat_path}\nLabel-encoder saved to {enc_path}")
    else:
        print("No true metadiscourse instances found to train marker_category model.")

if __name__ == "__main__":
    train_metadiscourse_model()
