import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_metadiscourse_model():
    input_csv = "features_and_labels.csv"
    model_output_path = "metadiscourse_model.joblib"

    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} rows from {input_csv}")
    except FileNotFoundError:
        print(f"Error: {input_csv} not found. Please run feature_extractor.py first.")
        return

    # Separate features (X) and labels (y)
    # We will train two models: one for is_metadiscourse and one for marker_category
    X = df.drop(columns=['is_metadiscourse_label', 'marker_category_label'])
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
        X_category = true_metadiscourse_df.drop(columns=['is_metadiscourse_label', 'marker_category_label'])
        y_category = true_metadiscourse_df['marker_category_label']

        print("\n--- Training model for marker_category ---")
        X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_category, y_category, test_size=0.2, random_state=42, stratify=y_category)

        model_marker_category = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model_marker_category.fit(X_train_cat, y_train_cat)

        y_pred_cat = model_marker_category.predict(X_test_cat)
        print("Classification Report (marker_category):")
        print(classification_report(y_test_cat, y_pred_cat))
        print(f"Accuracy (marker_category): {accuracy_score(y_test_cat, y_pred_cat):.4f}")

        # Save both models
        joblib.dump(model_is_metadiscourse, model_output_path.replace(".joblib", "_is_metadiscourse.joblib"))
        joblib.dump(model_marker_category, model_output_path.replace(".joblib", "_marker_category.joblib"))
        print(f"\nModels saved to {model_output_path.replace(".joblib", "_is_metadiscourse.joblib")} and {model_output_path.replace(".joblib", "_marker_category.joblib")}")
    else:
        print("No true metadiscourse instances found to train marker_category model.")

if __name__ == "__main__":
    train_metadiscourse_model()
