"""
Train ML classifier on optimized subset of dataset for faster training
"""

import pandas as pd
from ml_metadiscourse_classifier import MetadiscourseClassifier

def main():
    print("Loading synthetic metadiscourse dataset...")
    df = pd.read_csv('synthetic_metadiscourse_dataset.csv')
    
    print(f"Full dataset shape: {df.shape}")
    
    # Use a balanced sample of 10K for efficient training
    # Sample equal numbers of positive and negative examples
    positive_samples = df[df['is_metadiscourse'] == True].sample(n=2500, random_state=42)
    negative_samples = df[df['is_metadiscourse'] == False].sample(n=2500, random_state=42)
    
    balanced_df = pd.concat([positive_samples, negative_samples]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced training dataset shape: {balanced_df.shape}")
    print(f"Positive examples: {(balanced_df['is_metadiscourse'] == True).sum()}")
    print(f"Negative examples: {(balanced_df['is_metadiscourse'] == False).sum()}")
    
    # Train Random Forest (best performer)
    print("\n" + "="*60)
    print("Training Random Forest on balanced 5K dataset")
    print("="*60)
    
    classifier = MetadiscourseClassifier(model_type='random_forest')
    results = classifier.train(balanced_df, test_size=0.2)
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"CV Mean: {results['cv_mean']:.4f}")
    
    # Save the model
    classifier.save_model('metadiscourse_model_balanced_5k.joblib')
    
    # Show feature importance
    print(f"\nTop 15 Most Important Features:")
    feature_importance = classifier.get_feature_importance(top_n=15)
    print(feature_importance)
    
    # Test on a few examples
    print(f"\n" + "="*40)
    print("Testing on sample texts")
    print("="*40)
    
    test_texts = [
        "This study aims to demonstrate the effectiveness of the proposed method.",
        "However, further research is needed to validate these findings.",
        "The results clearly show a significant improvement in performance.",
        "I went to the store to buy groceries.",
        "In conclusion, our findings support the initial hypothesis."
    ]
    
    test_markers = ["demonstrate", "However", "clearly", "went", "conclusion"]
    
    predictions = classifier.predict(test_texts, test_markers)
    
    for pred in predictions:
        print(f"Text: {pred['text'][:50]}...")
        print(f"Marker: '{pred['marker_text']}' -> Metadiscourse: {pred['is_metadiscourse']} "
              f"(confidence: {pred['confidence']:.3f})")
        print()
    
    return classifier, results

if __name__ == "__main__":
    classifier, results = main()