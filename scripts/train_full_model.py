"""
Train ML classifier on full 100K synthetic dataset
"""

import pandas as pd
from ml_metadiscourse_classifier import MetadiscourseClassifier, train_multiple_models

def main():
    print("Loading full synthetic metadiscourse dataset (100K samples)...")
    df = pd.read_csv('synthetic_metadiscourse_dataset.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Positive examples: {(df['is_metadiscourse'] == True).sum()}")
    print(f"Negative examples: {(df['is_metadiscourse'] == False).sum()}")
    
    # Train Random Forest (best performer from previous test)
    print("\n" + "="*60)
    print("Training Random Forest on full dataset")
    print("="*60)
    
    classifier = MetadiscourseClassifier(model_type='random_forest')
    results = classifier.train(df, test_size=0.1)  # Use 10% for testing to have more training data
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"CV Mean: {results['cv_mean']:.4f}")
    
    # Save the final model
    classifier.save_model('metadiscourse_model_final_100k.joblib')
    
    # Show feature importance
    feature_importance = classifier.get_feature_importance(top_n=15)
    print(f"\nTop 15 Most Important Features:")
    print(feature_importance)
    
    return classifier, results

if __name__ == "__main__":
    classifier, results = main()