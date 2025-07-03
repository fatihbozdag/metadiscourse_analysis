"""
Machine Learning Classifier for Metadiscourse Analysis
Phase 1.2: Feature Engineering & Classification
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from ..features.spacy_extractor import SpacyFeatureExtractor

class MetadiscourseClassifier:
    """
    ML classifier for metadiscourse marker detection using linguistic features
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize classifier
        
        Args:
            model_type: 'random_forest', 'svm', or 'logistic_regression'
        """
        self.model_type = model_type
        self.feature_extractor = None
        self.classifier_pipeline = None
        self.feature_columns = None
        self.label_encoder = None
        self.is_trained = False
        
        # Initialize feature extractor
        print("Initializing Spacy feature extractor with transformer model...")
        self.feature_extractor = SpacyFeatureExtractor()
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and prepare data for training
        
        Args:
            df: DataFrame with columns ['text', 'marker_text', 'marker_category', 'is_metadiscourse']
            
        Returns:
            Tuple of (features, labels)
        """
        print("Extracting linguistic features...")
        
        # Extract features using Spacy
        df_with_features = self.feature_extractor.extract_features_from_dataset(df)
        
        # Select feature columns (those starting with 'feat_')
        feature_cols = [col for col in df_with_features.columns if col.startswith('feat_')]
        self.feature_columns = feature_cols
        
        print(f"Extracted {len(feature_cols)} features: {feature_cols}")
        
        # Prepare feature matrix
        X = df_with_features[feature_cols].copy()
        
        # Handle categorical features
        categorical_features = ['feat_pos_tag', 'feat_dependency_relation', 'feat_head_pos']
        
        for cat_feat in categorical_features:
            if cat_feat in X.columns:
                # Label encode categorical features
                le = LabelEncoder()
                X[cat_feat] = le.fit_transform(X[cat_feat].astype(str))
        
        # Convert boolean features to int
        boolean_features = [col for col in X.columns if X[col].dtype == 'bool']
        for bool_feat in boolean_features:
            X[bool_feat] = X[bool_feat].astype(int)
        
        # Prepare labels
        y = df_with_features['is_metadiscourse'].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label distribution: {np.bincount(y)}")
        
        return X.values, y
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Train the classifier on the dataset
        
        Args:
            df: Training DataFrame
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        print(f"Training {self.model_type} classifier...")
        
        # Prepare features and labels
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Create classifier pipeline
        self.classifier_pipeline = self._create_pipeline()
        
        # Train the model
        print("Fitting model...")
        self.classifier_pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        print("Evaluating model...")
        y_pred = self.classifier_pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Metadiscourse', 'Metadiscourse']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(self.classifier_pipeline, X_train, y_train, cv=5, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.is_trained = True
        
        return {
            'test_accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, target_names=['Non-Metadiscourse', 'Metadiscourse'], output_dict=True)
        }
    
    def _create_pipeline(self) -> Pipeline:
        """Create sklearn pipeline with preprocessing and classifier"""
        
        if self.model_type == 'random_forest':
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            classifier = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True
            )
        elif self.model_type == 'logistic_regression':
            classifier = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create pipeline with scaling and classifier
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
        
        return pipeline
    
    def predict(self, texts: List[str], markers: List[str]) -> List[Dict[str, Any]]:
        """
        Predict metadiscourse for new texts
        
        Args:
            texts: List of text samples
            markers: List of marker texts corresponding to each text
            
        Returns:
            List of prediction dictionaries
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create DataFrame for feature extraction
        df = pd.DataFrame({
            'text': texts,
            'marker_text': markers,
            'is_metadiscourse': [False] * len(texts),  # Dummy values
            'marker_category': ['unknown'] * len(texts)  # Dummy values
        })
        
        # Extract features
        X, _ = self.prepare_features(df)
        
        # Make predictions
        predictions = self.classifier_pipeline.predict(X)
        probabilities = self.classifier_pipeline.predict_proba(X)
        
        results = []
        for i, (text, marker) in enumerate(zip(texts, markers)):
            results.append({
                'text': text,
                'marker_text': marker,
                'is_metadiscourse': bool(predictions[i]),
                'confidence': float(probabilities[i].max()),
                'metadiscourse_probability': float(probabilities[i][1])
            })
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores (for tree-based models)
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if self.model_type not in ['random_forest']:
            print("Feature importance only available for tree-based models")
            return pd.DataFrame()
        
        # Get feature importance from the classifier
        classifier = self.classifier_pipeline.named_steps['classifier']
        importance_scores = classifier.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        model_data = {
            'pipeline': self.classifier_pipeline,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        
        self.classifier_pipeline = model_data['pipeline']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
    
    def hyperparameter_tuning(self, df: pd.DataFrame, cv_folds: int = 3):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            df: Training DataFrame
            cv_folds: Number of cross-validation folds
        """
        print("Performing hyperparameter tuning...")
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Define parameter grids for different models
        if self.model_type == 'random_forest':
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 15, 20, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'svm':
            param_grid = {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'classifier__kernel': ['rbf', 'poly']
            }
        elif self.model_type == 'logistic_regression':
            param_grid = {
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            }
        
        # Create base pipeline
        base_pipeline = self._create_pipeline()
        
        # Grid search
        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update classifier with best parameters
        self.classifier_pipeline = grid_search.best_estimator_
        self.is_trained = True
        
        return grid_search.best_params_, grid_search.best_score_

def train_multiple_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train and compare multiple model types
    
    Args:
        df: Training DataFrame
        
    Returns:
        Dictionary with results for each model type
    """
    model_types = ['logistic_regression', 'random_forest', 'svm']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} classifier")
        print(f"{'='*50}")
        
        try:
            classifier = MetadiscourseClassifier(model_type=model_type)
            model_results = classifier.train(df)
            
            results[model_type] = {
                'classifier': classifier,
                'metrics': model_results
            }
            
            # Save model
            classifier.save_model(f'metadiscourse_model_{model_type}.joblib')
            
        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
            results[model_type] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    # Load dataset
    print("Loading synthetic metadiscourse dataset...")
    df = pd.read_csv('synthetic_metadiscourse_dataset.csv')
    
    # Take a smaller sample for initial testing (remove this for full training)
    print("Using sample of 1000 records for testing...")
    df_sample = df.sample(n=1000, random_state=42)
    
    # Train multiple models for comparison
    results = train_multiple_models(df_sample)
    
    # Print comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*60}")
    
    for model_type, result in results.items():
        if 'error' not in result:
            metrics = result['metrics']
            print(f"{model_type.upper()}: Test Accuracy = {metrics['test_accuracy']:.4f}, "
                  f"CV Mean = {metrics['cv_mean']:.4f}")
        else:
            print(f"{model_type.upper()}: ERROR - {result['error']}")