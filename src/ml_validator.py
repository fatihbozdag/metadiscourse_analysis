#!/usr/bin/env python3
"""
Machine Learning Approach for Metadiscourse Detection
Learns patterns from data rather than using fixed rules
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import spacy
from typing import Dict, List, Tuple, Optional
import pickle
import json

class MetadiscourseFeatureExtractor:
    """Extract features for ML-based metadiscourse detection."""
    
    def __init__(self, nlp_model='en_core_web_sm'):
        self.nlp = spacy.load(nlp_model)
        self.feature_names = []
        
    def extract_features(self, text: str, candidate_span: Tuple[int, int]) -> np.ndarray:
        """Extract comprehensive features for a candidate metadiscourse marker."""
        
        doc = self.nlp(text)
        start_char, end_char = candidate_span
        
        # Find the tokens corresponding to the span
        target_tokens = []
        for token in doc:
            if token.idx >= start_char and token.idx < end_char:
                target_tokens.append(token)
        
        if not target_tokens:
            return np.zeros(50)  # Return zero features if no tokens found
        
        features = []
        
        # 1. Lexical features
        features.extend(self._extract_lexical_features(target_tokens))
        
        # 2. Syntactic features
        features.extend(self._extract_syntactic_features(target_tokens, doc))
        
        # 3. Positional features
        features.extend(self._extract_positional_features(target_tokens, doc))
        
        # 4. Contextual features
        features.extend(self._extract_contextual_features(target_tokens, doc))
        
        # 5. Semantic features
        features.extend(self._extract_semantic_features(target_tokens, doc))
        
        return np.array(features)
    
    def _extract_lexical_features(self, tokens: List) -> List[float]:
        """Extract lexical features from the tokens."""
        
        if not tokens:
            return [0.0] * 10
        
        features = []
        
        # Token length
        features.append(len(tokens))
        features.append(np.mean([len(token.text) for token in tokens]))
        
        # POS tag features
        pos_tags = [token.pos_ for token in tokens]
        features.append(1.0 if 'PRON' in pos_tags else 0.0)
        features.append(1.0 if 'VERB' in pos_tags else 0.0)
        features.append(1.0 if 'ADV' in pos_tags else 0.0)
        features.append(1.0 if 'CCONJ' in pos_tags else 0.0)
        
        # Lemma features
        lemmas = [token.lemma_.lower() for token in tokens]
        features.append(1.0 if any(lemma in ['i', 'we', 'you'] for lemma in lemmas) else 0.0)
        features.append(1.0 if any(lemma in ['but', 'however', 'therefore'] for lemma in lemmas) else 0.0)
        features.append(1.0 if any(lemma in ['may', 'might', 'could'] for lemma in lemmas) else 0.0)
        features.append(1.0 if any(lemma in ['clearly', 'obviously', 'certainly'] for lemma in lemmas) else 0.0)
        
        return features
    
    def _extract_syntactic_features(self, tokens: List, doc) -> List[float]:
        """Extract syntactic features."""
        
        if not tokens:
            return [0.0] * 10
        
        features = []
        
        # Dependency relations
        dep_rels = [token.dep_ for token in tokens]
        features.append(1.0 if 'nsubj' in dep_rels else 0.0)
        features.append(1.0 if 'dobj' in dep_rels else 0.0)
        features.append(1.0 if 'advmod' in dep_rels else 0.0)
        features.append(1.0 if 'cc' in dep_rels else 0.0)
        
        # Head token features
        heads = [token.head.pos_ for token in tokens]
        features.append(1.0 if 'VERB' in heads else 0.0)
        features.append(1.0 if 'NOUN' in heads else 0.0)
        
        # Sentence structure
        sent = tokens[0].sent
        features.append(len(sent) / 100.0)  # Normalized sentence length
        features.append(tokens[0].i / len(sent))  # Relative position in sentence
        
        # Clause features
        features.append(1.0 if any(token.dep_ == 'mark' for token in sent) else 0.0)
        features.append(1.0 if any(token.dep_ == 'ccomp' for token in sent) else 0.0)
        
        return features
    
    def _extract_positional_features(self, tokens: List, doc) -> List[float]:
        """Extract positional features."""
        
        if not tokens:
            return [0.0] * 8
        
        features = []
        
        # Document position
        doc_position = tokens[0].i / len(doc)
        features.append(doc_position)
        
        # Sentence position
        sent = tokens[0].sent
        sent_position = (tokens[0].i - sent.start) / len(sent)
        features.append(sent_position)
        
        # Paragraph position (approximated)
        para_breaks = [i for i, token in enumerate(doc) if token.text == '\n\n']
        if para_breaks:
            current_para = sum(1 for pb in para_breaks if pb < tokens[0].i)
            para_position = current_para / len(para_breaks)
            features.append(para_position)
        else:
            features.append(0.5)  # Middle if no paragraph breaks
        
        # Beginning/end indicators
        features.append(1.0 if sent_position < 0.1 else 0.0)  # Sentence beginning
        features.append(1.0 if sent_position > 0.9 else 0.0)  # Sentence end
        features.append(1.0 if doc_position < 0.1 else 0.0)   # Document beginning
        features.append(1.0 if doc_position > 0.9 else 0.0)   # Document end
        
        # Punctuation context
        features.append(1.0 if tokens[0].i > 0 and doc[tokens[0].i - 1].text in ',.;:' else 0.0)
        
        return features
    
    def _extract_contextual_features(self, tokens: List, doc) -> List[float]:
        """Extract contextual features from surrounding text."""
        
        if not tokens:
            return [0.0] * 12
        
        features = []
        
        # Window size for context
        window = 5
        start_idx = max(0, tokens[0].i - window)
        end_idx = min(len(doc), tokens[-1].i + window + 1)
        
        context_tokens = doc[start_idx:end_idx]
        
        # Context POS patterns
        context_pos = [token.pos_ for token in context_tokens]
        features.append(context_pos.count('VERB') / len(context_pos))
        features.append(context_pos.count('NOUN') / len(context_pos))
        features.append(context_pos.count('ADJ') / len(context_pos))
        features.append(context_pos.count('ADV') / len(context_pos))
        
        # Discourse markers in context
        discourse_markers = ['however', 'therefore', 'furthermore', 'moreover', 'nevertheless']
        context_text = ' '.join([token.text.lower() for token in context_tokens])
        features.append(sum(1 for dm in discourse_markers if dm in context_text) / len(discourse_markers))
        
        # Modal verbs in context
        modal_verbs = ['can', 'could', 'may', 'might', 'must', 'should', 'will', 'would']
        features.append(sum(1 for mv in modal_verbs if mv in context_text) / len(modal_verbs))
        
        # Hedging words in context
        hedging_words = ['perhaps', 'possibly', 'probably', 'likely', 'seem', 'appear']
        features.append(sum(1 for hw in hedging_words if hw in context_text) / len(hedging_words))
        
        # Boosting words in context
        boosting_words = ['clearly', 'obviously', 'certainly', 'definitely', 'undoubtedly']
        features.append(sum(1 for bw in boosting_words if bw in context_text) / len(boosting_words))
        
        # Citation patterns
        citation_patterns = ['according to', 'as noted by', 'research shows', 'studies indicate']
        features.append(sum(1 for cp in citation_patterns if cp in context_text) / len(citation_patterns))
        
        # Question patterns
        features.append(1.0 if '?' in context_text else 0.0)
        
        # Imperative patterns
        imperative_words = ['consider', 'note', 'see', 'look', 'examine']
        features.append(sum(1 for iw in imperative_words if iw in context_text) / len(imperative_words))
        
        # Negation
        features.append(1.0 if any(token.dep_ == 'neg' for token in context_tokens) else 0.0)
        
        return features
    
    def _extract_semantic_features(self, tokens: List, doc) -> List[float]:
        """Extract semantic features using word embeddings."""
        
        if not tokens:
            return [0.0] * 10
        
        features = []
        
        # Average word embedding similarity to metadiscourse prototypes
        metadiscourse_prototypes = {
            'hedging': ['perhaps', 'possibly', 'might', 'could'],
            'boosting': ['clearly', 'obviously', 'certainly', 'definitely'],
            'engagement': ['consider', 'note', 'see', 'you'],
            'transitions': ['however', 'therefore', 'furthermore', 'but']
        }
        
        for category, prototype_words in metadiscourse_prototypes.items():
            similarities = []
            for token in tokens:
                if token.has_vector:
                    for proto_word in prototype_words:
                        proto_token = self.nlp(proto_word)[0]
                        if proto_token.has_vector:
                            similarity = token.similarity(proto_token)
                            similarities.append(similarity)
            
            if similarities:
                features.append(np.mean(similarities))
                features.append(np.max(similarities))
            else:
                features.extend([0.0, 0.0])
        
        # Sentence-level semantic coherence
        sent = tokens[0].sent
        if sent.vector_norm > 0:
            token_similarities = []
            for token in tokens:
                if token.has_vector and token.vector_norm > 0:
                    similarity = token.similarity(sent)
                    token_similarities.append(similarity)
            
            if token_similarities:
                features.append(np.mean(token_similarities))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Pad to ensure consistent feature count
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]  # Ensure exactly 10 features

class MLMetadiscourseClassifier:
    """Machine learning classifier for metadiscourse detection."""
    
    def __init__(self):
        self.feature_extractor = MetadiscourseFeatureExtractor()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_training_data(self, texts: List[str], annotations: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from annotated examples."""
        
        X = []
        y = []
        
        for text, annotation in zip(texts, annotations):
            # Extract features for each annotated span
            for span_info in annotation.get('spans', []):
                start, end = span_info['start'], span_info['end']
                is_metadiscourse = span_info['is_metadiscourse']
                
                features = self.feature_extractor.extract_features(text, (start, end))
                X.append(features)
                y.append(1 if is_metadiscourse else 0)
        
        return np.array(X), np.array(y)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the classifier."""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.classifier, X_train_scaled, y_train, cv=5)
        
        # Feature importance
        feature_importance = self.classifier.feature_importances_
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance.tolist()
        }
    
    def predict_metadiscourse(self, text: str, candidate_spans: List[Tuple[int, int]]) -> List[Dict]:
        """Predict metadiscourse for candidate spans."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = []
        
        for start, end in candidate_spans:
            features = self.feature_extractor.extract_features(text, (start, end))
            features_scaled = self.scaler.transform([features])
            
            # Get prediction and probability
            prediction = self.classifier.predict(features_scaled)[0]
            probability = self.classifier.predict_proba(features_scaled)[0]
            
            predictions.append({
                'span': (start, end),
                'text': text[start:end],
                'is_metadiscourse': bool(prediction),
                'confidence': float(probability[1]),  # Probability of being metadiscourse
                'features': features.tolist()
            })
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']

class AdaptiveThresholdCalculator:
    """Calculate adaptive thresholds based on corpus statistics and ML predictions."""
    
    def __init__(self):
        self.corpus_stats = {}
        self.ml_classifier = None
        
    def calculate_adaptive_thresholds(self, texts: List[str], 
                                    ml_predictions: List[List[Dict]]) -> Dict:
        """Calculate thresholds that adapt to the specific corpus."""
        
        # Collect confidence scores for each category
        category_confidences = defaultdict(list)
        
        for predictions in ml_predictions:
            for pred in predictions:
                if pred['is_metadiscourse']:
                    # Would need to classify into categories here
                    # For now, use a simple heuristic
                    category = self._classify_metadiscourse_type(pred['text'])
                    category_confidences[category].append(pred['confidence'])
        
        # Calculate adaptive thresholds
        adaptive_thresholds = {}
        
        for category, confidences in category_confidences.items():
            if len(confidences) > 10:  # Need sufficient data
                # Use statistical measures
                mean_conf = np.mean(confidences)
                std_conf = np.std(confidences)
                
                # Set threshold at mean - 0.5 * std (conservative)
                threshold = max(0.5, mean_conf - 0.5 * std_conf)
                
                adaptive_thresholds[category] = {
                    'threshold': threshold,
                    'mean_confidence': mean_conf,
                    'std_confidence': std_conf,
                    'sample_size': len(confidences)
                }
        
        return adaptive_thresholds
    
    def _classify_metadiscourse_type(self, text: str) -> str:
        """Simple heuristic to classify metadiscourse type."""
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['i', 'we', 'our', 'my']):
            return 'self_mentions'
        elif any(word in text_lower for word in ['you', 'your', 'consider', 'note']):
            return 'engagement_markers'
        elif any(word in text_lower for word in ['may', 'might', 'could', 'perhaps']):
            return 'hedges'
        elif any(word in text_lower for word in ['clearly', 'obviously', 'certainly']):
            return 'boosters'
        elif any(word in text_lower for word in ['but', 'however', 'therefore', 'furthermore']):
            return 'transitions'
        else:
            return 'other'

# Integration example
def create_ml_enhanced_system():
    """Create an ML-enhanced metadiscourse detection system."""
    
    print("🤖 Creating ML-Enhanced Metadiscourse Detection System")
    print("=" * 60)
    
    # This would integrate with existing system
    # 1. Use ML classifier for initial detection
    # 2. Apply adaptive thresholds based on corpus statistics
    # 3. Continuously learn from new data
    
    recommendations = [
        "Train classifier on manually annotated metadiscourse examples",
        "Use feature-based approach rather than fixed regex patterns", 
        "Calculate adaptive thresholds from corpus statistics",
        "Implement active learning to continuously improve",
        "Validate with cross-corpus evaluation"
    ]
    
    return recommendations 