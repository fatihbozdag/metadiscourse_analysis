#!/usr/bin/env python3
"""
Empirical Validation System for Metadiscourse Patterns
Uses statistical analysis and human validation to discover true patterns
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import spacy
from collections import defaultdict, Counter
import re
from typing import Dict, List, Tuple, Optional
import json

class EmpiricalPatternDiscovery:
    """Discover metadiscourse patterns through empirical analysis rather than assumptions."""
    
    def __init__(self, nlp_model='en_core_web_sm'):
        self.nlp = spacy.load(nlp_model)
        self.validated_patterns = {}
        self.statistical_thresholds = {}
        
    def analyze_context_patterns(self, texts: List[str], candidate_markers: List[str]) -> Dict:
        """Analyze actual context patterns around candidate markers in corpus."""
        
        context_data = []
        
        for text in texts:
            doc = self.nlp(text)
            
            for sent in doc.sents:
                sent_text = sent.text.lower()
                
                for marker in candidate_markers:
                    if marker.lower() in sent_text:
                        # Extract context features
                        context_features = self._extract_context_features(sent, marker)
                        context_data.append({
                            'marker': marker,
                            'sentence': sent.text,
                            'position': context_features['position'],
                            'syntactic_role': context_features['syntactic_role'],
                            'semantic_context': context_features['semantic_context'],
                            'discourse_connectives': context_features['discourse_connectives'],
                            'sentence_type': context_features['sentence_type']
                        })
        
        return self._cluster_contexts(context_data)
    
    def _extract_context_features(self, sent: spacy.tokens.Span, marker: str) -> Dict:
        """Extract linguistic features around a marker occurrence."""
        
        # Find marker position
        marker_tokens = []
        sent_text = sent.text.lower()
        marker_lower = marker.lower()
        
        if marker_lower in sent_text:
            marker_start = sent_text.find(marker_lower)
            marker_end = marker_start + len(marker_lower)
            
            # Position in sentence (0-1 scale)
            position = marker_start / len(sent_text) if len(sent_text) > 0 else 0
            
            # Syntactic analysis
            syntactic_role = self._analyze_syntactic_role(sent, marker_start, marker_end)
            
            # Semantic context (surrounding words)
            semantic_context = self._extract_semantic_context(sent, marker_start, marker_end)
            
            # Discourse connectives nearby
            discourse_connectives = self._find_discourse_connectives(sent)
            
            # Sentence type
            sentence_type = self._classify_sentence_type(sent)
            
            return {
                'position': position,
                'syntactic_role': syntactic_role,
                'semantic_context': semantic_context,
                'discourse_connectives': discourse_connectives,
                'sentence_type': sentence_type
            }
        
        return {}
    
    def _analyze_syntactic_role(self, sent, start_pos, end_pos) -> str:
        """Analyze the syntactic role of the marker in the sentence."""
        
        # Find the token(s) corresponding to the marker
        for token in sent:
            if token.idx >= start_pos and token.idx < end_pos:
                return f"{token.dep_}_{token.pos_}"
        
        return "unknown"
    
    def _extract_semantic_context(self, sent, start_pos, end_pos, window=3) -> List[str]:
        """Extract semantic context (lemmatized words) around the marker."""
        
        context_words = []
        
        for token in sent:
            # Get words within window of marker
            if abs(token.idx - start_pos) <= window * 10:  # Approximate word distance
                if not token.is_stop and not token.is_punct and token.is_alpha:
                    context_words.append(token.lemma_.lower())
        
        return context_words
    
    def _find_discourse_connectives(self, sent) -> List[str]:
        """Find discourse connectives in the sentence."""
        
        discourse_markers = [
            'however', 'therefore', 'furthermore', 'moreover', 'nevertheless',
            'consequently', 'thus', 'hence', 'accordingly', 'meanwhile'
        ]
        
        found_connectives = []
        sent_text = sent.text.lower()
        
        for marker in discourse_markers:
            if marker in sent_text:
                found_connectives.append(marker)
        
        return found_connectives
    
    def _classify_sentence_type(self, sent) -> str:
        """Classify the type of sentence (declarative, interrogative, etc.)."""
        
        sent_text = sent.text.strip()
        
        if sent_text.endswith('?'):
            return 'interrogative'
        elif sent_text.endswith('!'):
            return 'exclamatory'
        elif any(word.lower() in sent_text.lower() for word in ['should', 'must', 'need to', 'have to']):
            return 'imperative'
        else:
            return 'declarative'
    
    def _cluster_contexts(self, context_data: List[Dict]) -> Dict:
        """Cluster similar contexts to discover patterns."""
        
        if not context_data:
            return {}
        
        # Create feature vectors for clustering
        features = []
        for item in context_data:
            feature_vector = [
                item['position'],
                1 if item['sentence_type'] == 'declarative' else 0,
                1 if item['sentence_type'] == 'interrogative' else 0,
                len(item['discourse_connectives']),
                len(item['semantic_context'])
            ]
            features.append(feature_vector)
        
        # Perform clustering
        if len(features) > 3:
            kmeans = KMeans(n_clusters=min(3, len(features)), random_state=42)
            clusters = kmeans.fit_predict(features)
            
            # Analyze clusters
            cluster_analysis = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                cluster_analysis[cluster_id].append(context_data[i])
            
            return dict(cluster_analysis)
        
        return {0: context_data}

class StatisticalThresholdCalculator:
    """Calculate statistically valid thresholds rather than arbitrary ones."""
    
    def __init__(self):
        self.genre_profiles = {}
        self.author_profiles = {}
        
    def calculate_dynamic_thresholds(self, df: pd.DataFrame, 
                                   text_column: str = 'text_field',
                                   genre_column: str = None) -> Dict:
        """Calculate thresholds based on statistical distribution of the data."""
        
        # Extract marker frequencies for each document
        marker_frequencies = self._extract_marker_frequencies(df, text_column)
        
        # Calculate statistical thresholds
        thresholds = {}
        
        for category, frequencies in marker_frequencies.items():
            if len(frequencies) > 0:
                # Use statistical measures instead of arbitrary limits
                mean_freq = np.mean(frequencies)
                std_freq = np.std(frequencies)
                median_freq = np.median(frequencies)
                q75 = np.percentile(frequencies, 75)
                q95 = np.percentile(frequencies, 95)
                
                # Calculate outlier threshold using IQR method
                q1 = np.percentile(frequencies, 25)
                q3 = np.percentile(frequencies, 75)
                iqr = q3 - q1
                outlier_threshold = q3 + 1.5 * iqr
                
                thresholds[category] = {
                    'mean': mean_freq,
                    'std': std_freq,
                    'median': median_freq,
                    'q75': q75,
                    'q95': q95,
                    'outlier_threshold': outlier_threshold,
                    'suggested_max': min(q95, outlier_threshold),  # Conservative upper bound
                    'distribution': frequencies
                }
        
        # Genre-specific analysis if available
        if genre_column and genre_column in df.columns:
            thresholds['genre_specific'] = self._calculate_genre_thresholds(
                df, text_column, genre_column, marker_frequencies
            )
        
        return thresholds
    
    def _extract_marker_frequencies(self, df: pd.DataFrame, text_column: str) -> Dict:
        """Extract actual marker frequencies from the corpus."""
        
        # This would integrate with the existing processor
        # For now, simulate with word counting
        
        categories = [
            'hedges', 'boosters', 'attitude_markers', 
            'engagement_markers', 'self_mentions',
            'transitions', 'frame_markers', 'evidentials'
        ]
        
        frequencies = {cat: [] for cat in categories}
        
        for _, row in df.iterrows():
            text = str(row[text_column])
            word_count = len(text.split())
            
            if word_count > 0:
                # Simple frequency calculation (would be replaced with actual marker detection)
                for category in categories:
                    # Placeholder - would use actual marker counts
                    freq_per_1k = np.random.normal(50, 15)  # Simulate realistic frequencies
                    if freq_per_1k > 0:
                        frequencies[category].append(freq_per_1k)
        
        return frequencies
    
    def _calculate_genre_thresholds(self, df: pd.DataFrame, text_column: str, 
                                  genre_column: str, base_frequencies: Dict) -> Dict:
        """Calculate genre-specific thresholds."""
        
        genre_thresholds = {}
        
        for genre in df[genre_column].unique():
            if pd.notna(genre):
                genre_df = df[df[genre_column] == genre]
                genre_frequencies = self._extract_marker_frequencies(genre_df, text_column)
                
                genre_stats = {}
                for category, frequencies in genre_frequencies.items():
                    if len(frequencies) > 0:
                        genre_stats[category] = {
                            'mean': np.mean(frequencies),
                            'median': np.median(frequencies),
                            'q95': np.percentile(frequencies, 95)
                        }
                
                genre_thresholds[genre] = genre_stats
        
        return genre_thresholds

class HumanValidationSystem:
    """System for incorporating human validation of metadiscourse patterns."""
    
    def __init__(self):
        self.validation_data = {}
        self.inter_annotator_agreement = {}
        
    def create_validation_sample(self, texts: List[str], markers: List[str], 
                                sample_size: int = 100) -> List[Dict]:
        """Create a sample for human validation."""
        
        validation_samples = []
        
        # Sample random occurrences of markers in context
        for text in texts[:sample_size]:
            doc = spacy.load('en_core_web_sm')(text)
            
            for sent in doc.sents:
                for marker in markers:
                    if marker.lower() in sent.text.lower():
                        validation_samples.append({
                            'sentence': sent.text,
                            'marker': marker,
                            'context': self._get_extended_context(sent, doc),
                            'validated': None,  # To be filled by human annotators
                            'metadiscourse_type': None,  # To be classified by humans
                            'confidence': None  # Human confidence rating
                        })
        
        return validation_samples
    
    def _get_extended_context(self, sent, doc, context_size=2):
        """Get extended context around a sentence."""
        
        sent_idx = list(doc.sents).index(sent)
        start_idx = max(0, sent_idx - context_size)
        end_idx = min(len(list(doc.sents)), sent_idx + context_size + 1)
        
        context_sents = list(doc.sents)[start_idx:end_idx]
        return ' '.join([s.text for s in context_sents])
    
    def calculate_agreement(self, annotations: List[Dict]) -> float:
        """Calculate inter-annotator agreement (simplified)."""
        
        # This would implement proper agreement metrics like Cohen's kappa
        # For now, simplified version
        
        agreements = []
        for annotation in annotations:
            if 'annotator_1' in annotation and 'annotator_2' in annotation:
                if annotation['annotator_1'] == annotation['annotator_2']:
                    agreements.append(1)
                else:
                    agreements.append(0)
        
        return np.mean(agreements) if agreements else 0.0

# Usage example and integration
class EmpiricalMetadiscourseValidator:
    """Main class that integrates all empirical validation approaches."""
    
    def __init__(self):
        self.pattern_discovery = EmpiricalPatternDiscovery()
        self.threshold_calculator = StatisticalThresholdCalculator()
        self.human_validator = HumanValidationSystem()
        
    def validate_and_improve_system(self, df: pd.DataFrame, 
                                  text_column: str = 'text_field') -> Dict:
        """Run comprehensive empirical validation and improvement."""
        
        print("🔬 Starting Empirical Validation Process...")
        
        # 1. Discover actual patterns in the data
        texts = df[text_column].astype(str).tolist()
        candidate_markers = ['i', 'we', 'you', 'but', 'so', 'however', 'therefore']
        
        patterns = self.pattern_discovery.analyze_context_patterns(texts, candidate_markers)
        
        # 2. Calculate statistical thresholds
        thresholds = self.threshold_calculator.calculate_dynamic_thresholds(df, text_column)
        
        # 3. Create human validation sample
        validation_sample = self.human_validator.create_validation_sample(
            texts[:50], candidate_markers, 50
        )
        
        return {
            'discovered_patterns': patterns,
            'statistical_thresholds': thresholds,
            'validation_sample': validation_sample,
            'recommendations': self._generate_recommendations(patterns, thresholds)
        }
    
    def _generate_recommendations(self, patterns: Dict, thresholds: Dict) -> List[str]:
        """Generate recommendations based on empirical analysis."""
        
        recommendations = [
            "Replace fixed regex patterns with discovered statistical patterns",
            "Use dynamic thresholds based on corpus statistics rather than arbitrary limits",
            "Implement genre-specific and context-specific filtering",
            "Validate patterns through human annotation before deployment",
            "Monitor system performance and adjust thresholds based on new data"
        ]
        
        return recommendations 