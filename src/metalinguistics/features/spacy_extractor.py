"""
Advanced Feature Extraction for Metadiscourse Analysis using Spacy
Phase 1: Foundation & Data-Driven Approach
"""

import spacy
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import re

@dataclass
class LinguisticFeatures:
    """Container for linguistic features extracted from text"""
    # Lexical features
    marker_text: str
    marker_length: int
    marker_word_count: int
    is_capitalized: bool
    has_punctuation: bool
    
    # Syntactic features (POS and dependencies)
    pos_tag: str
    dependency_relation: str
    head_pos: str
    syntactic_children: List[str]
    
    # Contextual features
    left_context_pos: List[str]
    right_context_pos: List[str]
    sentence_position: float  # 0.0 = start, 1.0 = end
    distance_to_sentence_start: int
    distance_to_sentence_end: int
    
    # Semantic features
    is_sentence_start: bool
    is_sentence_end: bool
    follows_punctuation: bool
    precedes_punctuation: bool
    
    # Academic context indicators
    in_academic_verb_phrase: bool
    academic_context_score: float
    
    # Word embedding features (to be added with transformers)
    embedding_similarity_to_academic: float = 0.0

class SpacyFeatureExtractor:
    """
    Advanced feature extractor using Spacy for comprehensive linguistic analysis
    """
    
    def __init__(self, model_name: str = "en_core_web_trf", use_mps: bool = True):
        """Initialize Spacy model with MPS acceleration and academic vocabulary"""
        import torch
        
        # Configure MPS acceleration if available and requested
        if use_mps and torch.backends.mps.is_available():
            spacy.prefer_gpu()
            print("Using MPS acceleration for transformer model")
        
        self.nlp = spacy.load(model_name)
        
        # Configure for batch processing efficiency
        self.nlp.max_length = 2000000  # Handle long texts
        
        # Academic context vocabulary
        self.academic_verbs = {
            'argue', 'demonstrate', 'show', 'indicate', 'suggest', 'propose',
            'conclude', 'find', 'observe', 'note', 'examine', 'analyze',
            'discuss', 'explore', 'investigate', 'study', 'research'
        }
        
        self.academic_nouns = {
            'study', 'research', 'analysis', 'investigation', 'examination',
            'discussion', 'paper', 'article', 'work', 'findings', 'results',
            'data', 'evidence', 'approach', 'method', 'framework'
        }
        
        self.discourse_markers = {
            'frame_markers': {'first', 'second', 'finally', 'conclusion', 'summary'},
            'transitions': {'however', 'therefore', 'moreover', 'furthermore'},
            'evidentials': {'according', 'show', 'demonstrate', 'indicate'},
            'code_glosses': {'namely', 'specifically', 'particularly', 'example'},
            'engagement_markers': {'note', 'see', 'consider', 'observe'},
            'self_mentions': {'we', 'our', 'my', 'author'},
            'boosters': {'clearly', 'obviously', 'definitely', 'certainly'},
            'hedges': {'might', 'could', 'perhaps', 'possibly', 'seem'}
        }
    
    def extract_features(self, text: str, marker_text: str, marker_start: int) -> LinguisticFeatures:
        """
        Extract comprehensive linguistic features for a potential metadiscourse marker
        
        Args:
            text: Full sentence/text containing the marker
            marker_text: The specific marker text to analyze
            marker_start: Character position where marker starts in text
            
        Returns:
            LinguisticFeatures object with all extracted features
        """
        doc = self.nlp(text)
        
        # Find the marker span in the processed document
        marker_span = self._find_marker_span(doc, marker_text, marker_start)
        if not marker_span:
            # Fallback: create minimal features if marker not found
            return self._create_fallback_features(marker_text)
        
        # Extract all feature categories
        lexical_features = self._extract_lexical_features(marker_text, marker_span)
        syntactic_features = self._extract_syntactic_features(marker_span, doc)
        contextual_features = self._extract_contextual_features(marker_span, doc)
        semantic_features = self._extract_semantic_features(marker_span, doc)
        academic_features = self._extract_academic_features(marker_span, doc)
        
        return LinguisticFeatures(
            # Lexical
            marker_text=lexical_features['marker_text'],
            marker_length=lexical_features['marker_length'],
            marker_word_count=lexical_features['marker_word_count'],
            is_capitalized=lexical_features['is_capitalized'],
            has_punctuation=lexical_features['has_punctuation'],
            
            # Syntactic
            pos_tag=syntactic_features['pos_tag'],
            dependency_relation=syntactic_features['dependency_relation'],
            head_pos=syntactic_features['head_pos'],
            syntactic_children=syntactic_features['syntactic_children'],
            
            # Contextual
            left_context_pos=contextual_features['left_context_pos'],
            right_context_pos=contextual_features['right_context_pos'],
            sentence_position=contextual_features['sentence_position'],
            distance_to_sentence_start=contextual_features['distance_to_sentence_start'],
            distance_to_sentence_end=contextual_features['distance_to_sentence_end'],
            
            # Semantic
            is_sentence_start=semantic_features['is_sentence_start'],
            is_sentence_end=semantic_features['is_sentence_end'],
            follows_punctuation=semantic_features['follows_punctuation'],
            precedes_punctuation=semantic_features['precedes_punctuation'],
            
            # Academic
            in_academic_verb_phrase=academic_features['in_academic_verb_phrase'],
            academic_context_score=academic_features['academic_context_score']
        )
    
    def _find_marker_span(self, doc, marker_text: str, marker_start: int):
        """Find the spacy span corresponding to the marker text"""
        marker_end = marker_start + len(marker_text)
        
        # Find tokens that overlap with marker position
        marker_tokens = []
        for token in doc:
            if token.idx >= marker_start and token.idx < marker_end:
                marker_tokens.append(token)
            elif token.idx + len(token.text) > marker_start and token.idx < marker_end:
                marker_tokens.append(token)
        
        if marker_tokens:
            return doc[marker_tokens[0].i:marker_tokens[-1].i + 1]
        return None
    
    def _extract_lexical_features(self, marker_text: str, marker_span) -> Dict[str, Any]:
        """Extract lexical features from marker text"""
        return {
            'marker_text': marker_text,
            'marker_length': len(marker_text),
            'marker_word_count': len(marker_text.split()),
            'is_capitalized': marker_text[0].isupper() if marker_text else False,
            'has_punctuation': bool(re.search(r'[^\w\s]', marker_text))
        }
    
    def _extract_syntactic_features(self, marker_span, doc) -> Dict[str, Any]:
        """Extract POS tags and dependency relations"""
        if len(marker_span) == 1:
            token = marker_span[0]
            return {
                'pos_tag': token.pos_,
                'dependency_relation': token.dep_,
                'head_pos': token.head.pos_ if token.head != token else 'ROOT',
                'syntactic_children': [child.pos_ for child in token.children]
            }
        else:
            # Multi-token marker - use head token
            head_token = marker_span.root
            return {
                'pos_tag': '_'.join([t.pos_ for t in marker_span]),
                'dependency_relation': head_token.dep_,
                'head_pos': head_token.head.pos_ if head_token.head != head_token else 'ROOT',
                'syntactic_children': [child.pos_ for child in head_token.children]
            }
    
    def _extract_contextual_features(self, marker_span, doc) -> Dict[str, Any]:
        """Extract contextual features based on surrounding tokens"""
        start_idx = marker_span.start
        end_idx = marker_span.end
        
        # Context window of 3 tokens on each side
        left_context = doc[max(0, start_idx-3):start_idx]
        right_context = doc[end_idx:min(len(doc), end_idx+3)]
        
        # Sentence position (0.0 = start, 1.0 = end)
        sentence_start = marker_span.sent.start
        sentence_end = marker_span.sent.end
        sentence_length = sentence_end - sentence_start
        marker_position = start_idx - sentence_start
        
        return {
            'left_context_pos': [token.pos_ for token in left_context],
            'right_context_pos': [token.pos_ for token in right_context],
            'sentence_position': marker_position / sentence_length if sentence_length > 0 else 0.5,
            'distance_to_sentence_start': marker_position,
            'distance_to_sentence_end': sentence_end - end_idx
        }
    
    def _extract_semantic_features(self, marker_span, doc) -> Dict[str, Any]:
        """Extract semantic and positional features"""
        start_idx = marker_span.start
        end_idx = marker_span.end
        
        # Check position relative to sentence boundaries
        is_sentence_start = start_idx == marker_span.sent.start
        is_sentence_end = end_idx == marker_span.sent.end
        
        # Check punctuation context
        follows_punct = False
        precedes_punct = False
        
        if start_idx > 0:
            prev_token = doc[start_idx - 1]
            follows_punct = prev_token.is_punct
        
        if end_idx < len(doc):
            next_token = doc[end_idx]
            precedes_punct = next_token.is_punct
        
        return {
            'is_sentence_start': is_sentence_start,
            'is_sentence_end': is_sentence_end,
            'follows_punctuation': follows_punct,
            'precedes_punctuation': precedes_punct
        }
    
    def _extract_academic_features(self, marker_span, doc) -> Dict[str, Any]:
        """Extract academic context features"""
        # Check if marker is in an academic verb phrase
        in_academic_vp = self._is_in_academic_verb_phrase(marker_span)
        
        # Calculate academic context score based on surrounding vocabulary
        academic_score = self._calculate_academic_context_score(marker_span, doc)
        
        return {
            'in_academic_verb_phrase': in_academic_vp,
            'academic_context_score': academic_score
        }
    
    def _is_in_academic_verb_phrase(self, marker_span) -> bool:
        """Check if marker is part of or near an academic verb phrase"""
        # Look for academic verbs in the dependency tree
        for token in marker_span:
            # Check if token's head is an academic verb
            if token.head.lemma_.lower() in self.academic_verbs:
                return True
            
            # Check if token governs an academic verb
            for child in token.children:
                if child.lemma_.lower() in self.academic_verbs:
                    return True
        
        return False
    
    def _calculate_academic_context_score(self, marker_span, doc) -> float:
        """Calculate academic context score based on surrounding vocabulary"""
        # Context window of 10 tokens on each side
        start_idx = max(0, marker_span.start - 10)
        end_idx = min(len(doc), marker_span.end + 10)
        context = doc[start_idx:end_idx]
        
        academic_count = 0
        total_content_words = 0
        
        for token in context:
            if not token.is_stop and not token.is_punct and len(token.text) > 2:
                total_content_words += 1
                lemma = token.lemma_.lower()
                if lemma in self.academic_verbs or lemma in self.academic_nouns:
                    academic_count += 1
        
        return academic_count / total_content_words if total_content_words > 0 else 0.0
    
    def _create_fallback_features(self, marker_text: str) -> LinguisticFeatures:
        """Create minimal features when marker span cannot be found"""
        return LinguisticFeatures(
            marker_text=marker_text,
            marker_length=len(marker_text),
            marker_word_count=len(marker_text.split()),
            is_capitalized=marker_text[0].isupper() if marker_text else False,
            has_punctuation=bool(re.search(r'[^\w\s]', marker_text)),
            pos_tag='UNKNOWN',
            dependency_relation='UNKNOWN',
            head_pos='UNKNOWN',
            syntactic_children=[],
            left_context_pos=[],
            right_context_pos=[],
            sentence_position=0.5,
            distance_to_sentence_start=0,
            distance_to_sentence_end=0,
            is_sentence_start=False,
            is_sentence_end=False,
            follows_punctuation=False,
            precedes_punctuation=False,
            in_academic_verb_phrase=False,
            academic_context_score=0.0
        )
    
    def extract_features_from_dataset(self, df: pd.DataFrame, 
                                    text_col: str = 'text',
                                    marker_col: str = 'marker_text',
                                    batch_size: int = 100) -> pd.DataFrame:
        """
        Extract features for all samples in a dataset with batch processing for efficiency
        
        Args:
            df: DataFrame with text and marker columns
            text_col: Name of column containing full text
            marker_col: Name of column containing marker text
            batch_size: Number of samples to process in each batch
            
        Returns:
            DataFrame with extracted features as additional columns
        """
        features_list = []
        
        print(f"Extracting features for {len(df)} samples using transformer model...")
        
        # Process in batches for efficiency with transformer model
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(df)-1)//batch_size + 1} (samples {batch_start}-{batch_end-1})")
            
            # Prepare texts for batch processing
            texts = batch_df[text_col].tolist()
            
            # Process texts in batch (more efficient for transformer models)
            docs = list(self.nlp.pipe(texts, batch_size=min(32, len(texts))))
            
            # Extract features for each sample in the batch
            for idx, (_, row) in enumerate(batch_df.iterrows()):
                marker_text = row[marker_col]
                text = row[text_col]
                doc = docs[idx]
                
                # Find marker position in text
                marker_start = text.lower().find(marker_text.lower())
                if marker_start == -1:
                    marker_start = 0  # Fallback
                
                # Use pre-processed doc for efficiency
                features = self._extract_features_from_doc(doc, marker_text, marker_start)
                features_dict = self._features_to_dict(features)
                features_list.append(features_dict)
        
        # Convert to DataFrame and combine with original
        features_df = pd.DataFrame(features_list)
        result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
        return result_df
    
    def _extract_features_from_doc(self, doc, marker_text: str, marker_start: int) -> LinguisticFeatures:
        """Extract features from a pre-processed spacy doc (for batch processing efficiency)"""
        
        # Find the marker span in the processed document
        marker_span = self._find_marker_span(doc, marker_text, marker_start)
        if not marker_span:
            # Fallback: create minimal features if marker not found
            return self._create_fallback_features(marker_text)
        
        # Extract all feature categories
        lexical_features = self._extract_lexical_features(marker_text, marker_span)
        syntactic_features = self._extract_syntactic_features(marker_span, doc)
        contextual_features = self._extract_contextual_features(marker_span, doc)
        semantic_features = self._extract_semantic_features(marker_span, doc)
        academic_features = self._extract_academic_features(marker_span, doc)
        
        return LinguisticFeatures(
            # Lexical
            marker_text=lexical_features['marker_text'],
            marker_length=lexical_features['marker_length'],
            marker_word_count=lexical_features['marker_word_count'],
            is_capitalized=lexical_features['is_capitalized'],
            has_punctuation=lexical_features['has_punctuation'],
            
            # Syntactic
            pos_tag=syntactic_features['pos_tag'],
            dependency_relation=syntactic_features['dependency_relation'],
            head_pos=syntactic_features['head_pos'],
            syntactic_children=syntactic_features['syntactic_children'],
            
            # Contextual
            left_context_pos=contextual_features['left_context_pos'],
            right_context_pos=contextual_features['right_context_pos'],
            sentence_position=contextual_features['sentence_position'],
            distance_to_sentence_start=contextual_features['distance_to_sentence_start'],
            distance_to_sentence_end=contextual_features['distance_to_sentence_end'],
            
            # Semantic
            is_sentence_start=semantic_features['is_sentence_start'],
            is_sentence_end=semantic_features['is_sentence_end'],
            follows_punctuation=semantic_features['follows_punctuation'],
            precedes_punctuation=semantic_features['precedes_punctuation'],
            
            # Academic
            in_academic_verb_phrase=academic_features['in_academic_verb_phrase'],
            academic_context_score=academic_features['academic_context_score']
        )
    
    def _features_to_dict(self, features: LinguisticFeatures) -> Dict[str, Any]:
        """Convert LinguisticFeatures object to dictionary for DataFrame"""
        return {
            'feat_marker_length': features.marker_length,
            'feat_marker_word_count': features.marker_word_count,
            'feat_is_capitalized': features.is_capitalized,
            'feat_has_punctuation': features.has_punctuation,
            'feat_pos_tag': features.pos_tag,
            'feat_dependency_relation': features.dependency_relation,
            'feat_head_pos': features.head_pos,
            'feat_syntactic_children_count': len(features.syntactic_children),
            'feat_left_context_pos_count': len(features.left_context_pos),
            'feat_right_context_pos_count': len(features.right_context_pos),
            'feat_sentence_position': features.sentence_position,
            'feat_distance_to_sentence_start': features.distance_to_sentence_start,
            'feat_distance_to_sentence_end': features.distance_to_sentence_end,
            'feat_is_sentence_start': features.is_sentence_start,
            'feat_is_sentence_end': features.is_sentence_end,
            'feat_follows_punctuation': features.follows_punctuation,
            'feat_precedes_punctuation': features.precedes_punctuation,
            'feat_in_academic_verb_phrase': features.in_academic_verb_phrase,
            'feat_academic_context_score': features.academic_context_score,
            'feat_embedding_similarity_to_academic': features.embedding_similarity_to_academic
        }

if __name__ == "__main__":
    # Test the feature extractor
    extractor = SpacyFeatureExtractor()
    
    # Test with example text
    test_text = "This study aims to demonstrate the effectiveness of the proposed method. However, further research is needed."
    test_marker = "However"
    test_start = test_text.find(test_marker)
    
    features = extractor.extract_features(test_text, test_marker, test_start)
    print("Extracted features:")
    print(f"POS tag: {features.pos_tag}")
    print(f"Dependency: {features.dependency_relation}")
    print(f"Academic context score: {features.academic_context_score}")
    print(f"Sentence position: {features.sentence_position}")