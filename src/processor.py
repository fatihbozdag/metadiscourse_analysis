"""Text processing module for metadiscourse analysis."""

import os
import re
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
import torch
torch.device('mps')
import spacy
import numpy as np
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Token, Span, Doc
from collections import defaultdict, Counter
import logging
import warnings
warnings.filterwarnings('ignore')

from markers import INTERACTIVE_MARKERS, INTERACTIONAL_MARKERS

class TextProcessor:
    """Process and analyze text documents for metadiscourse markers."""
    
    def __init__(self, model_name='en_core_web_trf'):
        """Initialize the processor with a spaCy model."""
        spacy.prefer_gpu()
        self.nlp = spacy.load(model_name)
        
        # Initialize simple marker lists
        self.interactive_markers = INTERACTIVE_MARKERS
        self.interactional_markers = INTERACTIONAL_MARKERS
        
        # Regex pattern for cleaning text
        self.clean_pattern = re.compile(r'[^\w\s]')
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for better marker detection.
        
        This preprocessing step:
        1. Normalizes whitespace
        2. Preserves important punctuation for contractions and hyphenated words
        3. Handles special cases like apostrophes in contractions
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Preserve contractions and special linguistic markers by temporarily replacing them
        contractions = [
            "'s", "'ve", "'re", "'ll", "'d", "'t", "'m",  # apostrophe contractions
            "n't", "cannot", "can't", "won't", "wouldn't",
            "i.e.", "e.g.", "etc.", "et al.", "vs.", "fig.", "tab."
        ]
        
        # Temporarily replace contractions with placeholders
        for i, contraction in enumerate(contractions):
            text = text.replace(contraction, f"__CONTRACTION_{i}__")
        
        # Preserve hyphenated words
        hyphen_pattern = re.compile(r'(\w+)-(\w+)')
        text = hyphen_pattern.sub(r'\1__HYPHEN__\2', text)
        
        # Clean text but preserve alphanumeric characters and whitespace
        # Use a more selective cleaning approach to maintain important linguistic features
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Restore contractions
        for i, contraction in enumerate(contractions):
            text = text.replace(f"__CONTRACTION_{i}__", contraction)
        
        # Restore hyphens
        text = text.replace("__HYPHEN__", "-")
        
        return text
    
    def process_text(self, text: str):
        """Process a single text and return marker counts."""
        try:
            # Clean text
            clean_text = self.clean_text(text)
            
            # Process with spaCy
            doc = self.nlp(clean_text)
            
            # Calculate word count
            word_count = len([token for token in doc if not token.is_punct and token.is_alpha])
            
            # Initialize counts for all categories
            interactive_counts = {
                'transitions': 0,
                'frame_markers': 0,
                'endophoric_markers': 0,
                'evidentials': 0,
                'code_glosses': 0
            }
            
            interactional_counts = {
                'hedges': 0,
                'boosters': 0,
                'attitude_markers': 0,
                'engagement_markers': 0,
                'self_mentions': 0
            }
            
            # Enhanced marker detection with context awareness
            text_lower = doc.text.lower()
            
            # Function to check if a marker is present with proper boundaries
            def is_marker_present(marker, text):
                # For single-word markers, check if they exist as whole words
                if len(marker.split()) == 1:
                    # Use regex to find whole word matches
                    pattern = r'\b' + re.escape(marker) + r'\b'
                    return bool(re.search(pattern, text))
                else:
                    # For multi-word phrases, allow for some flexibility
                    words = marker.split()
                    
                    # Check for exact phrase match first
                    if marker in text:
                        return True
                    
                    # Check for phrases with intervening punctuation or words (up to 2)
                    if len(words) > 1:
                        first_word_pattern = r'\b' + re.escape(words[0]) + r'\b'
                        last_word_pattern = r'\b' + re.escape(words[-1]) + r'\b'
                        
                        first_match = re.search(first_word_pattern, text)
                        last_match = re.search(last_word_pattern, text)
                        
                        if first_match and last_match:
                            # Check if the matches are in the right order and not too far apart
                            # Allow for up to 3 words between each marker word
                            first_pos = first_match.start()
                            last_pos = last_match.start()
                            
                            if first_pos < last_pos:
                                # Check if the intervening text is reasonable in length
                                intervening_text = text[first_pos:last_pos]
                                word_count = len(intervening_text.split())
                                
                                # Allow for reasonable number of intervening words based on phrase length
                                max_allowed = len(words) + 2  # Original words plus 2 extra
                                
                                if word_count <= max_allowed:
                                    # Check if middle words are present in order if there are more than 2 words
                                    if len(words) > 2:
                                        middle_words_present = True
                                        for middle_word in words[1:-1]:
                                            if middle_word not in intervening_text:
                                                middle_words_present = False
                                                break
                                        return middle_words_present
                                    return True
                    
                    return False
            
            # Count interactive markers with improved detection
            for category, markers in self.interactive_markers.items():
                for marker in markers:
                    if is_marker_present(marker, text_lower):
                        interactive_counts[category] += 1
            
            # Count interactional markers with improved detection
            for category, markers in self.interactional_markers.items():
                for marker in markers:
                    if is_marker_present(marker, text_lower):
                        interactional_counts[category] += 1
            
            # Handle polyfunctional markers (markers that can belong to multiple categories)
            polyfunctional_markers = {
                # Markers that can be both code glosses and boosters
                "in fact": [("interactive", "code_glosses"), ("interactional", "boosters")],
                "indeed": [("interactive", "code_glosses"), ("interactional", "boosters")],
                "actually": [("interactive", "code_glosses"), ("interactional", "boosters")],
                
                # Words that can be both transitions and frame markers
                "then": [("interactive", "transitions"), ("interactive", "frame_markers")],
                "next": [("interactive", "transitions"), ("interactive", "frame_markers")],
                
                # Words that can be both hedges and engagement markers
                "should": [("interactional", "hedges"), ("interactional", "engagement_markers")],
                "must": [("interactional", "boosters"), ("interactional", "engagement_markers")]
            }
            
            # Process polyfunctional markers
            for marker, categories in polyfunctional_markers.items():
                if is_marker_present(marker, text_lower):
                    for marker_type, category in categories:
                        if marker_type == "interactive":
                            interactive_counts[category] += 1
                        else:
                            interactional_counts[category] += 1
            
            # Combine counts
            flat_counts = {
                'interactive': interactive_counts,
                'interactional': interactional_counts
            }
            
            # Calculate frequencies per 1000 words
            frequencies = {
                'interactive': {},
                'interactional': {}
            }
            
            # Calculate frequencies if word count is positive
            if word_count > 0:
                for category, count in interactive_counts.items():
                    frequencies['interactive'][category] = (count / word_count) * 1000
                
                for category, count in interactional_counts.items():
                    frequencies['interactional'][category] = (count / word_count) * 1000
            else:
                # Set all frequencies to 0 if word count is 0
                for category in interactive_counts:
                    frequencies['interactive'][category] = 0
                for category in interactional_counts:
                    frequencies['interactional'][category] = 0
            
            return flat_counts, frequencies, word_count
            
        except Exception as e:
            print(f"Error in process_text: {str(e)}")
            # Return empty results with zeros
            empty_interactive = {'transitions': 0, 'frame_markers': 0, 'endophoric_markers': 0, 'evidentials': 0, 'code_glosses': 0}
            empty_interactional = {'hedges': 0, 'boosters': 0, 'attitude_markers': 0, 'engagement_markers': 0, 'self_mentions': 0}
            
            empty_counts = {'interactive': empty_interactive, 'interactional': empty_interactional}
            empty_frequencies = {'interactive': dict(empty_interactive), 'interactional': dict(empty_interactional)}
            
            return empty_counts, empty_frequencies, 0
    
    def process_corpus(self, input_path: str, text_field='text_field'):
        """Process a corpus of texts from a CSV file.
        
        Args:
            input_path: Path to the CSV file containing texts
            text_field: Name of the column containing the text to analyze
            
        Returns:
            DataFrame with analysis results
        """
        try:
            df = pd.read_csv(input_path)
            
            # Check if text field exists
            if text_field not in df.columns:
                raise ValueError(f"Text field '{text_field}' not found in CSV. Available columns: {', '.join(df.columns)}")
                
        except Exception as e:
            raise IOError(f"Error reading CSV file: {str(e)}")
        
        # Process texts
        results = []
        errors = []
        print("Processing texts...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # Convert to string to handle potential non-string inputs
                text = str(row[text_field])
                
                # Skip empty texts
                if not text or text.isspace():
                    errors.append({"document": f"doc_{idx}", "error": "Empty text"})
                    continue
                    
                counts, frequencies, word_count = self.process_text(text)
                
                result = {
                    'document': f"doc_{idx}",
                    'word_count': word_count
                }
                
                # Add metadata
                for col in df.columns:
                    if col != text_field:
                        result[col] = row[col]
                
                # Add marker counts and frequencies
                for category, count in counts["interactive"].items():
                    result[f'interactive_{category}_count'] = count
                    result[f'interactive_{category}_freq'] = frequencies["interactive"][category]
                
                for category, count in counts["interactional"].items():
                    result[f'interactional_{category}_count'] = count
                    result[f'interactional_{category}_freq'] = frequencies["interactional"][category]
                
                results.append(result)
                
            except Exception as e:
                error_msg = f"Error processing document {idx}: {str(e)}"
                print(error_msg)
                errors.append({"document": f"doc_{idx}", "error": str(e)})
                continue
        
        # Print error summary
        if errors:
            print(f"\nEncountered {len(errors)} errors during processing out of {len(df)} documents ({len(errors)/len(df)*100:.1f}%)")
        
        # Save errors to a separate DataFrame if there are any
        if errors:
            error_df = pd.DataFrame(errors)
            error_path = os.path.join(os.path.dirname(input_path), 'processing_errors.csv')
            error_df.to_csv(error_path, index=False)
            print(f"Saved error details to {error_path}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def get_summary_stats(self, df: pd.DataFrame):
        """Calculate summary statistics for the corpus."""
        # Calculate basic statistics
        total_texts = len(df)
        total_words = df['word_count'].sum()
        avg_words = df['word_count'].mean()
        
        print(f"\nCorpus Summary:\n")
        print(f"Total texts: {total_texts}")
        print(f"Total words: {total_words}")
        print(f"Average words per text: {avg_words:.2f}")
        
        # Calculate statistics for each marker category
        for col in df.columns:
            if col.endswith('_freq'):
                mean_val = df[col].mean()
                std_val = df[col].std()
                print(f"{col}: Mean = {mean_val:.2f}, Std = {std_val:.2f}")
        
        return {
            'total_texts': total_texts,
            'total_words': total_words,
            'avg_words': avg_words
        }

class EnhancedTextProcessor:
    """Enhanced text processor with improved context awareness and polyfunctional marker handling."""
    
    def __init__(self, model_name="en_core_web_sm", use_gpu=False):
        """Initialize the enhanced text processor."""
        self.model_name = model_name
        self.use_gpu = use_gpu
        
        # Initialize spaCy model
        try:
            if use_gpu:
                spacy.prefer_gpu()
            self.nlp = spacy.load(model_name)
            # Add custom sentence boundary detection
            self.nlp.add_pipe("sentencizer", first=True)
        except OSError:
            logging.warning(f"Model {model_name} not found. Using en_core_web_sm instead.")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize enhanced markers
        from markers import EnhancedMetadiscourseMarkers
        self.marker_system = EnhancedMetadiscourseMarkers()
        
        # Initialize matchers
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.pattern_matcher = Matcher(self.nlp.vocab)
        
        # Setup patterns
        self._setup_enhanced_patterns()
        
        # Context analysis components
        self.context_analyzer = ContextAnalyzer(self.nlp)
        self.polyfunctional_resolver = PolyfunctionalResolver(self.marker_system)
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'errors': 0,
            'polyfunctional_resolved': 0,
            'context_filtered': 0
        }
    
    def _setup_enhanced_patterns(self):
        """Setup enhanced pattern matching for better accuracy."""
        
        # Interactive marker patterns
        interactive_patterns = []
        for category, markers in self.marker_system.interactive_markers.items():
            if isinstance(markers, dict):
                for subcategory, marker_list in markers.items():
                    for marker in marker_list:
                        pattern_doc = self.nlp.make_doc(marker.lower())
                        interactive_patterns.append(pattern_doc)
            else:
                for marker in markers:
                    pattern_doc = self.nlp.make_doc(marker.lower())
                    interactive_patterns.append(pattern_doc)
        
        # Interactional marker patterns
        interactional_patterns = []
        for category, markers in self.marker_system.interactional_markers.items():
            if isinstance(markers, dict):
                for subcategory, marker_list in markers.items():
                    for marker in marker_list:
                        pattern_doc = self.nlp.make_doc(marker.lower())
                        interactional_patterns.append(pattern_doc)
            else:
                for marker in markers:
                    pattern_doc = self.nlp.make_doc(marker.lower())
                    interactional_patterns.append(pattern_doc)
        
        # Add patterns to phrase matcher
        self.phrase_matcher.add("INTERACTIVE", interactive_patterns)
        self.phrase_matcher.add("INTERACTIONAL", interactional_patterns)
        
        # Add context-sensitive patterns
        self._add_context_patterns()
    
    def _add_context_patterns(self):
        """Add context-sensitive patterns for better accuracy."""
        
        # Modal verb patterns (for hedging vs other functions)
        modal_hedge_pattern = [
            {"LOWER": {"IN": ["may", "might", "could", "would"]}},
            {"LOWER": {"IN": ["be", "have", "seem", "appear"]}, "OP": "?"},
            {"POS": {"IN": ["VERB", "ADJ"]}}
        ]
        self.pattern_matcher.add("MODAL_HEDGE", [modal_hedge_pattern])
        
        # Citation patterns (for evidentials)
        citation_pattern = [
            {"LOWER": {"IN": ["according", "states", "argues", "notes", "suggests"]}},
            {"LOWER": "to", "OP": "?"},
            {"POS": "PROPN", "OP": "+"}
        ]
        self.pattern_matcher.add("CITATION", [citation_pattern])
        
        # Self-mention patterns (organizational vs personal)
        org_self_mention = [
            {"LOWER": {"IN": ["i", "we"]}},
            {"LOWER": {"IN": ["argue", "suggest", "propose", "claim", "analyze", "examine"]}}
        ]
        self.pattern_matcher.add("ORG_SELF_MENTION", [org_self_mention])
        
        # Engagement patterns (direct address vs inclusive)
        direct_engagement = [
            {"LOWER": {"IN": ["you", "your"]}},
            {"POS": {"IN": ["VERB", "NOUN", "ADJ"]}}
        ]
        self.pattern_matcher.add("DIRECT_ENGAGEMENT", [direct_engagement])
    
    def process_text_enhanced(self, text: str, text_id: str = None) -> Dict:
        """Enhanced text processing with improved accuracy."""
        
        if not text or not isinstance(text, str):
            return self._empty_result()
        
        try:
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            
            # Process with spaCy
            doc = self.nlp(cleaned_text)
            
            # Extract markers with context analysis
            markers = self._extract_markers_with_context(doc)
            
            # Resolve polyfunctional markers
            resolved_markers = self.polyfunctional_resolver.resolve_markers(doc, markers)
            
            # Calculate statistics
            stats = self._calculate_enhanced_stats(resolved_markers, doc)
            
            # Update performance tracking
            self.stats['total_processed'] += 1
            if resolved_markers != markers:
                self.stats['polyfunctional_resolved'] += 1
            
            return {
                'text_id': text_id,
                'word_count': len([token for token in doc if not token.is_space]),
                'sentence_count': len(list(doc.sents)),
                'markers': resolved_markers,
                'statistics': stats,
                'processing_info': {
                    'model_used': self.model_name,
                    'polyfunctional_resolved': resolved_markers != markers,
                    'context_filtered': self.stats['context_filtered']
                }
            }
            
        except Exception as e:
            self.stats['errors'] += 1
            logging.error(f"Error processing text {text_id}: {str(e)}")
            return self._empty_result(error=str(e))
    
    def _empty_result(self, error=None):
        """Return empty result structure."""
        return {
            'text_id': None,
            'word_count': 0,
            'sentence_count': 0,
            'markers': {},
            'statistics': {},
            'processing_info': {'error': error or 'Failed to process text'}
        }
    
    def _clean_text(self, text):
        """Clean and preprocess text."""
        if not isinstance(text, str):
            return ""
        
        # Basic text cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        return text
    
    def _extract_markers_with_context(self, doc: Doc) -> Dict:
        """Extract markers with enhanced context analysis."""
        
        markers = defaultdict(list)
        
        # Use phrase matcher for initial detection
        matches = self.phrase_matcher(doc)
        pattern_matches = self.pattern_matcher(doc)
        
        # Process phrase matches
        for match_id, start, end in matches:
            span = doc[start:end]
            marker_text = span.text.lower()
            
            # Determine category and subcategory
            category_info = self._categorize_marker(marker_text, span)
            
            if category_info:
                try:
                    markers[category_info['category']].append({
                        'text': marker_text,
                        'start': start,
                        'end': end,
                        'subcategory': category_info['subcategory'],
                        'confidence': category_info['confidence'],
                        'context': self._extract_context(span)
                    })
                except KeyError as e:
                    logging.error(f"KeyError accessing category_info: {e}")
                    logging.error(f"category_info: {category_info}")
                    logging.error(f"marker_text: '{marker_text}'")
                    raise
        
        # Process pattern matches for context-sensitive detection
        for match_id, start, end in pattern_matches:
            span = doc[start:end]
            pattern_name = self.nlp.vocab.strings[match_id]
            
            # Handle context-specific patterns
            if pattern_name == "MODAL_HEDGE":
                self._process_modal_hedge(span, markers)
            elif pattern_name == "CITATION":
                self._process_citation(span, markers)
            elif pattern_name == "ORG_SELF_MENTION":
                self._process_self_mention(span, markers)
            elif pattern_name == "DIRECT_ENGAGEMENT":
                self._process_engagement(span, markers)
        
        return dict(markers)
    
    def _process_modal_hedge(self, span: Span, markers: Dict):
        """Process modal hedge patterns."""
        marker_text = span.text.lower()
        markers['interactional_hedges'].append({
            'text': marker_text,
            'start': span.start,
            'end': span.end,
            'subcategory': 'modal',
            'confidence': 0.85,
            'context': self._extract_context(span)
        })
    
    def _process_citation(self, span: Span, markers: Dict):
        """Process citation patterns."""
        marker_text = span.text.lower()
        markers['interactive_evidentials'].append({
            'text': marker_text,
            'start': span.start,
            'end': span.end,
            'subcategory': 'attribution',
            'confidence': 0.9,
            'context': self._extract_context(span)
        })
    
    def _process_self_mention(self, span: Span, markers: Dict):
        """Process self-mention patterns."""
        marker_text = span.text.lower()
        markers['interactional_self_mentions'].append({
            'text': marker_text,
            'start': span.start,
            'end': span.end,
            'subcategory': 'organizational',
            'confidence': 0.9,
            'context': self._extract_context(span)
        })
    
    def _process_engagement(self, span: Span, markers: Dict):
        """Process engagement patterns."""
        marker_text = span.text.lower()
        markers['interactional_engagement_markers'].append({
            'text': marker_text,
            'start': span.start,
            'end': span.end,
            'subcategory': 'direct_address',
            'confidence': 0.85,
            'context': self._extract_context(span)
        })
    
    def _categorize_marker(self, marker_text: str, span: Span) -> Optional[Dict]:
        """Categorize marker with confidence scoring."""
        
        # Check if marker is polyfunctional
        if marker_text in self.marker_system.polyfunctional_markers:
            return self._handle_polyfunctional_marker(marker_text, span)
        
        # Check interactive markers
        for category, subcategories in self.marker_system.interactive_markers.items():
            for subcategory, markers in subcategories.items():
                if marker_text in markers:
                    return {
                        'category': f'interactive_{category}',
                        'subcategory': subcategory,
                        'confidence': 0.9
                    }
        
        # Check interactional markers
        for category, subcategories in self.marker_system.interactional_markers.items():
            for subcategory, markers in subcategories.items():
                if marker_text in markers:
                    return {
                        'category': f'interactional_{category}',
                        'subcategory': subcategory,
                        'confidence': 0.9
                    }
        
        return None
    
    def _handle_polyfunctional_marker(self, marker_text: str, span: Span) -> Dict:
        """Handle polyfunctional markers with context-based resolution."""
        
        functions = self.marker_system.polyfunctional_markers[marker_text]
        
        # Use context to determine most likely function
        context_scores = []
        for func_type, category, subcategory, base_confidence in functions:
            context_score = self.context_analyzer.score_context(span, func_type, category, subcategory)
            total_score = base_confidence * context_score
            context_scores.append((func_type, category, subcategory, total_score))
        
        # Select highest scoring function
        best_function = max(context_scores, key=lambda x: x[3])
        
        return {
            'category': f'{best_function[0]}_{best_function[1]}',
            'subcategory': best_function[2],
            'confidence': best_function[3]
        }
    
    def _extract_context(self, span: Span) -> Dict:
        """Extract contextual information for a marker."""
        
        # Get surrounding context
        sent = span.sent
        context_start = max(0, span.start - sent.start - 3)
        context_end = min(len(sent), span.end - sent.start + 3)
        
        left_context = sent[context_start:span.start - sent.start]
        right_context = sent[span.end - sent.start:context_end]
        
        return {
            'sentence': sent.text,
            'left_context': left_context.text if left_context else "",
            'right_context': right_context.text if right_context else "",
            'pos_tags': [token.pos_ for token in span],
            'dependencies': [token.dep_ for token in span],
            'sentence_position': (span.start - sent.start) / len(sent)
        }
    
    def _calculate_enhanced_stats(self, markers: Dict, doc: Doc) -> Dict:
        """Calculate enhanced statistics with subcategory breakdown."""
        
        word_count = len([token for token in doc if not token.is_space])
        stats = {}
        
        # Calculate frequencies per category and subcategory
        for category, marker_list in markers.items():
            category_count = len(marker_list)
            stats[f'{category}_count'] = category_count
            stats[f'{category}_frequency'] = (category_count / word_count * 1000) if word_count > 0 else 0
            
            # Subcategory breakdown
            try:
                subcategory_counts = Counter([marker.get('subcategory', 'unknown') for marker in marker_list if isinstance(marker, dict)])
                for subcategory, count in subcategory_counts.items():
                    stats[f'{category}_{subcategory}_count'] = count
                    stats[f'{category}_{subcategory}_frequency'] = (count / word_count * 1000) if word_count > 0 else 0
            except Exception as e:
                logging.error(f"Error calculating subcategory stats for {category}: {e}")
                logging.error(f"Marker list: {marker_list}")
                stats[f'{category}_unknown_count'] = len(marker_list)
                stats[f'{category}_unknown_frequency'] = (len(marker_list) / word_count * 1000) if word_count > 0 else 0
        
        # Overall statistics
        total_interactive = sum(len(markers.get(cat, [])) for cat in markers if cat.startswith('interactive_'))
        total_interactional = sum(len(markers.get(cat, [])) for cat in markers if cat.startswith('interactional_'))
        
        stats['total_interactive'] = total_interactive
        stats['total_interactional'] = total_interactional
        stats['total_markers'] = total_interactive + total_interactional
        stats['interactive_frequency'] = (total_interactive / word_count * 1000) if word_count > 0 else 0
        stats['interactional_frequency'] = (total_interactional / word_count * 1000) if word_count > 0 else 0
        stats['total_frequency'] = (stats['total_markers'] / word_count * 1000) if word_count > 0 else 0
        
        return stats

class ContextAnalyzer:
    """Analyzes context to improve marker categorization accuracy."""
    
    def __init__(self, nlp):
        self.nlp = nlp
        
        # Context patterns for different functions
        self.context_patterns = {
            'hedge_indicators': ['uncertain', 'possible', 'likely', 'suggest', 'appear', 'seem'],
            'booster_indicators': ['certain', 'clear', 'obvious', 'definite', 'prove', 'demonstrate'],
            'citation_indicators': ['author', 'study', 'research', 'paper', 'article', 'scholar'],
            'engagement_indicators': ['reader', 'audience', 'consider', 'note', 'think'],
            'transition_indicators': ['sentence', 'paragraph', 'section', 'argument', 'point']
        }
    
    def score_context(self, span: Span, func_type: str, category: str, subcategory: str) -> float:
        """Score context relevance for a specific function."""
        
        sent = span.sent
        context_words = [token.text.lower() for token in sent if not token.is_stop]
        
        # Base score
        score = 0.5
        
        # Category-specific scoring
        if category == 'hedges':
            hedge_words = set(context_words) & set(self.context_patterns['hedge_indicators'])
            score += len(hedge_words) * 0.1
            
        elif category == 'boosters':
            booster_words = set(context_words) & set(self.context_patterns['booster_indicators'])
            score += len(booster_words) * 0.1
            
        elif category == 'evidentials':
            citation_words = set(context_words) & set(self.context_patterns['citation_indicators'])
            score += len(citation_words) * 0.15
            
        elif category == 'engagement_markers':
            engagement_words = set(context_words) & set(self.context_patterns['engagement_indicators'])
            score += len(engagement_words) * 0.1
            
        elif category == 'transitions':
            transition_words = set(context_words) & set(self.context_patterns['transition_indicators'])
            score += len(transition_words) * 0.1
        
        # Position-based scoring
        sent_position = (span.start - sent.start) / len(sent)
        
        if category in ['transitions', 'frame_markers']:
            # Transitions often appear at sentence beginnings
            if sent_position < 0.3:
                score += 0.1
        elif category in ['code_glosses', 'evidentials']:
            # Code glosses and evidentials often appear mid-sentence
            if 0.2 < sent_position < 0.8:
                score += 0.1
        
        return min(1.0, score)

class PolyfunctionalResolver:
    """Resolves polyfunctional markers based on context and confidence."""
    
    def __init__(self, marker_system):
        self.marker_system = marker_system
    
    def resolve_markers(self, doc: Doc, markers: Dict) -> Dict:
        """Resolve polyfunctional markers to their most likely function."""
        
        resolved_markers = defaultdict(list)
        
        for category, marker_list in markers.items():
            for marker in marker_list:
                try:
                    # Check if marker needs resolution
                    if (isinstance(marker, dict) and 
                        marker.get('confidence', 1.0) < 0.8 and 
                        marker.get('text', '') in self.marker_system.polyfunctional_markers):
                        # Apply additional resolution logic
                        resolved_category = self._resolve_polyfunctional(marker, doc)
                        resolved_markers[resolved_category].append(marker)
                    else:
                        resolved_markers[category].append(marker)
                except Exception as e:
                    logging.error(f"Error resolving marker {marker}: {e}")
                    # Fall back to original category
                    resolved_markers[category].append(marker)
        
        return dict(resolved_markers)
    
    def _resolve_polyfunctional(self, marker: Dict, doc: Doc) -> str:
        """Resolve a polyfunctional marker to its most likely category."""
        
        # Implementation of advanced resolution logic
        # This is a simplified version - could be expanded with ML models
        
        marker_text = marker['text']
        context = marker['context']
        
        # Context-based resolution rules
        if marker_text in ['in fact', 'indeed', 'actually']:
            # Check if used for emphasis (booster) or clarification (code gloss)
            if any(word in context['left_context'].lower() for word in ['but', 'however', 'although']):
                return 'interactional_boosters'
            else:
                return 'interactive_code_glosses'
        
        # Default to highest confidence category
        return marker.get('category', 'unknown_category')