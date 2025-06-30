#!/usr/bin/env python3
"""
Enhanced Filtering System for Metadiscourse Marker Detection
Addresses over-detection issues by implementing stricter contextual criteria
"""

import re
from typing import List, Dict, Set, Tuple
from spacy.tokens import Span, Doc

class MetadiscourseFilter:
    """Advanced filtering system to reduce false positives in metadiscourse detection."""
    
    def __init__(self):
        # Words that are often grammatical rather than metadiscourse
        self.high_frequency_exclusions = {
            'pronouns_content_contexts': {
                'you', 'your', 'we', 'our', 'i', 'my', 'me', 'us'
            },
            'common_conjunctions': {
                'but', 'so', 'because', 'also', 'after', 'before', 'since'
            },
            'modal_verbs_content': {
                'will', 'would', 'can', 'could', 'may', 'might', 'should'
            }
        }
        
        # Contexts where pronouns are likely content, not metadiscourse
        self.content_contexts = {
            'personal_experience': [
                r'\b(when|where|how|what|why)\s+(i|we|you)',
                r'\b(i|we|you)\s+(was|were|am|is|are|have|had)',
                r'\b(my|our|your)\s+(life|experience|family|friend|job|work|school)'
            ],
            'narrative_contexts': [
                r'\b(i|we|you)\s+(went|came|saw|met|did|made)',
                r'\b(yesterday|today|tomorrow|last|next)\s.*(i|we|you)',
                r'\b(i|we|you)\s+(remember|forgot|learned|studied)'
            ]
        }
        
        # Metadiscourse-specific contexts (keep these)
        self.metadiscourse_contexts = {
            'organizational': [
                r'\b(i|we)\s+(will|shall)\s+(discuss|examine|analyze|present|show)',
                r'\b(i|we)\s+(argue|claim|suggest|propose|conclude)',
                r'\b(as\s+)?(i|we)\s+(mentioned|noted|stated|discussed)',
                r'\bin\s+my\s+(view|opinion|analysis|study|research)'
            ],
            'stance_taking': [
                r'\b(i|we)\s+(believe|think|feel|assume|suppose)\s+that',
                r'\bin\s+my\s+(opinion|view|judgment)',
                r'\b(i|we)\s+(would\s+)?(argue|contend|maintain)\s+that'
            ],
            'reader_engagement': [
                r'\b(you\s+)?(can|should|must|need\s+to)\s+(see|note|consider|observe)',
                r'\bas\s+you\s+(can\s+see|know|might\s+expect)',
                r'\bimagine\s+(that\s+)?you',
                r'\bconsider\s+(the\s+)?(following|this|that)'
            ]
        }
        
        # Transition words that need discourse function check
        self.transition_discourse_patterns = {
            'adversative': [
                r'\b(however|nevertheless|nonetheless),?\s+',
                r'\bon\s+the\s+other\s+hand,?\s+',
                r'\b(but|yet)\s+(?!only|also|still|just)'  # but not as simple conjunction
            ],
            'causal': [
                r'\b(therefore|thus|consequently|hence),?\s+',
                r'\bas\s+a\s+result,?\s+',
                r'\bfor\s+this\s+reason,?\s+'
            ],
            'additive': [
                r'\b(furthermore|moreover|additionally|besides),?\s+',
                r'\bin\s+addition,?\s+',
                r'\balso,?\s+(?=\w+ly|\w+ing|\w+ed)'  # also + adverb/participle
            ]
        }
    
    def filter_pronouns(self, span: Span, marker_text: str, category: str) -> bool:
        """Filter out pronouns that are likely content rather than metadiscourse."""
        
        if marker_text.lower() not in self.high_frequency_exclusions['pronouns_content_contexts']:
            return True
            
        sent_text = span.sent.text.lower()
        
        # Check for content contexts (exclude if found)
        for context_type, patterns in self.content_contexts.items():
            for pattern in patterns:
                if re.search(pattern, sent_text):
                    return False
        
        # Check for metadiscourse contexts (include if found)
        for context_type, patterns in self.metadiscourse_contexts.items():
            for pattern in patterns:
                if re.search(pattern, sent_text):
                    return True
        
        # Default: exclude high-frequency pronouns without clear metadiscourse context
        return False
    
    def filter_transitions(self, span: Span, marker_text: str) -> bool:
        """Filter transitions to ensure they have discourse function."""
        
        marker_lower = marker_text.lower()
        
        # High-confidence transitions (keep)
        high_confidence_transitions = {
            'however', 'nevertheless', 'nonetheless', 'furthermore', 'moreover',
            'therefore', 'thus', 'consequently', 'in addition', 'on the other hand'
        }
        
        if marker_lower in high_confidence_transitions:
            return True
        
        # Check if common words have discourse function
        if marker_lower in self.high_frequency_exclusions['common_conjunctions']:
            return self._has_discourse_function(span, marker_lower)
        
        return True
    
    def _has_discourse_function(self, span: Span, marker_text: str) -> bool:
        """Check if a transition word has discourse function vs. grammatical function."""
        
        sent = span.sent
        marker_pos = span.start - sent.start
        
        # Position-based filtering
        sentence_position = marker_pos / len(sent)
        
        # Discourse markers often appear at sentence beginnings or after punctuation
        if sentence_position < 0.2:  # Beginning of sentence
            return True
            
        # Check if preceded by punctuation (comma, semicolon, period)
        if marker_pos > 0:
            prev_token = sent[marker_pos - 1]
            if prev_token.text in [',', ';', '.', ':', '—', '–']:
                return True
        
        # Check specific patterns
        sent_text = sent.text.lower()
        
        if marker_text == 'but':
            # "but" as discourse marker vs. simple conjunction
            # Discourse: "But this approach has limitations"
            # Simple: "not X but Y"
            if re.search(r'\bnot\s+\w+\s+but\s+', sent_text):
                return False
            if sentence_position < 0.3:
                return True
                
        elif marker_text == 'so':
            # "so" as conclusion marker vs. intensifier
            # Discourse: "So, we can conclude..."
            # Intensifier: "so important", "so difficult"
            if re.search(r'\bso\s+(important|difficult|easy|good|bad|much|many)', sent_text):
                return False
            if sentence_position < 0.3:
                return True
                
        elif marker_text == 'because':
            # "because" at sentence start is more likely discourse
            if sentence_position < 0.2:
                return True
                
        return False
    
    def filter_modal_verbs(self, span: Span, marker_text: str) -> bool:
        """Filter modal verbs to focus on epistemic/hedging uses."""
        
        marker_lower = marker_text.lower()
        
        if marker_lower not in self.high_frequency_exclusions['modal_verbs_content']:
            return True
        
        sent_text = span.sent.text.lower()
        
        # Exclude deontic/ability modals in clear content contexts
        content_patterns = [
            r'\b(will|would)\s+(go|come|take|give|get|make|do)',  # Future actions
            r'\b(can|could)\s+(see|hear|feel|touch|smell)',       # Ability/perception
            r'\b(should|must)\s+(do|take|give|pay|work)',         # Obligation/necessity
        ]
        
        for pattern in content_patterns:
            if re.search(pattern, sent_text):
                return False
        
        # Keep epistemic uses (uncertainty, possibility)
        epistemic_patterns = [
            r'\b(may|might|could|would)\s+(be|seem|appear|suggest|indicate)',
            r'\b(will|would)\s+(likely|probably|possibly|perhaps)',
            r'\bit\s+(may|might|could|would)\s+be\s+that'
        ]
        
        for pattern in epistemic_patterns:
            if re.search(pattern, sent_text):
                return True
        
        return True  # Default: keep for now, but could be stricter
    
    def apply_confidence_threshold(self, markers: Dict, min_confidence: float = 0.85) -> Dict:
        """Apply higher confidence threshold to reduce false positives."""
        
        filtered_markers = {}
        
        for category, marker_list in markers.items():
            filtered_list = []
            
            for marker in marker_list:
                if isinstance(marker, dict):
                    confidence = marker.get('confidence', 0.0)
                    
                    # Apply stricter confidence for problematic categories
                    if category in ['interactional_engagement_markers', 'interactional_self_mentions']:
                        threshold = max(min_confidence, 0.9)
                    elif category == 'interactive_transitions':
                        threshold = max(min_confidence, 0.85)
                    else:
                        threshold = min_confidence
                    
                    if confidence >= threshold:
                        filtered_list.append(marker)
            
            if filtered_list:
                filtered_markers[category] = filtered_list
        
        return filtered_markers
    
    def apply_frequency_caps(self, markers: Dict, doc: Doc) -> Dict:
        """Apply frequency caps to prevent over-detection of common words."""
        
        word_count = len([token for token in doc if not token.is_space])
        
        # Frequency caps per 1000 words
        frequency_caps = {
            'interactional_engagement_markers': min(15, word_count // 50),  # Max ~15 per 1000 words
            'interactional_self_mentions': min(12, word_count // 60),       # Max ~12 per 1000 words
            'interactive_transitions': min(25, word_count // 30),           # Max ~25 per 1000 words
        }
        
        capped_markers = {}
        
        for category, marker_list in markers.items():
            if category in frequency_caps:
                cap = frequency_caps[category]
                if len(marker_list) > cap:
                    # Keep highest confidence markers
                    sorted_markers = sorted(marker_list, 
                                          key=lambda x: x.get('confidence', 0), 
                                          reverse=True)
                    capped_markers[category] = sorted_markers[:cap]
                else:
                    capped_markers[category] = marker_list
            else:
                capped_markers[category] = marker_list
        
        return capped_markers


class EnhancedMetadiscourseProcessor:
    """Enhanced processor with strict filtering to reduce over-detection."""
    
    def __init__(self, base_processor):
        self.base_processor = base_processor
        self.filter = MetadiscourseFilter()
    
    def process_with_filtering(self, text: str, text_id: str = None) -> Dict:
        """Process text with enhanced filtering to reduce false positives."""
        
        # Get base results
        base_results = self.base_processor.process_text_enhanced(text, text_id)
        
        if 'markers' not in base_results or not base_results['markers']:
            return base_results
        
        # Apply filters
        filtered_markers = self._apply_contextual_filters(
            base_results['markers'], 
            self.base_processor.nlp(text)
        )
        
        # Apply confidence threshold
        filtered_markers = self.filter.apply_confidence_threshold(filtered_markers, 0.85)
        
        # Apply frequency caps
        filtered_markers = self.filter.apply_frequency_caps(
            filtered_markers, 
            self.base_processor.nlp(text)
        )
        
        # Recalculate statistics
        base_results['markers'] = filtered_markers
        base_results['statistics'] = self._recalculate_stats(filtered_markers, text)
        base_results['filtering_info'] = {
            'filters_applied': True,
            'confidence_threshold': 0.85,
            'frequency_caps_applied': True
        }
        
        return base_results
    
    def _apply_contextual_filters(self, markers: Dict, doc: Doc) -> Dict:
        """Apply contextual filters to remove false positives."""
        
        filtered_markers = {}
        
        for category, marker_list in markers.items():
            filtered_list = []
            
            for marker in marker_list:
                if not isinstance(marker, dict):
                    continue
                
                marker_text = marker.get('text', '').lower()
                start = marker.get('start', 0)
                end = marker.get('end', 0)
                
                if start < len(doc) and end <= len(doc):
                    span = doc[start:end]
                    
                    # Apply category-specific filters
                    keep_marker = True
                    
                    if category == 'interactional_engagement_markers':
                        keep_marker = self.filter.filter_pronouns(span, marker_text, category)
                    elif category == 'interactional_self_mentions':
                        keep_marker = self.filter.filter_pronouns(span, marker_text, category)
                    elif category == 'interactive_transitions':
                        keep_marker = self.filter.filter_transitions(span, marker_text)
                    elif 'modal' in category or marker_text in ['will', 'would', 'can', 'could', 'may', 'might']:
                        keep_marker = self.filter.filter_modal_verbs(span, marker_text)
                    
                    if keep_marker:
                        filtered_list.append(marker)
            
            if filtered_list:
                filtered_markers[category] = filtered_list
        
        return filtered_markers
    
    def _recalculate_stats(self, markers: Dict, text: str) -> Dict:
        """Recalculate statistics after filtering."""
        
        word_count = len(text.split())
        stats = {}
        
        # Calculate frequencies per category
        for category, marker_list in markers.items():
            category_count = len(marker_list)
            stats[f'{category}_count'] = category_count
            stats[f'{category}_frequency'] = (category_count / word_count * 1000) if word_count > 0 else 0
        
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