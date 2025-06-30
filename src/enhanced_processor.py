#!/usr/bin/env python3
"""
Enhanced Metadiscourse Processor with Evidence-Based Optimizations
Implements fixes based on validation results and human annotation analysis
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

from processor import EnhancedTextProcessor
from markers import INTERACTIVE_MARKERS, INTERACTIONAL_MARKERS

class EvidenceBasedProcessor(EnhancedTextProcessor):
    """Enhanced processor with evidence-based optimizations"""
    
    def __init__(self, config_file: str = None):
        super().__init__()
        
        # Load evidence-based configuration
        self.evidence_config = self.load_evidence_config(config_file)
        
        # Enhanced thresholds based on validation
        self.category_thresholds = self.evidence_config.get('thresholds', {
            'Interactive_Transitions': 0.85,
            'Interactional_Self_Mentions': 0.92,  # Higher due to over-detection
            'Interactional_Hedges': 0.80,
            'Interactive_Code_Glosses': 0.85,
            'Interactive_Engagement_Markers': 0.90,  # Higher due to over-detection
            'default': 0.85
        })
        
        # Context filters based on error analysis
        self.context_filters = self.evidence_config.get('context_filters', {})
        
        # Pattern-based exclusion rules
        self.exclusion_patterns = self.compile_exclusion_patterns()
        
        # Statistical limits based on corpus analysis
        self.density_limits = {
            'Interactive_Transitions': 25,  # per 1k words
            'Interactional_Self_Mentions': 12,
            'Interactive_Engagement_Markers': 15,
            'Interactional_Hedges': 20,
            'default': 30
        }
    
    def load_evidence_config(self, config_file: str) -> Dict:
        """Load evidence-based configuration from validation results"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default evidence-based configuration
        return {
            'thresholds': {},
            'context_filters': {},
            'validation_source': 'TED-MDB_analysis',
            'optimization_method': 'precision_recall_optimization'
        }
    
    def compile_exclusion_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for context-based exclusions"""
        patterns = {}
        
        # Self-mention exclusions (avoid narrative/quoted contexts)
        patterns['Interactional_Self_Mentions'] = [
            re.compile(r'"[^"]*\b(I|we|my|our)\b[^"]*"', re.IGNORECASE),  # Quoted speech
            re.compile(r'\b(said|told|asked|replied)\s+[^.]*\b(I|we)\b', re.IGNORECASE),  # Reported speech
            re.compile(r'\b(I|we)\s+(went|came|saw|did|was|were)\b', re.IGNORECASE),  # Narrative past
            re.compile(r'\bonce\s+I\b', re.IGNORECASE),  # Narrative markers
            re.compile(r'\byesterday\s+(I|we)\b', re.IGNORECASE),  # Temporal narrative
        ]
        
        # Engagement marker exclusions (avoid non-metadiscourse you/your)
        patterns['Interactive_Engagement_Markers'] = [
            re.compile(r'\byou\s+(are|were|have|had|will|can|should|must)\s+\w+ing\b', re.IGNORECASE),  # Content actions
            re.compile(r'\byour\s+(name|age|job|work|family|house|car)\b', re.IGNORECASE),  # Personal content
            re.compile(r'\bif\s+you\s+(want|need|like|prefer)\b', re.IGNORECASE),  # Conditional content
            re.compile(r'\byou\s+(go|come|see|do|get|take|make)\b', re.IGNORECASE),  # Action verbs
        ]
        
        # Transition exclusions (avoid temporal/causal non-metadiscourse)
        patterns['Interactive_Transitions'] = [
            re.compile(r'\b(then|next|after|before)\s+\w+\s+(went|came|did|was)\b', re.IGNORECASE),  # Narrative sequence
            re.compile(r'\bso\s+(I|we|he|she|they)\s+(went|came|did|was)\b', re.IGNORECASE),  # Causal narrative
            re.compile(r'\bbecause\s+(I|we|he|she|they)\s+(wanted|needed|had to)\b', re.IGNORECASE),  # Personal causation
        ]
        
        return patterns
    
    def apply_context_filtering(self, marker: Dict, full_text: str, sentence: str) -> bool:
        """Apply context-based filtering to reduce false positives"""
        category = marker.get('category', '')
        marker_text = marker.get('text', '')
        
        # Check exclusion patterns
        if category in self.exclusion_patterns:
            for pattern in self.exclusion_patterns[category]:
                if pattern.search(sentence):
                    return False  # Exclude this marker
        
        # Position-based filtering
        if not self.check_position_validity(marker, sentence):
            return False
        
        # Confidence-based filtering with category-specific thresholds
        threshold = self.category_thresholds.get(category, self.category_thresholds['default'])
        if marker.get('confidence', 0.8) < threshold:
            return False
        
        # Context coherence check
        if not self.check_context_coherence(marker, sentence):
            return False
        
        return True
    
    def check_position_validity(self, marker: Dict, sentence: str) -> bool:
        """Check if marker position suggests metadiscourse usage"""
        marker_text = marker.get('text', '').lower()
        category = marker.get('category', '')
        
        # Sentence-initial transitions are more likely metadiscourse
        if 'Transitions' in category:
            sentence_start = sentence.strip()[:20].lower()
            if sentence_start.startswith(marker_text):
                return True
            # Mid-sentence transitions need stronger context
            if marker_text in ['so', 'but', 'and']:
                # Check for argumentative context
                arg_indicators = ['argument', 'point', 'evidence', 'conclusion', 'therefore', 'thus']
                if any(ind in sentence.lower() for ind in arg_indicators):
                    return True
                return False
        
        # Self-mentions at sentence start more likely metadiscourse
        if 'Self_Mentions' in category:
            sentence_start = sentence.strip()[:10].lower()
            if sentence_start.startswith(marker_text):
                # Check for metadiscourse indicators
                meta_indicators = ['argue', 'claim', 'suggest', 'propose', 'conclude', 'show', 'demonstrate']
                if any(ind in sentence.lower() for ind in meta_indicators):
                    return True
                return False
        
        return True
    
    def check_context_coherence(self, marker: Dict, sentence: str) -> bool:
        """Check if surrounding context supports metadiscourse interpretation"""
        category = marker.get('category', '')
        sentence_lower = sentence.lower()
        
        # Academic/argumentative context indicators
        academic_indicators = [
            'research', 'study', 'analysis', 'evidence', 'data', 'findings',
            'argument', 'theory', 'hypothesis', 'conclusion', 'results',
            'literature', 'scholars', 'academic', 'scientific'
        ]
        
        # Metadiscourse context indicators
        metadiscourse_indicators = [
            'paper', 'article', 'chapter', 'section', 'discussion',
            'above', 'below', 'following', 'previous', 'mentioned',
            'reader', 'audience', 'consider', 'note that', 'observe'
        ]
        
        # Check for academic/metadiscourse context
        context_score = 0
        for indicator in academic_indicators + metadiscourse_indicators:
            if indicator in sentence_lower:
                context_score += 1
        
        # Different thresholds for different categories
        if 'Self_Mentions' in category:
            return context_score >= 1  # Need at least one academic indicator
        elif 'Engagement' in category:
            return context_score >= 1 or 'consider' in sentence_lower or 'note' in sentence_lower
        else:
            return True  # Other categories less strict
    
    def apply_density_limits(self, markers: List[Dict], word_count: int) -> List[Dict]:
        """Apply statistical density limits based on corpus analysis"""
        if word_count == 0:
            return markers
        
        # Group markers by category
        category_markers = defaultdict(list)
        for marker in markers:
            category = marker.get('category', 'unknown')
            category_markers[category].append(marker)
        
        # Apply limits per category
        filtered_markers = []
        
        for category, cat_markers in category_markers.items():
            limit_per_1k = self.density_limits.get(category, self.density_limits['default'])
            max_markers = int((word_count / 1000) * limit_per_1k)
            
            if len(cat_markers) <= max_markers:
                filtered_markers.extend(cat_markers)
            else:
                # Keep highest confidence markers
                sorted_markers = sorted(cat_markers, key=lambda x: x.get('confidence', 0.8), reverse=True)
                filtered_markers.extend(sorted_markers[:max_markers])
                
                # Log the filtering
                print(f"Density filter: {category} reduced from {len(cat_markers)} to {max_markers} markers")
        
        return filtered_markers
    
    def process_document(self, text: str, doc_id: str = "doc") -> Dict:
        """Enhanced document processing with evidence-based optimizations"""
        # First, run base processing
        base_result = super().process_text_enhanced(text, doc_id)
        
        if not base_result.get('success', False):
            return base_result
        
        # Get base markers - handle the nested structure
        base_markers_dict = base_result.get('markers', {})
        base_markers = []
        
        # Flatten the nested marker structure
        if isinstance(base_markers_dict, dict):
            for category, markers in base_markers_dict.items():
                if isinstance(markers, list):
                    for marker in markers:
                        if isinstance(marker, dict):
                            marker['category'] = category
                            base_markers.append(marker)
        
        word_count = base_result.get('word_count', 0)
        
        # Apply evidence-based filtering
        filtered_markers = []
        
        for marker in base_markers:
            # Find the sentence containing this marker
            marker_pos = marker.get('position', 0)
            sentence = self.extract_sentence_at_position(text, marker_pos)
            
            # Apply context filtering
            if self.apply_context_filtering(marker, text, sentence):
                filtered_markers.append(marker)
        
        # Apply density limits
        final_markers = self.apply_density_limits(filtered_markers, word_count)
        
        # Recalculate statistics
        enhanced_stats = self.calculate_enhanced_statistics(final_markers, word_count)
        
        # Create enhanced result
        enhanced_result = {
            'doc_id': doc_id,
            'success': True,
            'markers': final_markers,
            'statistics': enhanced_stats,
            'word_count': word_count,
            'processing_info': {
                'base_markers': len(base_markers),
                'context_filtered': len(base_markers) - len(filtered_markers),
                'density_filtered': len(filtered_markers) - len(final_markers),
                'final_markers': len(final_markers),
                'evidence_based_processing': True,
                'thresholds_applied': self.category_thresholds,
                'filters_applied': list(self.context_filters.keys())
            }
        }
        
        return enhanced_result
    
    def extract_sentence_at_position(self, text: str, position: int) -> str:
        """Extract the sentence containing the given character position"""
        # Simple sentence extraction - could be enhanced with better sentence segmentation
        sentences = re.split(r'[.!?]+', text)
        
        current_pos = 0
        for sentence in sentences:
            if current_pos <= position <= current_pos + len(sentence):
                return sentence.strip()
            current_pos += len(sentence) + 1  # +1 for the delimiter
        
        return ""
    
    def calculate_enhanced_statistics(self, markers: List[Dict], word_count: int) -> Dict:
        """Calculate enhanced statistics with validation metrics"""
        if not markers:
            return {'total_markers': 0, 'density_per_1k': 0, 'categories': {}}
        
        # Basic statistics
        total_markers = len(markers)
        density = (total_markers / word_count * 1000) if word_count > 0 else 0
        
        # Category breakdown
        category_stats = defaultdict(lambda: {'count': 0, 'confidence_scores': []})
        
        for marker in markers:
            category = marker.get('category', 'unknown')
            confidence = marker.get('confidence', 0.8)
            
            category_stats[category]['count'] += 1
            category_stats[category]['confidence_scores'].append(confidence)
        
        # Calculate category-specific metrics
        for category, stats in category_stats.items():
            scores = stats['confidence_scores']
            stats.update({
                'density_per_1k': (stats['count'] / word_count * 1000) if word_count > 0 else 0,
                'avg_confidence': np.mean(scores) if scores else 0,
                'confidence_std': np.std(scores) if scores else 0,
                'high_confidence_ratio': sum(1 for s in scores if s >= 0.9) / len(scores) if scores else 0
            })
            # Remove the raw scores to keep output clean
            del stats['confidence_scores']
        
        return {
            'total_markers': total_markers,
            'density_per_1k': density,
            'avg_confidence': np.mean([m.get('confidence', 0.8) for m in markers]),
            'categories': dict(category_stats),
            'quality_metrics': {
                'within_density_benchmarks': 40 <= density <= 75,
                'high_confidence_markers': sum(1 for m in markers if m.get('confidence', 0.8) >= 0.9),
                'evidence_based_filtering': True
            }
        }

def main():
    """Test the enhanced processor"""
    print("🧪 Testing Enhanced Evidence-Based Processor")
    print("=" * 50)
    
    # Initialize enhanced processor
    processor = EvidenceBasedProcessor()
    
    # Test with sample text
    test_text = """
    In this paper, I argue that metadiscourse markers play a crucial role in academic writing.
    First, let me explain what metadiscourse means. You might think that these markers are unnecessary,
    but I believe they help readers understand the text better. So, we need to study them carefully.
    I went to the library yesterday and found many books about this topic.
    """
    
    # Process document
    result = processor.process_document(test_text, "test_doc")
    
    # Display results
    if result.get('success', True):  # Default to True if success key missing
        stats = result.get('statistics', {})
        processing_info = result.get('processing_info', {})
        
        print(f"📊 PROCESSING RESULTS:")
        print(f"  Total markers: {stats.get('total_markers', 0)}")
        print(f"  Density: {stats.get('density_per_1k', 0):.1f} per 1k words")
        print(f"  Average confidence: {stats.get('avg_confidence', 0):.3f}")
        
        if processing_info:
            print(f"\n🔧 FILTERING APPLIED:")
            print(f"  Base markers: {processing_info.get('base_markers', 0)}")
            print(f"  Context filtered: {processing_info.get('context_filtered', 0)}")
            print(f"  Density filtered: {processing_info.get('density_filtered', 0)}")
            print(f"  Final markers: {processing_info.get('final_markers', 0)}")
        
        categories = stats.get('categories', {})
        if categories:
            print(f"\n📈 CATEGORY BREAKDOWN:")
            for category, cat_stats in categories.items():
                print(f"  {category}: {cat_stats.get('count', 0)} markers (density: {cat_stats.get('density_per_1k', 0):.1f})")
        
        quality = stats.get('quality_metrics', {})
        if quality:
            print(f"\n✅ QUALITY ASSESSMENT:")
            print(f"  Within benchmarks: {quality.get('within_density_benchmarks', False)}")
            print(f"  High confidence markers: {quality.get('high_confidence_markers', 0)}")
        
        # Show raw result if no expected structure
        if not stats and not processing_info:
            print(f"\n📄 RAW RESULT:")
            print(f"  Keys available: {list(result.keys())}")
            if 'markers' in result:
                markers = result['markers']
                if isinstance(markers, dict):
                    total_markers = sum(len(v) if isinstance(v, list) else 0 for v in markers.values())
                    print(f"  Total markers found: {total_markers}")
                    for cat, cat_markers in markers.items():
                        if isinstance(cat_markers, list):
                            print(f"    {cat}: {len(cat_markers)} markers")
        
    else:
        print("❌ Processing failed:", result.get('error', 'Unknown error'))

if __name__ == "__main__":
    main() 