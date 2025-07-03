"""
Enhanced Metadiscourse Analyzer with ML Classification
Integrates advanced NLP features with trained classifier
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ..ml.classifier import MetadiscourseClassifier

@dataclass
class EnhancedMarker:
    """Enhanced marker with ML confidence and linguistic features"""
    text: str
    category: str
    start_pos: int
    end_pos: int
    context: str
    confidence: float
    ml_prediction: bool
    ml_confidence: float
    linguistic_features: Dict[str, Any]
    validation_reason: str

class EnhancedMetadiscourseAnalyzer:
    """
    Enhanced analyzer combining rule-based patterns with ML classification
    """
    
    def __init__(self, model_path: str = "metadiscourse_model_balanced_5k.joblib"):
        """
        Initialize enhanced analyzer
        
        Args:
            model_path: Path to trained ML model
        """
        # Load trained ML classifier
        print("Loading trained ML classifier...")
        self.ml_classifier = MetadiscourseClassifier()
        try:
            self.ml_classifier.load_model(model_path)
            print(f"✓ ML model loaded successfully from {model_path}")
        except FileNotFoundError:
            print(f"⚠ Model file not found: {model_path}")
            print("Please train a model first using train_optimized_model.py")
            self.ml_classifier = None
        
        # Initialize pattern-based detection (fallback)
        self.patterns = self._load_patterns()
        
        # Category mappings
        self.category_keywords = {
            'transitions': ['however', 'therefore', 'moreover', 'furthermore', 'consequently', 'in contrast', 'on the other hand'],
            'frame_markers': ['first', 'second', 'finally', 'in conclusion', 'to summarize', 'next', 'section'],
            'evidentials': ['according to', 'demonstrate', 'show', 'indicate', 'suggest', 'report', 'find'],
            'code_glosses': ['namely', 'specifically', 'in other words', 'that is', 'such as', 'for example'],
            'engagement_markers': ['note that', 'consider', 'see', 'observe', 'you', 'we should'],
            'self_mentions': ['i', 'we', 'our', 'my', 'the author', 'this study'],
            'boosters': ['clearly', 'obviously', 'definitely', 'certainly', 'undoubtedly', 'strongly'],
            'hedges': ['might', 'could', 'perhaps', 'possibly', 'seem', 'appear', 'suggest']
        }
    
    def analyze_text(self, text: str, use_ml: bool = True, confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Analyze text for metadiscourse markers using enhanced approach
        
        Args:
            text: Input text to analyze
            use_ml: Whether to use ML classification
            confidence_threshold: Minimum confidence for ML predictions
            
        Returns:
            Dictionary with analysis results
        """
        # Find potential markers using pattern matching
        potential_markers = self._find_potential_markers(text)
        
        enhanced_markers = []
        
        for marker_info in potential_markers:
            # Get ML prediction if available
            ml_prediction = None
            ml_confidence = 0.0
            
            if use_ml and self.ml_classifier and self.ml_classifier.is_trained:
                try:
                    ml_results = self.ml_classifier.predict([text], [marker_info['text']])
                    if ml_results:
                        ml_prediction = ml_results[0]['is_metadiscourse']
                        ml_confidence = ml_results[0]['metadiscourse_probability']
                except Exception as e:
                    print(f"ML prediction error for '{marker_info['text']}': {e}")
                    ml_prediction = None
                    ml_confidence = 0.0
            
            # Enhanced validation combining ML and rules
            final_prediction, final_confidence, reason = self._enhanced_validation(
                marker_info, ml_prediction, ml_confidence, confidence_threshold
            )
            
            if final_prediction:
                enhanced_marker = EnhancedMarker(
                    text=marker_info['text'],
                    category=marker_info['category'],
                    start_pos=marker_info['start'],
                    end_pos=marker_info['end'],
                    context=marker_info['context'],
                    confidence=final_confidence,
                    ml_prediction=ml_prediction if ml_prediction is not None else False,
                    ml_confidence=ml_confidence,
                    linguistic_features=marker_info.get('features', {}),
                    validation_reason=reason
                )
                enhanced_markers.append(enhanced_marker)
        
        # Generate analysis summary
        summary = self._generate_summary(enhanced_markers, text)
        
        return {
            'text': text,
            'markers': enhanced_markers,
            'summary': summary,
            'analysis_method': 'enhanced_ml' if use_ml else 'pattern_based'
        }
    
    def _find_potential_markers(self, text: str) -> List[Dict[str, Any]]:
        """Find potential markers using pattern matching"""
        potential_markers = []
        
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                # Find all occurrences of the keyword
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                
                for match in re.finditer(pattern, text.lower()):
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Extract context (50 characters before and after)
                    context_start = max(0, start_pos - 50)
                    context_end = min(len(text), end_pos + 50)
                    context = text[context_start:context_end]
                    
                    marker_info = {
                        'text': text[start_pos:end_pos],
                        'category': category,
                        'start': start_pos,
                        'end': end_pos,
                        'context': context
                    }
                    
                    potential_markers.append(marker_info)
        
        return potential_markers
    
    def _enhanced_validation(self, marker_info: Dict[str, Any], 
                           ml_prediction: Optional[bool], 
                           ml_confidence: float, 
                           threshold: float) -> Tuple[bool, float, str]:
        """
        Enhanced validation combining ML and rule-based approaches
        
        Returns:
            Tuple of (is_valid, confidence, reason)
        """
        # If ML is confident enough, trust it
        if ml_prediction is not None and ml_confidence >= threshold:
            return ml_prediction, ml_confidence, f"ML prediction (confidence: {ml_confidence:.3f})"
        
        # Fallback to rule-based validation
        rule_based_result = self._rule_based_validation(marker_info)
        
        if ml_prediction is not None:
            # Combine ML and rules when ML confidence is low
            if ml_prediction == rule_based_result['is_valid']:
                # Agreement between ML and rules
                combined_confidence = (ml_confidence + rule_based_result['confidence']) / 2
                return rule_based_result['is_valid'], combined_confidence, "ML + Rule agreement"
            else:
                # Disagreement - use rules if ML confidence is very low
                if ml_confidence < 0.3:
                    return rule_based_result['is_valid'], rule_based_result['confidence'], "Rule-based (low ML confidence)"
                else:
                    return ml_prediction, ml_confidence * 0.8, "ML prediction (disagreement with rules)"
        
        # No ML prediction available - use rules only
        return rule_based_result['is_valid'], rule_based_result['confidence'], "Rule-based only"
    
    def _rule_based_validation(self, marker_info: Dict[str, Any]) -> Dict[str, Any]:
        """Simple rule-based validation as fallback"""
        # Basic academic context detection
        context = marker_info['context'].lower()
        marker_text = marker_info['text'].lower()
        
        # Check for academic indicators in context
        academic_indicators = [
            'study', 'research', 'analysis', 'investigation', 'paper',
            'findings', 'results', 'data', 'evidence', 'method'
        ]
        
        academic_score = sum(1 for indicator in academic_indicators if indicator in context)
        
        # Check for non-academic indicators
        non_academic_indicators = [
            'went', 'came', 'said', 'told', 'family', 'friend',
            'home', 'store', 'movie', 'game', 'food'
        ]
        
        non_academic_score = sum(1 for indicator in non_academic_indicators if indicator in context)
        
        # Simple scoring
        if academic_score > non_academic_score:
            confidence = min(0.8, 0.4 + academic_score * 0.1)
            return {'is_valid': True, 'confidence': confidence}
        else:
            confidence = min(0.8, 0.3 + non_academic_score * 0.1)
            return {'is_valid': False, 'confidence': confidence}
    
    def _generate_summary(self, markers: List[EnhancedMarker], text: str) -> Dict[str, Any]:
        """Generate analysis summary"""
        if not markers:
            return {
                'total_markers': 0,
                'categories': {},
                'density': 0.0,
                'avg_confidence': 0.0
            }
        
        # Count by category
        category_counts = {}
        total_confidence = 0
        ml_predictions = 0
        
        for marker in markers:
            category_counts[marker.category] = category_counts.get(marker.category, 0) + 1
            total_confidence += marker.confidence
            if marker.ml_prediction:
                ml_predictions += 1
        
        # Calculate metrics
        word_count = len(text.split())
        density = len(markers) / word_count if word_count > 0 else 0
        avg_confidence = total_confidence / len(markers)
        
        return {
            'total_markers': len(markers),
            'categories': category_counts,
            'density': density,
            'avg_confidence': avg_confidence,
            'ml_predictions': ml_predictions,
            'ml_percentage': ml_predictions / len(markers) if markers else 0
        }
    
    def _load_patterns(self) -> Dict[str, Any]:
        """Load pattern configurations (placeholder)"""
        return {}
    
    def export_results(self, results: Dict[str, Any], format: str = 'json') -> str:
        """Export analysis results in specified format"""
        if format == 'json':
            # Convert markers to serializable format
            serializable_results = {
                'text': results['text'],
                'analysis_method': results['analysis_method'],
                'summary': results['summary'],
                'markers': []
            }
            
            for marker in results['markers']:
                marker_dict = {
                    'text': marker.text,
                    'category': marker.category,
                    'start_pos': marker.start_pos,
                    'end_pos': marker.end_pos,
                    'context': marker.context,
                    'confidence': marker.confidence,
                    'ml_prediction': marker.ml_prediction,
                    'ml_confidence': marker.ml_confidence,
                    'validation_reason': marker.validation_reason
                }
                serializable_results['markers'].append(marker_dict)
            
            return json.dumps(serializable_results, indent=2)
        
        elif format == 'csv':
            # Convert to DataFrame for CSV export
            marker_data = []
            for marker in results['markers']:
                marker_data.append({
                    'marker_text': marker.text,
                    'category': marker.category,
                    'start_pos': marker.start_pos,
                    'end_pos': marker.end_pos,
                    'confidence': marker.confidence,
                    'ml_prediction': marker.ml_prediction,
                    'ml_confidence': marker.ml_confidence,
                    'validation_reason': marker.validation_reason
                })
            
            df = pd.DataFrame(marker_data)
            return df.to_csv(index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")

def main():
    """Test the enhanced analyzer"""
    # Initialize analyzer
    analyzer = EnhancedMetadiscourseAnalyzer()
    
    # Test text
    test_text = """
    This study aims to demonstrate the effectiveness of the proposed methodology. 
    However, further research is needed to validate these findings in different contexts.
    The results clearly show a significant improvement in performance. 
    In conclusion, our findings support the initial hypothesis and contribute to the field.
    I went to the store yesterday to buy groceries for dinner.
    """
    
    print("Analyzing test text with enhanced ML approach...")
    results = analyzer.analyze_text(test_text, use_ml=True)
    
    print(f"\nFound {results['summary']['total_markers']} metadiscourse markers:")
    print(f"Average confidence: {results['summary']['avg_confidence']:.3f}")
    print(f"ML predictions: {results['summary']['ml_predictions']}/{results['summary']['total_markers']}")
    
    print("\nDetected markers:")
    for marker in results['markers']:
        print(f"  '{marker.text}' ({marker.category}) - Confidence: {marker.confidence:.3f}")
        print(f"    Reason: {marker.validation_reason}")
    
    # Export results
    json_output = analyzer.export_results(results, 'json')
    with open('enhanced_analysis_results.json', 'w') as f:
        f.write(json_output)
    
    print(f"\nResults exported to 'enhanced_analysis_results.json'")

if __name__ == "__main__":
    main()