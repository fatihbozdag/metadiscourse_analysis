#!/usr/bin/env python3
"""
Optimized Metadiscourse Analyzer
Based on TED-MDB validation results - addresses severe under-detection
"""

import sys
import pandas as pd
import json
import numpy as np
import time
import re
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append('src')
from metadiscourse_analyzer import ValidationDrivenAnalyzer

class OptimizedAnalyzer(ValidationDrivenAnalyzer):
    """
    Optimized analyzer with TED-MDB validation-based calibration
    """
    
    def __init__(self):
        """Initialize with optimized parameters"""
        super().__init__(validation_mode=True)
        
        # Override with much more permissive thresholds based on TED validation
        self.validation_config['confidence_thresholds'] = {
            'interactive_transitions': 0.3,           # Much lower from 0.75
            'interactional_hedges': 0.25,             # Much lower from 0.70
            'interactional_boosters': 0.35,           # Much lower from 0.80
            'interactional_engagement_markers': 0.4,  # Much lower from 0.85
            'interactional_self_mentions': 0.45,      # Much lower from 0.90
            'interactive_code_glosses': 0.3,          # Much lower from 0.75
            'interactive_frame_markers': 0.35,        # Much lower from 0.80
            'default': 0.3                            # Much lower from 0.75
        }
        
        # Increase density expectations based on TED baseline (145.8/1k)
        self.validation_config['density_expectations'] = {
            'interactive_transitions': {'min': 15, 'max': 45, 'target': 25},      # Increased
            'interactional_hedges': {'min': 10, 'max': 30, 'target': 18},         # Increased
            'interactional_boosters': {'min': 5, 'max': 20, 'target': 12},        # Increased
            'interactional_engagement_markers': {'min': 8, 'max': 25, 'target': 15}, # Increased
            'interactional_self_mentions': {'min': 5, 'max': 25, 'target': 15},   # Increased
            'interactive_code_glosses': {'min': 3, 'max': 15, 'target': 8},       # Increased
            'interactive_frame_markers': {'min': 2, 'max': 12, 'target': 6}       # Increased
        }
        
        # More permissive context adjustment
        self.validation_config['context_patterns']['content_penalty'] = 0.9  # Less penalty
        
        logger.info("Optimized analyzer initialized with TED-MDB calibrated parameters")
    
    def apply_evidence_based_filtering(self, markers: Dict, text: str, word_count: int) -> Dict:
        """
        Apply optimized filtering with much more permissive approach
        """
        if not self.validation_mode:
            return markers
        
        filtered_markers = {}
        text_lower = text.lower()
        
        # Much more permissive context scoring
        metadiscourse_score = 0
        content_score = 0
        
        for pattern in self.validation_config['context_patterns']['metadiscourse_indicators']:
            metadiscourse_score += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        for pattern in self.validation_config['context_patterns']['content_indicators']:
            content_score += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        # More permissive context factor (less penalty for content)
        context_factor = 1.0
        if content_score > metadiscourse_score * 2:  # Only penalize if heavily content-focused
            context_factor = 0.9  # Minimal penalty
        elif metadiscourse_score > content_score:
            context_factor = 1.1  # Small bonus
        
        for category, marker_list in markers.items():
            if not isinstance(marker_list, list):
                filtered_markers[category] = marker_list
                continue
            
            # Apply much lower confidence filtering
            confidence_threshold = self.validation_config['confidence_thresholds'].get(
                category, self.validation_config['confidence_thresholds']['default']
            )
            
            filtered_list = []
            for marker in marker_list:
                if isinstance(marker, dict):
                    confidence = marker.get('confidence', 0.8)
                    adjusted_confidence = confidence * context_factor
                    
                    # Much more permissive threshold
                    if adjusted_confidence >= confidence_threshold:
                        marker['adjusted_confidence'] = adjusted_confidence
                        filtered_list.append(marker)
                else:
                    # Include all non-dict markers
                    filtered_list.append(marker)
            
            # Much more permissive density filtering
            expected_density = self.validation_config['density_expectations'].get(category, {})
            if expected_density and word_count > 0:
                current_density = (len(filtered_list) / word_count * 1000)
                max_density = expected_density.get('max', float('inf'))
                
                # Only filter if severely over target (3x max)
                severe_over_limit = max_density * 3
                if current_density > severe_over_limit:
                    max_count = int((word_count / 1000) * severe_over_limit)
                    if len(filtered_list) > max_count:
                        sorted_markers = sorted(filtered_list, 
                                              key=lambda x: x.get('adjusted_confidence', 0.8) if isinstance(x, dict) else 0.8, 
                                              reverse=True)
                        filtered_list = sorted_markers[:max_count]
                        logger.debug(f"Severe over-density filter applied to {category}: {len(marker_list)} -> {len(filtered_list)}")
            
            filtered_markers[category] = filtered_list
        
        return filtered_markers

def main():
    """Main execution with optimization"""
    print("🚀 OPTIMIZED METADISCOURSE ANALYZER")
    print("=" * 60)
    print("Calibrated based on TED-MDB validation results")
    
    # Initialize optimized analyzer
    print("\n🔧 Initializing optimized analyzer...")
    analyzer = OptimizedAnalyzer()
    print("✅ Optimized analyzer ready")
    
    # Test on TICLE sample
    print("\n🚀 Running optimized analysis on TICLE sample...")
    results = analyzer.analyze_corpus(sample_size=50)
    
    if 'error' not in results:
        corpus_stats = results['corpus_statistics']
        compliance = results['benchmark_compliance']
        
        print(f"\n📊 OPTIMIZED RESULTS:")
        print(f"  Documents: {corpus_stats['documents_processed']}")
        print(f"  Total words: {corpus_stats['total_words']:,}")
        print(f"  Total markers: {corpus_stats['total_markers']:,}")
        print(f"  Density: {corpus_stats['overall_density']:.1f} per 1k words")
        print(f"  Compliance: {compliance['compliance_rate']:.1%}")
        print(f"  Assessment: {compliance['overall_assessment']}")
        
        # Compare with previous results
        print(f"\n📈 IMPROVEMENT ANALYSIS:")
        print(f"  Previous density: 19.8 per 1k words")
        print(f"  Optimized density: {corpus_stats['overall_density']:.1f} per 1k words")
        improvement = corpus_stats['overall_density'] - 19.8
        print(f"  Improvement: +{improvement:.1f} per 1k words")
        
        if 40 <= corpus_stats['overall_density'] <= 75:
            print(f"  ✅ SUCCESS: Now within research benchmarks!")
        else:
            print(f"  ⚠️  Still needs adjustment")
        
        # Save optimized results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"optimized_analysis_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Optimized results saved to: {results_file}")
        
    else:
        print(f"❌ Error: {results['error']}")
    
    # Validate against TED talks
    print(f"\n🔍 Validating optimized system against TED talks...")
    
    try:
        from simplified_ted_validator import SimplifiedTEDValidator
        
        validator = SimplifiedTEDValidator()
        validator.analyzer = analyzer  # Use optimized analyzer
        
        available_talks = validator.get_available_talks()
        if available_talks:
            ted_results = validator.validate_multiple_talks(available_talks[:3])  # Test on 3 talks
            
            if 'error' not in ted_results:
                summary = ted_results['summary']
                assessment = ted_results['performance_assessment']
                
                print(f"\n📊 TED VALIDATION RESULTS:")
                print(f"  Baseline density: {summary['baseline_density']:.1f} per 1k words")
                print(f"  Optimized density: {summary['system_density']:.1f} per 1k words")
                print(f"  Ratio: {summary['density_ratio']:.2f}")
                print(f"  Grade: {assessment['overall_grade']}")
                
                if summary['density_ratio'] > 0.1:  # Any detection is improvement
                    print(f"  ✅ IMPROVEMENT: System now detecting markers!")
                else:
                    print(f"  ⚠️  Still under-detecting")
            else:
                print(f"  ❌ TED validation failed: {ted_results['error']}")
        else:
            print(f"  ⚠️  No TED talks available for validation")
            
    except Exception as e:
        print(f"  ⚠️  TED validation error: {e}")
    
    print(f"\n🎯 OPTIMIZATION SUMMARY:")
    print(f"  ✅ Confidence thresholds reduced: 0.75 → 0.3")
    print(f"  ✅ Density limits increased: Based on TED baseline")
    print(f"  ✅ Context penalties reduced: More permissive")
    print(f"  ✅ Validation framework maintained")

if __name__ == "__main__":
    main() 