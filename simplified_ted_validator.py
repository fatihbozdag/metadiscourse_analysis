#!/usr/bin/env python3
"""
Simplified TED-MDB Validator
Compares metadiscourse marker density and patterns between our system and TED talks
"""

import sys
import pandas as pd
import json
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append('src')
from metadiscourse_analyzer import ValidationDrivenAnalyzer

class SimplifiedTEDValidator:
    """
    Simplified validator that compares marker density patterns
    """
    
    def __init__(self):
        """Initialize validator"""
        self.ted_mdb_path = Path("Ted-MDB-Annotations")
        self.analyzer = ValidationDrivenAnalyzer(validation_mode=True)
        
        # Common discourse markers that should appear in both systems
        self.common_markers = {
            'transitions': ['and', 'but', 'so', 'however', 'therefore', 'thus', 'moreover', 
                          'furthermore', 'also', 'first', 'second', 'finally', 'then', 'next'],
            'hedges': ['perhaps', 'maybe', 'possibly', 'might', 'could', 'would', 'seem', 
                      'appear', 'suggest', 'indicate', 'likely'],
            'boosters': ['clearly', 'obviously', 'certainly', 'definitely', 'indeed', 
                        'undoubtedly', 'surely', 'absolutely'],
            'code_glosses': ['for example', 'for instance', 'such as', 'in other words', 
                           'that is', 'namely', 'specifically', 'i.e.', 'e.g.'],
            'engagement': ['you', 'your', 'we', 'our', 'consider', 'note', 'see', 'look'],
            'self_mentions': ['I', 'my', 'we', 'our', 'us']
        }
        
        logger.info("Simplified TED Validator initialized")
    
    def load_ted_talk(self, talk_id: str, language: str = "English") -> Dict:
        """Load TED talk text"""
        lang_path = self.ted_mdb_path / language
        raw_path = lang_path / "raw" / "01" / f"talk_{talk_id}_{language.lower()[:2]}.txt"
        
        try:
            with open(raw_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            # Extract talk content
            lines = raw_content.strip().split('\n')
            talk_content = ""
            for line in lines:
                if line.startswith('talkid:') or line.strip() == "":
                    continue
                if ':' in line and len(line.split(':')) == 2 and not line.startswith('http'):
                    continue  # Skip title
                talk_content += line + " "
            
            return {
                'talk_id': talk_id,
                'text': talk_content.strip(),
                'word_count': len(talk_content.split())
            }
            
        except Exception as e:
            logger.error(f"Error loading TED talk {talk_id}: {e}")
            return None
    
    def count_discourse_markers(self, text: str) -> Dict:
        """Count discourse markers in text using simple pattern matching"""
        text_lower = text.lower()
        marker_counts = defaultdict(int)
        
        for category, markers in self.common_markers.items():
            for marker in markers:
                # Use word boundaries for better matching
                pattern = r'\b' + re.escape(marker.lower()) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                marker_counts[category] += matches
        
        return dict(marker_counts)
    
    def validate_ted_talk(self, talk_id: str) -> Dict:
        """Validate single TED talk"""
        logger.info(f"Validating TED talk {talk_id}")
        
        # Load TED talk
        talk_data = self.load_ted_talk(talk_id)
        if not talk_data:
            return {'error': f'Could not load talk {talk_id}'}
        
        text = talk_data['text']
        word_count = talk_data['word_count']
        
        # Count markers using simple pattern matching (baseline)
        baseline_markers = self.count_discourse_markers(text)
        baseline_total = sum(baseline_markers.values())
        baseline_density = (baseline_total / word_count * 1000) if word_count > 0 else 0
        
        # Run our system
        system_result = self.analyzer.analyze_document(text, f"ted_{talk_id}")
        if 'error' in system_result:
            return {'error': f'System failed: {system_result["error"]}'}
        
        system_markers = system_result['markers']
        system_total = sum(len(v) if isinstance(v, list) else 0 for v in system_markers.values())
        system_density = (system_total / word_count * 1000) if word_count > 0 else 0
        
        # Calculate validation metrics
        density_ratio = system_density / baseline_density if baseline_density > 0 else 0
        
        return {
            'talk_id': talk_id,
            'word_count': word_count,
            'baseline_markers': baseline_markers,
            'baseline_total': baseline_total,
            'baseline_density': baseline_density,
            'system_total': system_total,
            'system_density': system_density,
            'density_ratio': density_ratio,
            'system_stats': system_result['statistics'],
            'benchmark_compliance': system_result['benchmark_compliance']
        }
    
    def validate_multiple_talks(self, talk_ids: List[str]) -> Dict:
        """Validate multiple TED talks"""
        results = []
        errors = []
        
        for talk_id in talk_ids:
            result = self.validate_ted_talk(talk_id)
            if 'error' in result:
                errors.append(result)
            else:
                results.append(result)
        
        if not results:
            return {'error': 'No successful validations', 'errors': errors}
        
        # Aggregate results
        total_words = sum(r['word_count'] for r in results)
        total_baseline = sum(r['baseline_total'] for r in results)
        total_system = sum(r['system_total'] for r in results)
        
        avg_baseline_density = (total_baseline / total_words * 1000) if total_words > 0 else 0
        avg_system_density = (total_system / total_words * 1000) if total_words > 0 else 0
        avg_density_ratio = avg_system_density / avg_baseline_density if avg_baseline_density > 0 else 0
        
        # Performance assessment
        assessment = self._assess_performance(avg_system_density, avg_baseline_density, avg_density_ratio)
        
        return {
            'summary': {
                'talks_validated': len(results),
                'total_words': total_words,
                'baseline_total': total_baseline,
                'system_total': total_system,
                'baseline_density': avg_baseline_density,
                'system_density': avg_system_density,
                'density_ratio': avg_density_ratio
            },
            'individual_results': results,
            'validation_errors': errors,
            'performance_assessment': assessment
        }
    
    def _assess_performance(self, system_density: float, baseline_density: float, ratio: float) -> Dict:
        """Assess performance against baseline"""
        assessment = {
            'density_comparison': '',
            'ratio_assessment': '',
            'overall_grade': '',
            'recommendations': []
        }
        
        # Compare to research benchmarks (40-75 per 1k words)
        if 40 <= system_density <= 75:
            assessment['density_comparison'] = f'EXCELLENT - Within research benchmarks ({system_density:.1f})'
        elif 25 <= system_density <= 90:
            assessment['density_comparison'] = f'GOOD - Close to benchmarks ({system_density:.1f})'
        elif system_density < 25:
            assessment['density_comparison'] = f'LOW - Under-detection ({system_density:.1f})'
        else:
            assessment['density_comparison'] = f'HIGH - Over-detection ({system_density:.1f})'
        
        # Ratio assessment
        if 0.8 <= ratio <= 1.2:
            assessment['ratio_assessment'] = f'EXCELLENT - Well calibrated vs baseline ({ratio:.2f})'
        elif 0.5 <= ratio <= 1.5:
            assessment['ratio_assessment'] = f'GOOD - Reasonable vs baseline ({ratio:.2f})'
        else:
            assessment['ratio_assessment'] = f'POOR - Poorly calibrated vs baseline ({ratio:.2f})'
        
        # Overall grade
        benchmark_ok = 40 <= system_density <= 75
        ratio_ok = 0.5 <= ratio <= 1.5
        
        if benchmark_ok and ratio_ok:
            assessment['overall_grade'] = 'A - EXCELLENT'
        elif benchmark_ok or ratio_ok:
            assessment['overall_grade'] = 'B - GOOD'
        else:
            assessment['overall_grade'] = 'C/D - NEEDS IMPROVEMENT'
        
        # Recommendations
        if system_density < 40:
            assessment['recommendations'].append(f"UNDER-DETECTION: Increase sensitivity (current: {system_density:.1f}, target: 40-75)")
        elif system_density > 75:
            assessment['recommendations'].append(f"OVER-DETECTION: Increase precision (current: {system_density:.1f}, target: 40-75)")
        
        if ratio < 0.5:
            assessment['recommendations'].append(f"TOO CONSERVATIVE: System detects much less than baseline ({ratio:.2f})")
        elif ratio > 1.5:
            assessment['recommendations'].append(f"TOO AGGRESSIVE: System detects much more than baseline ({ratio:.2f})")
        
        return assessment
    
    def get_available_talks(self) -> List[str]:
        """Get available TED talk IDs"""
        lang_path = self.ted_mdb_path / "English" / "raw" / "01"
        
        if not lang_path.exists():
            return []
        
        talk_ids = []
        for file_path in lang_path.glob("talk_*.txt"):
            filename = file_path.stem
            if filename.startswith("talk_"):
                parts = filename.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    talk_ids.append(parts[1])
        
        return sorted(list(set(talk_ids)))

def main():
    """Main validation execution"""
    print("🔍 SIMPLIFIED TED-MDB VALIDATION")
    print("=" * 50)
    
    # Initialize validator
    validator = SimplifiedTEDValidator()
    
    # Get available talks
    available_talks = validator.get_available_talks()
    print(f"📚 Available TED talks: {len(available_talks)}")
    print(f"   Talk IDs: {available_talks}")
    
    if not available_talks:
        print("❌ No TED talks found")
        return
    
    # Validate talks
    print(f"\n🚀 Running validation...")
    results = validator.validate_multiple_talks(available_talks)
    
    if 'error' in results:
        print(f"❌ Validation failed: {results['error']}")
        return
    
    # Display results
    summary = results['summary']
    assessment = results['performance_assessment']
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"  Talks validated: {summary['talks_validated']}")
    print(f"  Total words: {summary['total_words']:,}")
    print(f"  Baseline markers: {summary['baseline_total']} ({summary['baseline_density']:.1f}/1k)")
    print(f"  System markers: {summary['system_total']} ({summary['system_density']:.1f}/1k)")
    print(f"  Density ratio: {summary['density_ratio']:.2f}")
    
    print(f"\n🎯 PERFORMANCE ASSESSMENT:")
    print(f"  Density: {assessment['density_comparison']}")
    print(f"  Ratio: {assessment['ratio_assessment']}")
    print(f"  Grade: {assessment['overall_grade']}")
    
    if assessment['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in assessment['recommendations']:
            print(f"  • {rec}")
    
    # Show individual results
    print(f"\n📋 INDIVIDUAL TALK RESULTS:")
    for result in results['individual_results']:
        print(f"  Talk {result['talk_id']}: {result['word_count']} words, "
              f"baseline {result['baseline_density']:.1f}/1k, "
              f"system {result['system_density']:.1f}/1k, "
              f"ratio {result['density_ratio']:.2f}")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ted_validation_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Final assessment
    if assessment['overall_grade'].startswith('A'):
        print(f"\n✅ VALIDATION SUCCESS: System well-calibrated against TED talks")
    elif assessment['overall_grade'].startswith('B'):
        print(f"\n✅ VALIDATION GOOD: System performs reasonably well")
    else:
        print(f"\n⚠️  VALIDATION NEEDS IMPROVEMENT: System requires calibration")

if __name__ == "__main__":
    main() 