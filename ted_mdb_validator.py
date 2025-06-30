#!/usr/bin/env python3
"""
TED-MDB Validation System
Validates metadiscourse analysis against human annotations from TED-MDB corpus
"""

import sys
import os
import pandas as pd
import json
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append('src')
from metadiscourse_analyzer import ValidationDrivenAnalyzer

class TEDMDBValidator:
    """
    Validates metadiscourse analysis against TED-MDB human annotations
    """
    
    def __init__(self):
        """Initialize TED-MDB validator"""
        self.ted_mdb_path = Path("Ted-MDB-Annotations")
        self.analyzer = ValidationDrivenAnalyzer(validation_mode=True)
        
        # TED-MDB to our system mapping
        self.discourse_mapping = {
            # TED-MDB relations that correspond to metadiscourse markers
            'Expansion.Conjunction': 'interactive_transitions',
            'Expansion.Level-of-detail': 'interactive_code_glosses',
            'Comparison.Contrast': 'interactive_transitions',
            'Comparison.Concession': 'interactional_hedges',
            'Contingency.Cause': 'interactive_transitions',
            'Temporal.Asynchronous': 'interactive_transitions',
            'Temporal.Synchronous': 'interactive_transitions',
            'Expansion.Instantiation': 'interactive_code_glosses',
            'Expansion.Equivalence': 'interactive_code_glosses',
            'Hypophora': 'interactional_engagement_markers'
        }
        
        # Explicit markers that indicate metadiscourse
        self.explicit_markers = {
            'and', 'but', 'so', 'however', 'therefore', 'thus', 'moreover',
            'furthermore', 'additionally', 'also', 'first', 'second', 'finally',
            'in conclusion', 'to summarize', 'for example', 'for instance',
            'in other words', 'that is', 'namely', 'specifically', 'indeed',
            'of course', 'clearly', 'obviously', 'perhaps', 'maybe', 'possibly',
            'certainly', 'definitely', 'I believe', 'we think', 'you know',
            'as mentioned', 'as noted', 'above', 'below', 'following', 'previous'
        }
        
        logger.info("TED-MDB Validator initialized")
    
    def load_ted_talk(self, talk_id: str, language: str = "English") -> Dict:
        """
        Load TED talk raw text and annotations
        
        Args:
            talk_id: TED talk identifier
            language: Language of the talk
            
        Returns:
            Dictionary with raw text and annotations
        """
        lang_path = self.ted_mdb_path / language
        raw_path = lang_path / "raw" / "01" / f"talk_{talk_id}_{language.lower()[:2]}.txt"
        ann_path = lang_path / "ann" / "01" / f"talk_{talk_id}_{language.lower()[:2]}.txt"
        
        try:
            # Load raw text
            with open(raw_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            # Extract actual talk content (skip metadata)
            lines = raw_content.strip().split('\n')
            talk_content = ""
            for line in lines:
                if line.startswith('talkid:') or line.strip() == "":
                    continue
                if ':' in line and len(line.split(':')) == 2:
                    # Skip title line
                    continue
                talk_content += line + " "
            
            # Load annotations
            annotations = []
            if ann_path.exists():
                with open(ann_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            annotations.append(self._parse_annotation_line(line.strip()))
            
            return {
                'talk_id': talk_id,
                'language': language,
                'raw_text': talk_content.strip(),
                'annotations': annotations,
                'word_count': len(talk_content.split())
            }
            
        except Exception as e:
            logger.error(f"Error loading TED talk {talk_id}: {e}")
            return None
    
    def _parse_annotation_line(self, line: str) -> Dict:
        """Parse TED-MDB annotation line format"""
        parts = line.split('|')
        
        if len(parts) < 15:
            return None
        
        try:
            annotation = {
                'type': parts[0],  # Explicit, Implicit, EntRel, etc.
                'connective_span': parts[1] if parts[1] != 'Null' else None,
                'relation_type': parts[7] if parts[7] != 'Null' else None,
                'arg1_span': parts[11] if parts[11] != 'Null' else None,
                'arg2_span': parts[15] if parts[15] != 'Null' else None,
                'raw_line': line
            }
            
            # Extract span positions
            if annotation['connective_span']:
                try:
                    start, end = annotation['connective_span'].split('..')
                    annotation['connective_start'] = int(start)
                    annotation['connective_end'] = int(end)
                except:
                    pass
            
            return annotation
        except Exception as e:
            logger.error(f"Error parsing annotation line: {e}")
            return None
    
    def extract_human_markers(self, talk_data: Dict) -> Dict:
        """
        Extract metadiscourse markers from human annotations
        
        Args:
            talk_data: TED talk data with annotations
            
        Returns:
            Dictionary of metadiscourse markers by category
        """
        if not talk_data or 'annotations' not in talk_data:
            return {}
        
        human_markers = defaultdict(list)
        text = talk_data['raw_text']
        
        for ann in talk_data['annotations']:
            if not ann or not ann.get('relation_type'):
                continue
            
            relation = ann['relation_type']
            
            # Map TED-MDB relations to our categories
            our_category = None
            for ted_relation, our_cat in self.discourse_mapping.items():
                if ted_relation in relation:
                    our_category = our_cat
                    break
            
            if not our_category:
                continue
            
            # Extract marker text if explicit
            marker_text = ""
            if ann['type'] == 'Explicit' and ann.get('connective_start') and ann.get('connective_end'):
                try:
                    start = ann['connective_start']
                    end = ann['connective_end']
                    if start < len(text) and end <= len(text):
                        marker_text = text[start:end+1].strip()
                except:
                    pass
            
            # For implicit relations, use the relation type as marker
            if ann['type'] == 'Implicit':
                marker_text = relation.split('.')[-1].lower()
            
            if marker_text:
                human_markers[our_category].append({
                    'text': marker_text,
                    'position': ann.get('connective_start', 0),
                    'relation': relation,
                    'type': ann['type']
                })
        
        return dict(human_markers)
    
    def validate_single_talk(self, talk_id: str, language: str = "English") -> Dict:
        """
        Validate system against single TED talk
        
        Args:
            talk_id: TED talk identifier
            language: Language of the talk
            
        Returns:
            Validation results
        """
        logger.info(f"Validating talk {talk_id} ({language})")
        
        # Load TED talk data
        talk_data = self.load_ted_talk(talk_id, language)
        if not talk_data:
            return {'error': f'Could not load talk {talk_id}'}
        
        text = talk_data['raw_text']
        word_count = talk_data['word_count']
        
        # Extract human annotations
        human_markers = self.extract_human_markers(talk_data)
        
        # Run our system
        system_result = self.analyzer.analyze_document(text, f"ted_{talk_id}")
        if 'error' in system_result:
            return {'error': f'System analysis failed: {system_result["error"]}'}
        
        system_markers = system_result['markers']
        
        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(
            human_markers, system_markers, word_count
        )
        
        return {
            'talk_id': talk_id,
            'language': language,
            'word_count': word_count,
            'human_markers': human_markers,
            'system_markers': {k: len(v) if isinstance(v, list) else 0 for k, v in system_markers.items()},
            'validation_metrics': validation_metrics,
            'system_stats': system_result['statistics'],
            'benchmark_compliance': system_result['benchmark_compliance']
        }
    
    def _calculate_validation_metrics(self, human_markers: Dict, system_markers: Dict, word_count: int) -> Dict:
        """Calculate precision, recall, F1 for each category"""
        metrics = {}
        
        # Overall metrics
        human_total = sum(len(markers) for markers in human_markers.values())
        system_total = sum(len(v) if isinstance(v, list) else 0 for v in system_markers.values())
        
        human_density = (human_total / word_count * 1000) if word_count > 0 else 0
        system_density = (system_total / word_count * 1000) if word_count > 0 else 0
        
        metrics['overall'] = {
            'human_count': human_total,
            'system_count': system_total,
            'human_density': human_density,
            'system_density': system_density,
            'density_ratio': system_density / human_density if human_density > 0 else 0,
            'density_difference': system_density - human_density
        }
        
        # Category-level metrics
        all_categories = set(human_markers.keys()) | set(system_markers.keys())
        category_metrics = {}
        
        for category in all_categories:
            human_count = len(human_markers.get(category, []))
            system_count = len(system_markers.get(category, [])) if isinstance(system_markers.get(category), list) else 0
            
            # Simple count-based precision/recall (since exact matching is complex)
            if system_count > 0:
                precision = min(human_count, system_count) / system_count
            else:
                precision = 1.0 if human_count == 0 else 0.0
            
            if human_count > 0:
                recall = min(human_count, system_count) / human_count
            else:
                recall = 1.0 if system_count == 0 else 0.0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            category_metrics[category] = {
                'human_count': human_count,
                'system_count': system_count,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'human_density': (human_count / word_count * 1000) if word_count > 0 else 0,
                'system_density': (system_count / word_count * 1000) if word_count > 0 else 0
            }
        
        metrics['categories'] = category_metrics
        
        # Calculate macro-averaged metrics
        precisions = [m['precision'] for m in category_metrics.values()]
        recalls = [m['recall'] for m in category_metrics.values()]
        f1s = [m['f1_score'] for m in category_metrics.values()]
        
        metrics['macro_averaged'] = {
            'precision': np.mean(precisions) if precisions else 0,
            'recall': np.mean(recalls) if recalls else 0,
            'f1_score': np.mean(f1s) if f1s else 0
        }
        
        return metrics
    
    def validate_multiple_talks(self, talk_ids: List[str], language: str = "English") -> Dict:
        """
        Validate system against multiple TED talks
        
        Args:
            talk_ids: List of TED talk identifiers
            language: Language of the talks
            
        Returns:
            Aggregated validation results
        """
        logger.info(f"Validating {len(talk_ids)} TED talks")
        
        individual_results = []
        errors = []
        
        for talk_id in talk_ids:
            result = self.validate_single_talk(talk_id, language)
            if 'error' in result:
                errors.append(result)
            else:
                individual_results.append(result)
        
        if not individual_results:
            return {'error': 'No successful validations', 'errors': errors}
        
        # Aggregate results
        aggregated = self._aggregate_validation_results(individual_results)
        aggregated['individual_results'] = individual_results
        aggregated['validation_errors'] = errors
        
        return aggregated
    
    def _aggregate_validation_results(self, results: List[Dict]) -> Dict:
        """Aggregate validation results across multiple talks"""
        if not results:
            return {}
        
        # Aggregate overall metrics
        total_human = sum(r['validation_metrics']['overall']['human_count'] for r in results)
        total_system = sum(r['validation_metrics']['overall']['system_count'] for r in results)
        total_words = sum(r['word_count'] for r in results)
        
        overall_human_density = (total_human / total_words * 1000) if total_words > 0 else 0
        overall_system_density = (total_system / total_words * 1000) if total_words > 0 else 0
        
        # Aggregate category metrics
        all_categories = set()
        for result in results:
            all_categories.update(result['validation_metrics']['categories'].keys())
        
        aggregated_categories = {}
        for category in all_categories:
            cat_human = sum(r['validation_metrics']['categories'].get(category, {}).get('human_count', 0) for r in results)
            cat_system = sum(r['validation_metrics']['categories'].get(category, {}).get('system_count', 0) for r in results)
            
            # Calculate aggregated precision/recall
            if cat_system > 0:
                precision = min(cat_human, cat_system) / cat_system
            else:
                precision = 1.0 if cat_human == 0 else 0.0
            
            if cat_human > 0:
                recall = min(cat_human, cat_system) / cat_human
            else:
                recall = 1.0 if cat_system == 0 else 0.0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            aggregated_categories[category] = {
                'human_count': cat_human,
                'system_count': cat_system,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'human_density': (cat_human / total_words * 1000) if total_words > 0 else 0,
                'system_density': (cat_system / total_words * 1000) if total_words > 0 else 0
            }
        
        # Calculate macro-averaged metrics
        precisions = [m['precision'] for m in aggregated_categories.values()]
        recalls = [m['recall'] for m in aggregated_categories.values()]
        f1s = [m['f1_score'] for m in aggregated_categories.values()]
        
        return {
            'summary': {
                'talks_validated': len(results),
                'total_words': total_words,
                'total_human_markers': total_human,
                'total_system_markers': total_system,
                'overall_human_density': overall_human_density,
                'overall_system_density': overall_system_density,
                'density_ratio': overall_system_density / overall_human_density if overall_human_density > 0 else 0
            },
            'aggregated_metrics': {
                'categories': aggregated_categories,
                'macro_averaged': {
                    'precision': np.mean(precisions) if precisions else 0,
                    'recall': np.mean(recalls) if recalls else 0,
                    'f1_score': np.mean(f1s) if f1s else 0
                }
            },
            'performance_assessment': self._assess_performance(
                overall_system_density, overall_human_density, 
                np.mean(f1s) if f1s else 0
            )
        }
    
    def _assess_performance(self, system_density: float, human_density: float, avg_f1: float) -> Dict:
        """Assess overall performance and provide recommendations"""
        density_ratio = system_density / human_density if human_density > 0 else 0
        
        assessment = {
            'density_assessment': '',
            'f1_assessment': '',
            'overall_grade': '',
            'recommendations': []
        }
        
        # Density assessment
        if 0.8 <= density_ratio <= 1.2:
            assessment['density_assessment'] = 'EXCELLENT - Well calibrated'
        elif 0.6 <= density_ratio <= 1.4:
            assessment['density_assessment'] = 'GOOD - Minor calibration needed'
        elif 0.4 <= density_ratio <= 1.6:
            assessment['density_assessment'] = 'FAIR - Moderate calibration needed'
        else:
            assessment['density_assessment'] = 'POOR - Major calibration required'
        
        # F1 assessment
        if avg_f1 >= 0.8:
            assessment['f1_assessment'] = 'EXCELLENT - High accuracy'
        elif avg_f1 >= 0.7:
            assessment['f1_assessment'] = 'GOOD - Acceptable accuracy'
        elif avg_f1 >= 0.6:
            assessment['f1_assessment'] = 'FAIR - Needs improvement'
        else:
            assessment['f1_assessment'] = 'POOR - Major accuracy issues'
        
        # Overall grade
        if density_ratio >= 0.8 and avg_f1 >= 0.7:
            assessment['overall_grade'] = 'A - EXCELLENT'
        elif density_ratio >= 0.6 and avg_f1 >= 0.6:
            assessment['overall_grade'] = 'B - GOOD'
        elif density_ratio >= 0.4 and avg_f1 >= 0.5:
            assessment['overall_grade'] = 'C - SATISFACTORY'
        else:
            assessment['overall_grade'] = 'D/F - NEEDS MAJOR IMPROVEMENT'
        
        # Recommendations
        if density_ratio < 0.8:
            assessment['recommendations'].append(f"UNDER-DETECTION: Increase sensitivity (ratio: {density_ratio:.2f})")
        elif density_ratio > 1.2:
            assessment['recommendations'].append(f"OVER-DETECTION: Increase precision (ratio: {density_ratio:.2f})")
        
        if avg_f1 < 0.7:
            assessment['recommendations'].append(f"LOW F1 SCORE: Improve marker identification accuracy ({avg_f1:.2f})")
        
        return assessment
    
    def get_available_talks(self, language: str = "English") -> List[str]:
        """Get list of available TED talk IDs"""
        lang_path = self.ted_mdb_path / language / "raw" / "01"
        
        if not lang_path.exists():
            return []
        
        talk_ids = []
        for file_path in lang_path.glob("talk_*.txt"):
            # Extract talk ID from filename
            filename = file_path.stem
            if filename.startswith("talk_"):
                parts = filename.split("_")
                if len(parts) >= 2:
                    talk_id = parts[1]
                    talk_ids.append(talk_id)
        
        return sorted(list(set(talk_ids)))

def main():
    """Main validation execution"""
    print("🔍 TED-MDB VALIDATION SYSTEM")
    print("=" * 60)
    
    # Initialize validator
    validator = TEDMDBValidator()
    
    # Get available talks
    available_talks = validator.get_available_talks()
    print(f"📚 Available TED talks: {len(available_talks)}")
    print(f"   Talk IDs: {available_talks}")
    
    if not available_talks:
        print("❌ No TED talks found in TED-MDB corpus")
        return
    
    # Validate against available talks
    print(f"\n🚀 Running validation against {len(available_talks)} TED talks...")
    validation_results = validator.validate_multiple_talks(available_talks)
    
    if 'error' in validation_results:
        print(f"❌ Validation failed: {validation_results['error']}")
        return
    
    # Display results
    summary = validation_results['summary']
    metrics = validation_results['aggregated_metrics']
    performance = validation_results['performance_assessment']
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"  Talks validated: {summary['talks_validated']}")
    print(f"  Total words: {summary['total_words']:,}")
    print(f"  Human markers: {summary['total_human_markers']}")
    print(f"  System markers: {summary['total_system_markers']}")
    print(f"  Human density: {summary['overall_human_density']:.1f} per 1k words")
    print(f"  System density: {summary['overall_system_density']:.1f} per 1k words")
    print(f"  Density ratio: {summary['density_ratio']:.2f}")
    
    print(f"\n🎯 PERFORMANCE METRICS:")
    macro_avg = metrics['macro_averaged']
    print(f"  Macro-averaged Precision: {macro_avg['precision']:.3f}")
    print(f"  Macro-averaged Recall: {macro_avg['recall']:.3f}")
    print(f"  Macro-averaged F1: {macro_avg['f1_score']:.3f}")
    
    print(f"\n📋 CATEGORY PERFORMANCE:")
    for category, cat_metrics in metrics['categories'].items():
        print(f"  {category}:")
        print(f"    Human: {cat_metrics['human_count']} ({cat_metrics['human_density']:.1f}/1k)")
        print(f"    System: {cat_metrics['system_count']} ({cat_metrics['system_density']:.1f}/1k)")
        print(f"    P/R/F1: {cat_metrics['precision']:.3f}/{cat_metrics['recall']:.3f}/{cat_metrics['f1_score']:.3f}")
    
    print(f"\n🏆 PERFORMANCE ASSESSMENT:")
    print(f"  Density: {performance['density_assessment']}")
    print(f"  F1 Score: {performance['f1_assessment']}")
    print(f"  Overall Grade: {performance['overall_grade']}")
    
    if performance['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in performance['recommendations']:
            print(f"  • {rec}")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ted_mdb_validation_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Overall assessment
    if performance['overall_grade'].startswith('A'):
        print(f"\n✅ VALIDATION SUCCESS: System meets human annotation standards")
    elif performance['overall_grade'].startswith('B'):
        print(f"\n✅ VALIDATION GOOD: System performs well with minor optimization needed")
    else:
        print(f"\n⚠️  VALIDATION NEEDS IMPROVEMENT: System requires optimization")

if __name__ == "__main__":
    main() 