#!/usr/bin/env python3
"""
Accuracy Checker for Metadiscourse Analysis System
Validates system performance against research benchmarks and validation config
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccuracyChecker:
    """
    Comprehensive accuracy validation for metadiscourse analysis system
    """
    
    def __init__(self, validation_config_path: str = "validation_config.json"):
        """Initialize accuracy checker with validation configuration"""
        with open(validation_config_path, 'r') as f:
            self.config = json.load(f)
        
        self.benchmarks = self.config['research_benchmarks']
        self.thresholds = self.config['validation_thresholds']
        
    def load_latest_results(self) -> Dict:
        """Load the most recent analysis results"""
        results_files = [
            "final_optimized_results.json",
            "enhanced_analysis_results.json", 
            "calibrated_analysis_results.json",
            "analysis_results.json"
        ]
        
        # Also check results directory
        if os.path.exists("results/"):
            results_dir_files = [f"results/{f}" for f in os.listdir("results/") 
                               if f.endswith('.json')]
            results_files.extend(results_dir_files)
        
        # Find the most recent file
        latest_file = None
        latest_time = 0
        
        for file_path in results_files:
            if os.path.exists(file_path):
                mod_time = os.path.getmtime(file_path)
                if mod_time > latest_time:
                    latest_time = mod_time
                    latest_file = file_path
        
        if not latest_file:
            raise FileNotFoundError("No analysis results files found")
        
        logger.info(f"Loading latest results from: {latest_file}")
        
        # Handle large files by sampling
        file_size = os.path.getsize(latest_file)
        if file_size > 10 * 1024 * 1024:  # > 10MB
            logger.warning(f"Large file detected ({file_size/1024/1024:.1f}MB). Sampling...")
            return self._sample_large_file(latest_file)
        else:
            with open(latest_file, 'r') as f:
                return json.load(f)
    
    def _sample_large_file(self, file_path: str, sample_size: int = 50) -> Dict:
        """Sample from large result files for analysis"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if 'document_results' in data:
            docs = data['document_results']
            if len(docs) > sample_size:
                # Take a representative sample
                sampled_docs = docs[:sample_size]
                logger.info(f"Sampled {len(sampled_docs)} documents from {len(docs)} total")
                data['document_results'] = sampled_docs
                data['sampled'] = True
                data['original_count'] = len(docs)
        
        return data
    
    def analyze_density_compliance(self, results: Dict) -> Dict:
        """Check compliance with research benchmark density (40-75 per 1k words)"""
        target_min = self.benchmarks['overall_density']['min']
        target_max = self.benchmarks['overall_density']['max']
        target_optimal = self.benchmarks['overall_density']['optimal']
        
        if 'overall_stats' in results:
            actual_density = results['overall_stats'].get('markers_per_1k_words', 0)
        elif 'document_results' in results:
            # Calculate from document results
            total_markers = sum(doc.get('total_markers', 0) for doc in results['document_results'])
            total_words = sum(doc.get('word_count', 0) for doc in results['document_results'])
            actual_density = (total_markers / total_words * 1000) if total_words > 0 else 0
        else:
            actual_density = 0
        
        compliance = target_min <= actual_density <= target_max
        distance_from_optimal = abs(actual_density - target_optimal)
        
        # Grade the density
        if compliance:
            if distance_from_optimal <= 5:
                grade = "A (Excellent)"
            elif distance_from_optimal <= 10:
                grade = "B+ (Very Good)"
            else:
                grade = "B (Good)"
        else:
            if actual_density < target_min:
                grade = "C (Under-detection)"
            else:
                grade = "D (Over-detection)"
        
        return {
            'actual_density': actual_density,
            'target_range': [target_min, target_max],
            'target_optimal': target_optimal,
            'compliant': compliance,
            'distance_from_optimal': distance_from_optimal,
            'grade': grade,
            'recommendation': self._get_density_recommendation(actual_density, target_min, target_max)
        }
    
    def analyze_category_distribution(self, results: Dict) -> Dict:
        """Analyze category distribution against research benchmarks"""
        target_dist = self.benchmarks['category_distribution']
        
        # Extract actual distribution
        if 'overall_stats' in results and 'category_counts' in results['overall_stats']:
            actual_counts = results['overall_stats']['category_counts']
            total_markers = sum(actual_counts.values())
            actual_percentages = {cat: (count/total_markers*100) for cat, count in actual_counts.items()}
        elif 'document_results' in results:
            # Calculate from document results
            category_totals = {}
            for doc in results['document_results']:
                for cat, count in doc.get('categories', {}).items():
                    category_totals[cat] = category_totals.get(cat, 0) + count
            
            total_markers = sum(category_totals.values())
            actual_percentages = {cat: (count/total_markers*100) for cat, count in category_totals.items()}
        else:
            actual_percentages = {}
        
        # Map category names
        category_mapping = {
            'transitions': 'interactive_transitions',
            'hedges': 'interactional_hedges', 
            'boosters': 'interactional_boosters',
            'engagement_markers': 'interactional_engagement_markers',
            'self_mentions': 'interactional_self_mentions',
            'code_glosses': 'interactive_code_glosses',
            'frame_markers': 'interactive_frame_markers'
        }
        
        analysis = {}
        total_deviation = 0
        
        for actual_cat, target_cat in category_mapping.items():
            if target_cat in target_dist:
                actual_pct = actual_percentages.get(actual_cat, 0)
                target_pct = target_dist[target_cat]['target_percentage']
                deviation = abs(actual_pct - target_pct)
                total_deviation += deviation
                
                analysis[actual_cat] = {
                    'actual_percentage': actual_pct,
                    'target_percentage': target_pct,
                    'deviation': deviation,
                    'status': 'Good' if deviation <= 5 else 'Needs adjustment'
                }
        
        # Overall grade
        avg_deviation = total_deviation / len(analysis) if analysis else 100
        if avg_deviation <= 3:
            grade = "A (Excellent balance)"
        elif avg_deviation <= 7:
            grade = "B (Good balance)"
        elif avg_deviation <= 12:
            grade = "C (Moderate imbalance)"
        else:
            grade = "D (Poor balance)"
        
        return {
            'category_analysis': analysis,
            'average_deviation': avg_deviation,
            'grade': grade,
            'recommendations': self._get_balance_recommendations(analysis)
        }
    
    def analyze_confidence_scores(self, results: Dict) -> Dict:
        """Analyze confidence score distribution and quality"""
        if 'document_results' not in results:
            return {'error': 'No document results available for confidence analysis'}
        
        confidences = []
        category_confidences = {}
        
        for doc in results['document_results']:
            if 'confidence_score' in doc:
                confidences.append(doc['confidence_score'])
            
            # Extract detailed markers if available
            if 'detailed_markers' in doc:
                for marker in doc['detailed_markers']:
                    conf = marker.get('confidence', 0)
                    cat = marker.get('category', 'unknown')
                    
                    if cat not in category_confidences:
                        category_confidences[cat] = []
                    category_confidences[cat].append(conf)
        
        overall_analysis = {}
        if confidences:
            overall_analysis = {
                'mean_confidence': np.mean(confidences),
                'median_confidence': np.median(confidences),
                'std_confidence': np.std(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences),
                'high_confidence_docs': sum(1 for c in confidences if c >= 0.8) / len(confidences) * 100
            }
        
        category_analysis = {}
        for cat, confs in category_confidences.items():
            if confs:
                category_analysis[cat] = {
                    'mean_confidence': np.mean(confs),
                    'count': len(confs),
                    'threshold_compliance': sum(1 for c in confs if c >= self.thresholds['confidence_levels'].get(cat, 0.75)) / len(confs) * 100
                }
        
        return {
            'overall_confidence': overall_analysis,
            'category_confidence': category_analysis,
            'quality_assessment': self._assess_confidence_quality(overall_analysis)
        }
    
    def _get_density_recommendation(self, actual: float, min_target: float, max_target: float) -> str:
        """Generate density adjustment recommendations"""
        if actual < min_target:
            deficit = min_target - actual
            return f"Increase detection by {deficit:.1f} markers/1k words. Lower confidence thresholds or expand patterns."
        elif actual > max_target:
            excess = actual - max_target
            return f"Reduce detection by {excess:.1f} markers/1k words. Raise confidence thresholds or add filters."
        else:
            return "Density is within research benchmarks. Maintain current settings."
    
    def _get_balance_recommendations(self, analysis: Dict) -> List[str]:
        """Generate category balance recommendations"""
        recommendations = []
        
        for cat, data in analysis.items():
            deviation = data['deviation']
            if deviation > 5:
                if data['actual_percentage'] > data['target_percentage']:
                    recommendations.append(f"Reduce {cat} detection (currently {deviation:.1f}% above target)")
                else:
                    recommendations.append(f"Increase {cat} detection (currently {deviation:.1f}% below target)")
        
        return recommendations
    
    def _assess_confidence_quality(self, confidence_data: Dict) -> str:
        """Assess overall confidence quality"""
        if not confidence_data:
            return "No confidence data available"
        
        mean_conf = confidence_data.get('mean_confidence', 0)
        high_conf_pct = confidence_data.get('high_confidence_docs', 0)
        
        if mean_conf >= 0.8 and high_conf_pct >= 80:
            return "Excellent (High confidence, reliable detections)"
        elif mean_conf >= 0.7 and high_conf_pct >= 60:
            return "Good (Solid confidence levels)"
        elif mean_conf >= 0.6 and high_conf_pct >= 40:
            return "Moderate (Acceptable but room for improvement)"
        else:
            return "Poor (Low confidence, needs threshold adjustment)"
    
    def generate_accuracy_report(self) -> Dict:
        """Generate comprehensive accuracy report"""
        logger.info("Starting comprehensive accuracy analysis...")
        
        try:
            results = self.load_latest_results()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'source_file': getattr(results, 'source_file', 'Unknown'),
                'sampled': results.get('sampled', False),
                'analysis': {}
            }
            
            # Density compliance analysis
            logger.info("Analyzing density compliance...")
            report['analysis']['density'] = self.analyze_density_compliance(results)
            
            # Category distribution analysis  
            logger.info("Analyzing category distribution...")
            report['analysis']['category_distribution'] = self.analyze_category_distribution(results)
            
            # Confidence score analysis
            logger.info("Analyzing confidence scores...")
            report['analysis']['confidence'] = self.analyze_confidence_scores(results)
            
            # Overall grade
            density_grade = report['analysis']['density']['grade'].split()[0]
            category_grade = report['analysis']['category_distribution']['grade'].split()[0]
            
            grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
            avg_grade = (grade_map.get(density_grade, 1) + grade_map.get(category_grade, 1)) / 2
            
            if avg_grade >= 3.5:
                overall_grade = "A (Research Ready)"
            elif avg_grade >= 2.5:
                overall_grade = "B (Good Performance)"
            elif avg_grade >= 1.5:
                overall_grade = "C (Needs Improvement)"
            else:
                overall_grade = "D (Major Issues)"
            
            report['overall_grade'] = overall_grade
            report['summary'] = self._generate_summary(report['analysis'])
            
            return report
            
        except Exception as e:
            logger.error(f"Error during accuracy analysis: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'Failed'
            }
    
    def _generate_summary(self, analysis: Dict) -> str:
        """Generate executive summary of accuracy analysis"""
        summary_parts = []
        
        # Density summary
        density = analysis['density']
        summary_parts.append(f"Density: {density['actual_density']:.1f}/1k words ({density['grade']})")
        
        # Category summary
        category = analysis['category_distribution']
        summary_parts.append(f"Category Balance: {category['grade']}")
        
        # Confidence summary
        confidence = analysis['confidence']
        if 'overall_confidence' in confidence and confidence['overall_confidence']:
            conf_data = confidence['overall_confidence']
            summary_parts.append(f"Confidence: {conf_data.get('mean_confidence', 0):.1%} average")
        
        return " | ".join(summary_parts)

def main():
    """Run accuracy check and generate report"""
    checker = AccuracyChecker()
    report = checker.generate_accuracy_report()
    
    # Save report
    output_file = f"accuracy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 METADISCOURSE ANALYSIS ACCURACY REPORT")
    print("="*60)
    
    if 'error' in report:
        print(f"❌ ERROR: {report['error']}")
        return
    
    print(f"📊 Overall Grade: {report['overall_grade']}")
    print(f"📝 Summary: {report['summary']}")
    
    # Density analysis
    density = report['analysis']['density']
    print(f"\n🎯 DENSITY ANALYSIS")
    print(f"   Current: {density['actual_density']:.1f} markers per 1,000 words")
    print(f"   Target: {density['target_range'][0]}-{density['target_range'][1]} (optimal: {density['target_optimal']})")
    print(f"   Status: {density['grade']}")
    print(f"   Action: {density['recommendation']}")
    
    # Category analysis
    category = report['analysis']['category_distribution']
    print(f"\n⚖️ CATEGORY BALANCE")
    print(f"   Overall: {category['grade']}")
    print(f"   Average deviation: {category['average_deviation']:.1f}%")
    
    if category['recommendations']:
        print("   Recommendations:")
        for rec in category['recommendations']:
            print(f"   • {rec}")
    
    # Confidence analysis
    confidence = report['analysis']['confidence']
    if 'overall_confidence' in confidence and confidence['overall_confidence']:
        conf_data = confidence['overall_confidence']
        print(f"\n🔍 CONFIDENCE ANALYSIS")
        print(f"   Average: {conf_data.get('mean_confidence', 0):.1%}")
        print(f"   High confidence docs: {conf_data.get('high_confidence_docs', 0):.1f}%")
        print(f"   Quality: {confidence['quality_assessment']}")
    
    print(f"\n📄 Full report saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main() 