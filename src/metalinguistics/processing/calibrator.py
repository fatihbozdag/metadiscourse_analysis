"""
Post-Processing Calibration and Balancing
Phase 2.5: Move calibration/balancing to reporting stage, focus core detection on linguistic accuracy
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import json

@dataclass
class CalibrationReport:
    """Report of calibration and balancing analysis"""
    original_markers: int
    calibrated_markers: int
    density_adjustment: float
    category_adjustments: Dict[str, float]
    confidence_threshold_used: float
    linguistic_accuracy_preserved: bool
    recommendations: List[str]

@dataclass
class MarkerStatistics:
    """Statistics for metadiscourse markers"""
    total_count: int
    density_per_100_words: float
    category_distribution: Dict[str, int]
    confidence_distribution: Dict[str, float]
    linguistic_quality_score: float

class PostProcessingCalibrator:
    """
    Post-processing calibrator that analyzes and reports on marker characteristics
    without compromising the core linguistic detection accuracy
    """
    
    def __init__(self):
        """Initialize calibrator with target distributions from literature"""
        # Target distributions based on academic literature
        self.target_densities = {
            'academic_paper': {'min': 2.5, 'max': 8.0, 'optimal': 4.5},  # markers per 100 words
            'research_article': {'min': 3.0, 'max': 10.0, 'optimal': 5.5},
            'thesis_chapter': {'min': 2.0, 'max': 7.0, 'optimal': 4.0},
            'general_academic': {'min': 2.0, 'max': 9.0, 'optimal': 5.0}
        }
        
        # Expected category distributions (percentages)
        self.target_category_distributions = {
            'transitions': {'min': 15, 'max': 35, 'optimal': 25},
            'frame_markers': {'min': 10, 'max': 25, 'optimal': 18},
            'evidentials': {'min': 8, 'max': 20, 'optimal': 15},
            'code_glosses': {'min': 5, 'max': 18, 'optimal': 12},
            'engagement_markers': {'min': 8, 'max': 22, 'optimal': 15},
            'self_mentions': {'min': 5, 'max': 15, 'optimal': 10},
            'boosters': {'min': 3, 'max': 12, 'optimal': 8},
            'hedges': {'min': 8, 'max': 20, 'optimal': 15}
        }
        
        # Minimum confidence thresholds for different purposes
        self.confidence_thresholds = {
            'high_precision': 0.8,
            'balanced': 0.6,
            'high_recall': 0.4,
            'exploratory': 0.2
        }
    
    def analyze_markers(self, markers: List[Any], text: str, 
                       genre: str = 'general_academic') -> MarkerStatistics:
        """
        Analyze marker characteristics without modifying them
        
        Args:
            markers: List of detected markers
            text: Original text
            genre: Text genre for comparison
            
        Returns:
            MarkerStatistics object with comprehensive analysis
        """
        word_count = len(text.split())
        
        # Calculate density
        density = (len(markers) / word_count) * 100 if word_count > 0 else 0
        
        # Category distribution
        category_counts = defaultdict(int)
        confidence_scores = []
        
        for marker in markers:
            category_counts[marker.category] += 1
            confidence_scores.append(marker.confidence)
        
        # Calculate confidence distribution
        confidence_dist = self._calculate_confidence_distribution(confidence_scores)
        
        # Calculate linguistic quality score
        quality_score = self._calculate_linguistic_quality(markers)
        
        return MarkerStatistics(
            total_count=len(markers),
            density_per_100_words=density,
            category_distribution=dict(category_counts),
            confidence_distribution=confidence_dist,
            linguistic_quality_score=quality_score
        )
    
    def calibrate_for_purpose(self, markers: List[Any], text: str,
                            purpose: str = 'balanced',
                            genre: str = 'general_academic') -> CalibrationReport:
        """
        Generate calibration report and recommendations without modifying core detection
        
        Args:
            markers: Original detected markers
            text: Source text
            purpose: Analysis purpose ('high_precision', 'balanced', 'high_recall', 'exploratory')
            genre: Text genre
            
        Returns:
            CalibrationReport with analysis and recommendations
        """
        # Analyze current state
        stats = self.analyze_markers(markers, text, genre)
        
        # Apply confidence threshold for the purpose
        threshold = self.confidence_thresholds[purpose]
        filtered_markers = [m for m in markers if m.confidence >= threshold]
        
        # Analyze filtered results
        filtered_stats = self.analyze_markers(filtered_markers, text, genre)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(stats, filtered_stats, genre, purpose)
        
        # Calculate adjustments
        density_adjustment = self._calculate_density_adjustment(filtered_stats, genre)
        category_adjustments = self._calculate_category_adjustments(filtered_stats, genre)
        
        return CalibrationReport(
            original_markers=len(markers),
            calibrated_markers=len(filtered_markers),
            density_adjustment=density_adjustment,
            category_adjustments=category_adjustments,
            confidence_threshold_used=threshold,
            linguistic_accuracy_preserved=True,  # We don't modify core detection
            recommendations=recommendations
        )
    
    def generate_balanced_subset(self, markers: List[Any], text: str,
                               target_density: Optional[float] = None,
                               preserve_high_confidence: bool = True) -> Tuple[List[Any], CalibrationReport]:
        """
        Generate a balanced subset for reporting while preserving linguistic accuracy
        
        Args:
            markers: All detected markers
            text: Source text
            target_density: Target density (markers per 100 words)
            preserve_high_confidence: Whether to preserve high-confidence markers
            
        Returns:
            Tuple of (subset_markers, calibration_report)
        """
        word_count = len(text.split())
        
        if target_density is None:
            target_density = self.target_densities['general_academic']['optimal']
        
        target_count = int((target_density * word_count) / 100)
        
        # Sort markers by confidence and linguistic quality
        scored_markers = self._score_markers_for_selection(markers)
        
        # Select subset while maintaining category balance
        subset = self._select_balanced_subset(scored_markers, target_count, preserve_high_confidence)
        
        # Generate report
        original_stats = self.analyze_markers(markers, text)
        subset_stats = self.analyze_markers(subset, text)
        
        report = CalibrationReport(
            original_markers=len(markers),
            calibrated_markers=len(subset),
            density_adjustment=(len(subset) - len(markers)) / len(markers) if markers else 0,
            category_adjustments=self._compare_category_distributions(original_stats, subset_stats),
            confidence_threshold_used=min(m.confidence for m in subset) if subset else 0.0,
            linguistic_accuracy_preserved=True,
            recommendations=self._generate_subset_recommendations(original_stats, subset_stats)
        )
        
        return subset, report
    
    def _calculate_confidence_distribution(self, confidence_scores: List[float]) -> Dict[str, float]:
        """Calculate confidence distribution statistics"""
        if not confidence_scores:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': np.mean(confidence_scores),
            'std': np.std(confidence_scores),
            'min': np.min(confidence_scores),
            'max': np.max(confidence_scores)
        }
    
    def _calculate_linguistic_quality(self, markers: List[Any]) -> float:
        """Calculate overall linguistic quality score"""
        if not markers:
            return 0.0
        
        quality_factors = []
        
        for marker in markers:
            # Factor 1: Confidence score
            quality_factors.append(marker.confidence)
            
            # Factor 2: Validation method (ML predictions get higher score)
            if hasattr(marker, 'ml_prediction') and marker.ml_prediction:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.6)
            
            # Factor 3: Academic context (if available)
            if hasattr(marker, 'linguistic_features'):
                academic_score = marker.linguistic_features.get('feat_academic_context_score', 0.5)
                quality_factors.append(academic_score)
        
        return np.mean(quality_factors)
    
    def _calculate_density_adjustment(self, stats: MarkerStatistics, genre: str) -> float:
        """Calculate recommended density adjustment"""
        target_range = self.target_densities.get(genre, self.target_densities['general_academic'])
        current_density = stats.density_per_100_words
        
        if current_density < target_range['min']:
            return (target_range['min'] - current_density) / current_density if current_density > 0 else 0
        elif current_density > target_range['max']:
            return (target_range['max'] - current_density) / current_density
        else:
            return 0.0  # Within acceptable range
    
    def _calculate_category_adjustments(self, stats: MarkerStatistics, genre: str) -> Dict[str, float]:
        """Calculate recommended category adjustments"""
        adjustments = {}
        total_markers = stats.total_count
        
        if total_markers == 0:
            return adjustments
        
        for category, count in stats.category_distribution.items():
            current_percentage = (count / total_markers) * 100
            target_range = self.target_category_distributions.get(category)
            
            if target_range:
                if current_percentage < target_range['min']:
                    adjustments[category] = target_range['min'] - current_percentage
                elif current_percentage > target_range['max']:
                    adjustments[category] = target_range['max'] - current_percentage
                else:
                    adjustments[category] = 0.0
        
        return adjustments
    
    def _generate_recommendations(self, original_stats: MarkerStatistics,
                                filtered_stats: MarkerStatistics,
                                genre: str, purpose: str) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Density recommendations
        target_range = self.target_densities.get(genre, self.target_densities['general_academic'])
        current_density = filtered_stats.density_per_100_words
        
        if current_density < target_range['min']:
            recommendations.append(
                f"Consider lowering confidence threshold to increase marker density "
                f"(current: {current_density:.1f}, target: {target_range['min']}-{target_range['max']})"
            )
        elif current_density > target_range['max']:
            recommendations.append(
                f"Consider raising confidence threshold to reduce marker density "
                f"(current: {current_density:.1f}, target: {target_range['min']}-{target_range['max']})"
            )
        
        # Category distribution recommendations
        total_markers = filtered_stats.total_count
        if total_markers > 0:
            for category, count in filtered_stats.category_distribution.items():
                percentage = (count / total_markers) * 100
                target_range = self.target_category_distributions.get(category)
                
                if target_range:
                    if percentage < target_range['min']:
                        recommendations.append(
                            f"Low {category} representation ({percentage:.1f}%, "
                            f"expected: {target_range['min']}-{target_range['max']}%)"
                        )
                    elif percentage > target_range['max']:
                        recommendations.append(
                            f"High {category} representation ({percentage:.1f}%, "
                            f"expected: {target_range['min']}-{target_range['max']}%)"
                        )
        
        # Quality recommendations
        if filtered_stats.linguistic_quality_score < 0.6:
            recommendations.append(
                f"Consider improving marker detection quality "
                f"(current score: {filtered_stats.linguistic_quality_score:.2f})"
            )
        
        return recommendations
    
    def _score_markers_for_selection(self, markers: List[Any]) -> List[Tuple[Any, float]]:
        """Score markers for subset selection"""
        scored = []
        
        for marker in markers:
            score = marker.confidence  # Base score
            
            # Boost for ML predictions
            if hasattr(marker, 'ml_prediction') and marker.ml_prediction:
                score += 0.1
            
            # Boost for academic context
            if hasattr(marker, 'linguistic_features'):
                academic_score = marker.linguistic_features.get('feat_academic_context_score', 0)
                score += academic_score * 0.2
            
            scored.append((marker, score))
        
        return sorted(scored, key=lambda x: x[1], reverse=True)
    
    def _select_balanced_subset(self, scored_markers: List[Tuple[Any, float]],
                              target_count: int, preserve_high_confidence: bool) -> List[Any]:
        """Select balanced subset maintaining category distribution"""
        if target_count >= len(scored_markers):
            return [marker for marker, _ in scored_markers]
        
        # If preserving high confidence, take top markers first
        if preserve_high_confidence:
            high_conf_threshold = 0.8
            high_conf_markers = [(m, s) for m, s in scored_markers if m.confidence >= high_conf_threshold]
            
            if len(high_conf_markers) >= target_count:
                return [marker for marker, _ in high_conf_markers[:target_count]]
            else:
                # Take all high confidence + fill remaining slots
                selected = [marker for marker, _ in high_conf_markers]
                remaining_needed = target_count - len(selected)
                remaining_markers = [m for m, s in scored_markers if m.confidence < high_conf_threshold]
                selected.extend(remaining_markers[:remaining_needed])
                return selected
        
        # Standard balanced selection
        return [marker for marker, _ in scored_markers[:target_count]]
    
    def _compare_category_distributions(self, original: MarkerStatistics,
                                      subset: MarkerStatistics) -> Dict[str, float]:
        """Compare category distributions between original and subset"""
        adjustments = {}
        
        orig_total = original.total_count
        subset_total = subset.total_count
        
        if orig_total == 0 or subset_total == 0:
            return adjustments
        
        for category in set(original.category_distribution.keys()) | set(subset.category_distribution.keys()):
            orig_pct = (original.category_distribution.get(category, 0) / orig_total) * 100
            subset_pct = (subset.category_distribution.get(category, 0) / subset_total) * 100
            adjustments[category] = subset_pct - orig_pct
        
        return adjustments
    
    def _generate_subset_recommendations(self, original: MarkerStatistics,
                                       subset: MarkerStatistics) -> List[str]:
        """Generate recommendations for subset selection"""
        recommendations = []
        
        reduction_pct = ((original.total_count - subset.total_count) / original.total_count) * 100
        recommendations.append(f"Reduced marker count by {reduction_pct:.1f}%")
        
        if subset.linguistic_quality_score > original.linguistic_quality_score:
            improvement = ((subset.linguistic_quality_score - original.linguistic_quality_score) / 
                          original.linguistic_quality_score) * 100
            recommendations.append(f"Improved linguistic quality by {improvement:.1f}%")
        
        return recommendations
    
    def export_calibration_report(self, report: CalibrationReport, filepath: str):
        """Export calibration report to JSON"""
        report_dict = {
            'original_markers': report.original_markers,
            'calibrated_markers': report.calibrated_markers,
            'density_adjustment': report.density_adjustment,
            'category_adjustments': report.category_adjustments,
            'confidence_threshold_used': report.confidence_threshold_used,
            'linguistic_accuracy_preserved': report.linguistic_accuracy_preserved,
            'recommendations': report.recommendations
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)

def main():
    """Test the post-processing calibrator"""
    print("Testing Post-Processing Calibrator...")
    
    # This would normally use real markers from the enhanced analyzer
    # For demo purposes, we'll create mock markers
    class MockMarker:
        def __init__(self, category, confidence, ml_prediction=True):
            self.category = category
            self.confidence = confidence
            self.ml_prediction = ml_prediction
            self.linguistic_features = {'feat_academic_context_score': confidence * 0.8}
    
    # Create test markers
    test_markers = [
        MockMarker('transitions', 0.9),
        MockMarker('frame_markers', 0.8),
        MockMarker('evidentials', 0.7),
        MockMarker('boosters', 0.6),
        MockMarker('hedges', 0.5),
        MockMarker('self_mentions', 0.4),
        MockMarker('code_glosses', 0.8),
        MockMarker('engagement_markers', 0.7)
    ]
    
    test_text = "This is a sample academic text with approximately fifty words for testing " \
                "the calibration system and its ability to analyze marker density and " \
                "distribution patterns in academic discourse for validation purposes."
    
    calibrator = PostProcessingCalibrator()
    
    # Test analysis
    stats = calibrator.analyze_markers(test_markers, test_text)
    print(f"Original markers: {stats.total_count}")
    print(f"Density: {stats.density_per_100_words:.1f} markers per 100 words")
    print(f"Quality score: {stats.linguistic_quality_score:.2f}")
    
    # Test calibration
    report = calibrator.calibrate_for_purpose(test_markers, test_text, 'balanced')
    print(f"\nCalibrated markers: {report.calibrated_markers}")
    print(f"Recommendations: {len(report.recommendations)}")
    for rec in report.recommendations:
        print(f"  - {rec}")
    
    # Export report
    calibrator.export_calibration_report(report, 'calibration_report.json')
    print("\nCalibration report exported to 'calibration_report.json'")

if __name__ == "__main__":
    main()