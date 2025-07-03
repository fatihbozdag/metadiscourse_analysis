"""
Enhanced Deduplication using Confidence and Linguistic Specificity
Phase 2.6: Smart deduplication based on overlap resolution with linguistic priority
"""

from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass
import numpy as np
from enum import Enum

class OverlapType(Enum):
    """Types of marker overlap"""
    EXACT_MATCH = "exact_match"           # Identical boundaries
    SUBSTRING = "substring"               # One marker is substring of another
    PARTIAL_OVERLAP = "partial_overlap"   # Markers partially overlap
    ADJACENT = "adjacent"                 # Markers are adjacent
    NESTED = "nested"                     # One marker nested within another

@dataclass
class OverlapResolution:
    """Resolution decision for overlapping markers"""
    kept_marker: Any
    removed_markers: List[Any]
    overlap_type: OverlapType
    confidence_factor: float
    specificity_factor: float
    resolution_reason: str

class EnhancedDeduplicator:
    """
    Enhanced deduplication system that considers confidence and linguistic specificity
    """
    
    def __init__(self):
        """Initialize deduplicator with specificity rules"""
        # Linguistic specificity hierarchy (higher = more specific)
        self.category_specificity = {
            'evidentials': 0.9,        # "according to research" vs "according"
            'frame_markers': 0.8,      # "in conclusion" vs "conclusion"
            'code_glosses': 0.8,       # "in other words" vs "words"
            'transitions': 0.7,        # "however" vs "however,"
            'engagement_markers': 0.6,  # "note that" vs "note"
            'self_mentions': 0.5,      # "our study" vs "our"
            'boosters': 0.4,           # "clearly demonstrates" vs "clearly"
            'hedges': 0.4              # "might suggest" vs "might"
        }
        
        # Specificity boost for multi-word markers
        self.multiword_boost = 0.2
        
        # Academic context boost
        self.academic_context_boost = 0.15
    
    def deduplicate_markers(self, markers: List[Any], 
                          preserve_high_confidence: bool = True,
                          min_confidence_diff: float = 0.2) -> Tuple[List[Any], List[OverlapResolution]]:
        """
        Deduplicate markers using enhanced confidence and specificity analysis
        
        Args:
            markers: List of detected markers
            preserve_high_confidence: Whether to preserve high-confidence markers
            min_confidence_diff: Minimum confidence difference for resolution
            
        Returns:
            Tuple of (deduplicated_markers, resolution_log)
        """
        if not markers:
            return [], []
        
        # Sort markers by position for efficient overlap detection
        sorted_markers = sorted(markers, key=lambda m: (m.start_pos, m.end_pos))
        
        # Find all overlaps
        overlaps = self._find_overlaps(sorted_markers)
        
        # Resolve overlaps
        resolution_log = []
        markers_to_remove = set()
        
        for overlap_group in overlaps:
            resolution = self._resolve_overlap(overlap_group, preserve_high_confidence, min_confidence_diff)
            resolution_log.append(resolution)
            
            # Mark markers for removal
            for marker in resolution.removed_markers:
                markers_to_remove.add(id(marker))
        
        # Create final deduplicated list
        deduplicated = [m for m in markers if id(m) not in markers_to_remove]
        
        return deduplicated, resolution_log
    
    def _find_overlaps(self, sorted_markers: List[Any]) -> List[List[Any]]:
        """Find all overlapping marker groups"""
        overlap_groups = []
        used_markers = set()
        
        for i, marker in enumerate(sorted_markers):
            if id(marker) in used_markers:
                continue
            
            # Find all markers that overlap with this one
            overlap_group = [marker]
            used_markers.add(id(marker))
            
            for j in range(i + 1, len(sorted_markers)):
                other_marker = sorted_markers[j]
                
                if id(other_marker) in used_markers:
                    continue
                
                if self._markers_overlap(marker, other_marker):
                    overlap_group.append(other_marker)
                    used_markers.add(id(other_marker))
                elif other_marker.start_pos >= marker.end_pos:
                    # No more possible overlaps for this marker
                    break
            
            # Only add groups with actual overlaps
            if len(overlap_group) > 1:
                overlap_groups.append(overlap_group)
        
        return overlap_groups
    
    def _markers_overlap(self, marker1: Any, marker2: Any, 
                        adjacency_threshold: int = 2) -> bool:
        """Check if two markers overlap or are adjacent"""
        # Check for character-level overlap
        if (marker1.start_pos < marker2.end_pos and marker2.start_pos < marker1.end_pos):
            return True
        
        # Check for adjacency (markers very close to each other)
        if (abs(marker1.end_pos - marker2.start_pos) <= adjacency_threshold or
            abs(marker2.end_pos - marker1.start_pos) <= adjacency_threshold):
            return True
        
        return False
    
    def _resolve_overlap(self, overlap_group: List[Any], 
                        preserve_high_confidence: bool,
                        min_confidence_diff: float) -> OverlapResolution:
        """Resolve overlap between multiple markers"""
        if len(overlap_group) == 1:
            return OverlapResolution(
                kept_marker=overlap_group[0],
                removed_markers=[],
                overlap_type=OverlapType.EXACT_MATCH,
                confidence_factor=1.0,
                specificity_factor=1.0,
                resolution_reason="No overlap to resolve"
            )
        
        # Determine overlap type
        overlap_type = self._classify_overlap(overlap_group)
        
        # Calculate scores for each marker
        marker_scores = []
        for marker in overlap_group:
            score = self._calculate_marker_score(marker)
            marker_scores.append((marker, score))
        
        # Sort by score (highest first)
        marker_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_marker, best_score = marker_scores[0]
        other_markers = [marker for marker, _ in marker_scores[1:]]
        
        # Special handling for high confidence preservation
        if preserve_high_confidence:
            high_conf_markers = [m for m, s in marker_scores if m.confidence >= 0.8]
            if len(high_conf_markers) == 1:
                best_marker = high_conf_markers[0]
                other_markers = [m for m in overlap_group if m != best_marker]
        
        # Generate resolution reason
        reason = self._generate_resolution_reason(best_marker, other_markers, overlap_type)
        
        return OverlapResolution(
            kept_marker=best_marker,
            removed_markers=other_markers,
            overlap_type=overlap_type,
            confidence_factor=best_marker.confidence,
            specificity_factor=self._calculate_specificity_score(best_marker),
            resolution_reason=reason
        )
    
    def _classify_overlap(self, overlap_group: List[Any]) -> OverlapType:
        """Classify the type of overlap between markers"""
        if len(overlap_group) == 2:
            m1, m2 = overlap_group
            
            # Exact match
            if m1.start_pos == m2.start_pos and m1.end_pos == m2.end_pos:
                return OverlapType.EXACT_MATCH
            
            # Substring relationship
            if (m1.start_pos >= m2.start_pos and m1.end_pos <= m2.end_pos) or \
               (m2.start_pos >= m1.start_pos and m2.end_pos <= m1.end_pos):
                return OverlapType.SUBSTRING
            
            # Nested (one contains the other with different boundaries)
            if (m1.start_pos < m2.start_pos and m1.end_pos > m2.end_pos) or \
               (m2.start_pos < m1.start_pos and m2.end_pos > m1.end_pos):
                return OverlapType.NESTED
            
            # Adjacent
            if abs(m1.end_pos - m2.start_pos) <= 2 or abs(m2.end_pos - m1.start_pos) <= 2:
                return OverlapType.ADJACENT
            
            # Partial overlap
            return OverlapType.PARTIAL_OVERLAP
        
        # For groups with more than 2 markers, use general classification
        return OverlapType.PARTIAL_OVERLAP
    
    def _calculate_marker_score(self, marker: Any) -> float:
        """Calculate comprehensive score for marker priority"""
        score = 0.0
        
        # 1. Base confidence (40% of score)
        score += marker.confidence * 0.4
        
        # 2. Linguistic specificity (30% of score)
        specificity = self._calculate_specificity_score(marker)
        score += specificity * 0.3
        
        # 3. Academic context (20% of score)
        academic_score = self._calculate_academic_score(marker)
        score += academic_score * 0.2
        
        # 4. ML prediction boost (10% of score)
        if hasattr(marker, 'ml_prediction') and marker.ml_prediction:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_specificity_score(self, marker: Any) -> float:
        """Calculate linguistic specificity score"""
        base_specificity = self.category_specificity.get(marker.category, 0.5)
        
        # Boost for multi-word markers
        word_count = len(marker.text.split())
        if word_count > 1:
            base_specificity += self.multiword_boost * min(word_count - 1, 3)
        
        # Boost for longer markers (more specific)
        length_boost = min(len(marker.text) / 20, 0.1)  # Up to 0.1 boost for long markers
        
        return min(1.0, base_specificity + length_boost)
    
    def _calculate_academic_score(self, marker: Any) -> float:
        """Calculate academic context relevance score"""
        if hasattr(marker, 'linguistic_features'):
            return marker.linguistic_features.get('feat_academic_context_score', 0.5)
        
        # Fallback: simple academic context heuristics
        academic_indicators = ['study', 'research', 'analysis', 'findings', 'data']
        context_lower = marker.context.lower() if hasattr(marker, 'context') else ''
        
        academic_count = sum(1 for indicator in academic_indicators if indicator in context_lower)
        return min(1.0, 0.3 + academic_count * 0.1)
    
    def _generate_resolution_reason(self, kept_marker: Any, 
                                  removed_markers: List[Any], 
                                  overlap_type: OverlapType) -> str:
        """Generate human-readable reason for resolution decision"""
        kept_score = self._calculate_marker_score(kept_marker)
        
        reasons = []
        
        # Confidence comparison
        conf_diffs = [kept_marker.confidence - m.confidence for m in removed_markers]
        avg_conf_diff = np.mean(conf_diffs) if conf_diffs else 0
        
        if avg_conf_diff > 0.2:
            reasons.append(f"higher confidence ({kept_marker.confidence:.2f})")
        
        # Specificity comparison
        kept_specificity = self._calculate_specificity_score(kept_marker)
        spec_diffs = [kept_specificity - self._calculate_specificity_score(m) for m in removed_markers]
        avg_spec_diff = np.mean(spec_diffs) if spec_diffs else 0
        
        if avg_spec_diff > 0.1:
            reasons.append("higher linguistic specificity")
        
        # Length comparison
        if len(kept_marker.text) > max(len(m.text) for m in removed_markers):
            reasons.append("longer/more complete phrase")
        
        # ML prediction
        if hasattr(kept_marker, 'ml_prediction') and kept_marker.ml_prediction:
            ml_count = sum(1 for m in removed_markers 
                          if hasattr(m, 'ml_prediction') and m.ml_prediction)
            if ml_count < len(removed_markers):
                reasons.append("ML-validated")
        
        # Default reason
        if not reasons:
            reasons.append(f"higher overall score ({kept_score:.2f})")
        
        reason_text = ", ".join(reasons)
        return f"Kept '{kept_marker.text}' over {len(removed_markers)} overlapping marker(s) due to {reason_text} ({overlap_type.value})"
    
    def analyze_deduplication(self, original_markers: List[Any], 
                            deduplicated_markers: List[Any],
                            resolution_log: List[OverlapResolution]) -> Dict[str, Any]:
        """Analyze the impact of deduplication"""
        analysis = {
            'original_count': len(original_markers),
            'deduplicated_count': len(deduplicated_markers),
            'removed_count': len(original_markers) - len(deduplicated_markers),
            'removal_percentage': ((len(original_markers) - len(deduplicated_markers)) / len(original_markers) * 100) if original_markers else 0,
            'overlap_types': {},
            'avg_confidence_change': 0.0,
            'specificity_improvements': 0,
            'resolution_summary': []
        }
        
        # Analyze overlap types
        for resolution in resolution_log:
            overlap_type = resolution.overlap_type.value
            analysis['overlap_types'][overlap_type] = analysis['overlap_types'].get(overlap_type, 0) + 1
        
        # Calculate confidence changes
        if original_markers and deduplicated_markers:
            orig_avg_conf = np.mean([m.confidence for m in original_markers])
            dedup_avg_conf = np.mean([m.confidence for m in deduplicated_markers])
            analysis['avg_confidence_change'] = dedup_avg_conf - orig_avg_conf
        
        # Count specificity improvements
        for resolution in resolution_log:
            kept_spec = self._calculate_specificity_score(resolution.kept_marker)
            removed_specs = [self._calculate_specificity_score(m) for m in resolution.removed_markers]
            
            if removed_specs and kept_spec > max(removed_specs):
                analysis['specificity_improvements'] += 1
        
        # Create resolution summary
        for resolution in resolution_log:
            analysis['resolution_summary'].append({
                'kept': resolution.kept_marker.text,
                'removed': [m.text for m in resolution.removed_markers],
                'reason': resolution.resolution_reason
            })
        
        return analysis

def test_enhanced_deduplicator():
    """Test the enhanced deduplication system"""
    
    class MockMarker:
        def __init__(self, text, category, start_pos, end_pos, confidence, 
                    ml_prediction=True, context="academic research context"):
            self.text = text
            self.category = category
            self.start_pos = start_pos
            self.end_pos = end_pos
            self.confidence = confidence
            self.ml_prediction = ml_prediction
            self.context = context
            self.linguistic_features = {'feat_academic_context_score': confidence * 0.8}
    
    # Create test markers with overlaps
    test_markers = [
        MockMarker("however", "transitions", 0, 7, 0.7),
        MockMarker("However,", "transitions", 0, 8, 0.8),  # Overlaps with above
        MockMarker("in conclusion", "frame_markers", 50, 63, 0.9),
        MockMarker("conclusion", "frame_markers", 53, 63, 0.6),  # Substring of above
        MockMarker("such as", "code_glosses", 100, 107, 0.8),
        MockMarker("as", "code_glosses", 105, 107, 0.4),  # Overlaps with above
        MockMarker("clearly", "boosters", 150, 157, 0.7),
        MockMarker("demonstrates", "evidentials", 200, 212, 0.8)
    ]
    
    print("Testing Enhanced Deduplication...")
    print(f"Original markers: {len(test_markers)}")
    for marker in test_markers:
        print(f"  '{marker.text}' ({marker.start_pos}-{marker.end_pos}) conf={marker.confidence}")
    
    deduplicator = EnhancedDeduplicator()
    deduplicated, resolution_log = deduplicator.deduplicate_markers(test_markers)
    
    print(f"\nDeduplicated markers: {len(deduplicated)}")
    for marker in deduplicated:
        print(f"  '{marker.text}' ({marker.start_pos}-{marker.end_pos}) conf={marker.confidence}")
    
    print(f"\nResolution log ({len(resolution_log)} resolutions):")
    for i, resolution in enumerate(resolution_log, 1):
        print(f"{i}. {resolution.resolution_reason}")
    
    # Analyze deduplication
    analysis = deduplicator.analyze_deduplication(test_markers, deduplicated, resolution_log)
    print(f"\nDeduplication Analysis:")
    print(f"  Removed {analysis['removed_count']} markers ({analysis['removal_percentage']:.1f}%)")
    print(f"  Average confidence change: {analysis['avg_confidence_change']:+.3f}")
    print(f"  Specificity improvements: {analysis['specificity_improvements']}")

if __name__ == "__main__":
    test_enhanced_deduplicator()