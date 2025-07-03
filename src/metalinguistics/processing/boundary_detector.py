"""
Intelligent Marker Boundary Detection using Linguistic Features
Phase 2.4: Replace fixed heuristics with smart boundary detection
"""

import spacy
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class BoundaryType(Enum):
    """Types of boundary detection strategies"""
    SINGLE_TOKEN = "single_token"
    MULTI_TOKEN_PHRASE = "multi_token_phrase"
    DEPENDENCY_BASED = "dependency_based"
    SYNTACTIC_CHUNK = "syntactic_chunk"

@dataclass
class MarkerBoundary:
    """Represents detected marker boundaries"""
    start_char: int
    end_char: int
    start_token: int
    end_token: int
    text: str
    confidence: float
    boundary_type: BoundaryType
    linguistic_justification: str

class IntelligentBoundaryDetector:
    """
    Intelligent boundary detection using linguistic features instead of fixed heuristics
    """
    
    def __init__(self, model_name: str = "en_core_web_trf"):
        """Initialize with Spacy model"""
        self.nlp = spacy.load(model_name)
        
        # Metadiscourse phrase patterns with their typical structures
        self.phrase_patterns = {
            'frame_markers': {
                'in conclusion': {'type': 'prepositional_phrase', 'min_tokens': 2, 'max_tokens': 3},
                'first of all': {'type': 'prepositional_phrase', 'min_tokens': 3, 'max_tokens': 3},
                'on the other hand': {'type': 'prepositional_phrase', 'min_tokens': 5, 'max_tokens': 5},
                'to summarize': {'type': 'infinitive_phrase', 'min_tokens': 2, 'max_tokens': 2},
                'next section': {'type': 'noun_phrase', 'min_tokens': 2, 'max_tokens': 2}
            },
            'transitions': {
                'however': {'type': 'adverb', 'min_tokens': 1, 'max_tokens': 1},
                'on the contrary': {'type': 'prepositional_phrase', 'min_tokens': 4, 'max_tokens': 4},
                'in contrast': {'type': 'prepositional_phrase', 'min_tokens': 2, 'max_tokens': 2},
                'as a result': {'type': 'prepositional_phrase', 'min_tokens': 3, 'max_tokens': 3}
            },
            'code_glosses': {
                'in other words': {'type': 'prepositional_phrase', 'min_tokens': 3, 'max_tokens': 3},
                'that is': {'type': 'demonstrative_phrase', 'min_tokens': 2, 'max_tokens': 2},
                'such as': {'type': 'prepositional_phrase', 'min_tokens': 2, 'max_tokens': 2},
                'for example': {'type': 'prepositional_phrase', 'min_tokens': 2, 'max_tokens': 2}
            },
            'engagement_markers': {
                'note that': {'type': 'verb_phrase', 'min_tokens': 2, 'max_tokens': 2},
                'consider the': {'type': 'verb_phrase', 'min_tokens': 2, 'max_tokens': 3},
                'see figure': {'type': 'verb_phrase', 'min_tokens': 2, 'max_tokens': 2}
            },
            'evidentials': {
                'according to': {'type': 'prepositional_phrase', 'min_tokens': 2, 'max_tokens': 2},
                'research shows': {'type': 'verb_phrase', 'min_tokens': 2, 'max_tokens': 2},
                'studies indicate': {'type': 'verb_phrase', 'min_tokens': 2, 'max_tokens': 2}
            }
        }
    
    def detect_boundaries(self, text: str, potential_markers: List[str]) -> List[MarkerBoundary]:
        """
        Detect intelligent boundaries for potential markers in text
        
        Args:
            text: Input text
            potential_markers: List of potential marker strings to find boundaries for
            
        Returns:
            List of detected marker boundaries with confidence scores
        """
        doc = self.nlp(text)
        boundaries = []
        
        for marker in potential_markers:
            marker_boundaries = self._find_marker_boundaries(doc, marker, text)
            boundaries.extend(marker_boundaries)
        
        # Remove duplicates and sort by confidence
        unique_boundaries = self._remove_overlapping_boundaries(boundaries)
        return sorted(unique_boundaries, key=lambda x: x.confidence, reverse=True)
    
    def _find_marker_boundaries(self, doc, marker: str, original_text: str) -> List[MarkerBoundary]:
        """Find all possible boundaries for a specific marker"""
        boundaries = []
        marker_lower = marker.lower()
        
        # Strategy 1: Exact phrase matching with linguistic validation
        boundaries.extend(self._exact_phrase_boundaries(doc, marker, original_text))
        
        # Strategy 2: Dependency-based boundary detection
        boundaries.extend(self._dependency_based_boundaries(doc, marker, original_text))
        
        # Strategy 3: Syntactic chunk boundaries
        boundaries.extend(self._syntactic_chunk_boundaries(doc, marker, original_text))
        
        return boundaries
    
    def _exact_phrase_boundaries(self, doc, marker: str, original_text: str) -> List[MarkerBoundary]:
        """Find boundaries using exact phrase matching with linguistic validation"""
        boundaries = []
        marker_lower = marker.lower()
        text_lower = original_text.lower()
        
        # Find all occurrences of the marker
        for match in re.finditer(re.escape(marker_lower), text_lower):
            start_char = match.start()
            end_char = match.end()
            
            # Find corresponding tokens in spacy doc
            start_token_idx, end_token_idx = self._char_to_token_indices(doc, start_char, end_char)
            
            if start_token_idx is not None and end_token_idx is not None:
                # Validate linguistic coherence
                token_span = doc[start_token_idx:end_token_idx + 1]
                confidence = self._calculate_phrase_confidence(token_span, marker)
                
                if confidence > 0.3:  # Minimum confidence threshold
                    boundary = MarkerBoundary(
                        start_char=start_char,
                        end_char=end_char,
                        start_token=start_token_idx,
                        end_token=end_token_idx,
                        text=original_text[start_char:end_char],
                        confidence=confidence,
                        boundary_type=BoundaryType.MULTI_TOKEN_PHRASE if len(token_span) > 1 else BoundaryType.SINGLE_TOKEN,
                        linguistic_justification=self._get_phrase_justification(token_span)
                    )
                    boundaries.append(boundary)
        
        return boundaries
    
    def _dependency_based_boundaries(self, doc, marker: str, original_text: str) -> List[MarkerBoundary]:
        """Find boundaries using dependency parsing"""
        boundaries = []
        marker_tokens = marker.lower().split()
        
        # Look for dependency patterns that match the marker
        for i, token in enumerate(doc):
            if token.text.lower() == marker_tokens[0]:
                # Try to find the complete phrase using dependencies
                phrase_tokens = self._extract_dependency_phrase(token, marker_tokens)
                
                if phrase_tokens and len(phrase_tokens) >= len(marker_tokens):
                    start_char = phrase_tokens[0].idx
                    end_char = phrase_tokens[-1].idx + len(phrase_tokens[-1].text)
                    
                    confidence = self._calculate_dependency_confidence(phrase_tokens, marker)
                    
                    if confidence > 0.4:
                        boundary = MarkerBoundary(
                            start_char=start_char,
                            end_char=end_char,
                            start_token=phrase_tokens[0].i,
                            end_token=phrase_tokens[-1].i,
                            text=original_text[start_char:end_char],
                            confidence=confidence,
                            boundary_type=BoundaryType.DEPENDENCY_BASED,
                            linguistic_justification=self._get_dependency_justification(phrase_tokens)
                        )
                        boundaries.append(boundary)
        
        return boundaries
    
    def _syntactic_chunk_boundaries(self, doc, marker: str, original_text: str) -> List[MarkerBoundary]:
        """Find boundaries using syntactic chunking (noun phrases, verb phrases, etc.)"""
        boundaries = []
        marker_lower = marker.lower()
        
        # Check noun chunks
        for chunk in doc.noun_chunks:
            if marker_lower in chunk.text.lower():
                confidence = self._calculate_chunk_confidence(chunk, marker)
                
                if confidence > 0.5:
                    boundary = MarkerBoundary(
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        start_token=chunk.start,
                        end_token=chunk.end - 1,
                        text=chunk.text,
                        confidence=confidence,
                        boundary_type=BoundaryType.SYNTACTIC_CHUNK,
                        linguistic_justification=f"Noun chunk: {chunk.root.dep_} ({chunk.root.head.text})"
                    )
                    boundaries.append(boundary)
        
        return boundaries
    
    def _char_to_token_indices(self, doc, start_char: int, end_char: int) -> Tuple[Optional[int], Optional[int]]:
        """Convert character indices to token indices"""
        start_token_idx = None
        end_token_idx = None
        
        for token in doc:
            if token.idx <= start_char < token.idx + len(token.text):
                start_token_idx = token.i
            if token.idx < end_char <= token.idx + len(token.text):
                end_token_idx = token.i
        
        return start_token_idx, end_token_idx
    
    def _calculate_phrase_confidence(self, token_span, marker: str) -> float:
        """Calculate confidence for a phrase boundary based on linguistic features"""
        confidence = 0.5  # Base confidence
        
        # Check if tokens form a coherent phrase
        if len(token_span) == 1:
            token = token_span[0]
            # Single token markers get higher confidence if they're function words
            if token.pos_ in ['ADV', 'CONJ', 'SCONJ', 'CCONJ']:
                confidence += 0.3
            elif token.dep_ in ['advmod', 'cc', 'mark']:
                confidence += 0.2
        else:
            # Multi-token phrases
            # Check if it's a known phrase pattern
            marker_lower = marker.lower()
            for category, patterns in self.phrase_patterns.items():
                if marker_lower in patterns:
                    pattern_info = patterns[marker_lower]
                    if pattern_info['min_tokens'] <= len(token_span) <= pattern_info['max_tokens']:
                        confidence += 0.4
                        break
            
            # Check syntactic coherence
            if self._is_syntactically_coherent(token_span):
                confidence += 0.2
        
        return min(1.0, confidence)
    
    def _calculate_dependency_confidence(self, phrase_tokens: List, marker: str) -> float:
        """Calculate confidence for dependency-based boundaries"""
        confidence = 0.6  # Higher base for dependency-based detection
        
        # Check if the phrase has a clear syntactic head
        heads = [token.head for token in phrase_tokens]
        if len(set(heads)) <= 2:  # Most tokens share 1-2 heads
            confidence += 0.2
        
        # Check for prepositions at the start (common in metadiscourse)
        if phrase_tokens and phrase_tokens[0].pos_ == 'ADP':
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_chunk_confidence(self, chunk, marker: str) -> float:
        """Calculate confidence for syntactic chunk boundaries"""
        confidence = 0.7  # High base confidence for syntactic chunks
        
        # Prefer chunks that start with the marker
        if chunk.text.lower().startswith(marker.lower()):
            confidence += 0.2
        
        # Check chunk type
        if chunk.root.dep_ in ['nsubj', 'dobj', 'pobj']:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _extract_dependency_phrase(self, start_token, target_tokens: List[str]) -> Optional[List]:
        """Extract a phrase based on dependency relations"""
        phrase_tokens = [start_token]
        
        # Try to find the rest of the tokens in the immediate dependency neighborhood
        for target_token in target_tokens[1:]:
            found = False
            
            # Check children
            for child in start_token.children:
                if child.text.lower() == target_token:
                    phrase_tokens.append(child)
                    found = True
                    break
            
            # Check siblings (same head)
            if not found:
                for token in start_token.doc:
                    if (token.head == start_token.head and 
                        token.text.lower() == target_token and
                        abs(token.i - start_token.i) <= 3):  # Within reasonable distance
                        phrase_tokens.append(token)
                        found = True
                        break
            
            if not found:
                return None
        
        # Sort by token position
        phrase_tokens.sort(key=lambda t: t.i)
        return phrase_tokens
    
    def _is_syntactically_coherent(self, token_span) -> bool:
        """Check if a token span forms a syntactically coherent phrase"""
        if len(token_span) <= 1:
            return True
        
        # Check if tokens are contiguous or nearly contiguous
        positions = [token.i for token in token_span]
        max_gap = max(positions) - min(positions) - len(positions) + 1
        
        if max_gap > 2:  # Too many gaps
            return False
        
        # Check if there's a clear head-dependent relationship
        heads = [token.head for token in token_span]
        internal_heads = [head for head in heads if head in token_span]
        
        return len(internal_heads) > 0  # At least one internal dependency
    
    def _get_phrase_justification(self, token_span) -> str:
        """Generate linguistic justification for phrase boundary"""
        if len(token_span) == 1:
            token = token_span[0]
            return f"Single token: {token.pos_} ({token.dep_})"
        else:
            pos_tags = [token.pos_ for token in token_span]
            return f"Multi-token phrase: {' + '.join(pos_tags)}"
    
    def _get_dependency_justification(self, phrase_tokens: List) -> str:
        """Generate justification for dependency-based boundary"""
        if not phrase_tokens:
            return "Empty phrase"
        
        deps = [token.dep_ for token in phrase_tokens]
        return f"Dependency phrase: {' -> '.join(deps)}"
    
    def _remove_overlapping_boundaries(self, boundaries: List[MarkerBoundary]) -> List[MarkerBoundary]:
        """Remove overlapping boundaries, keeping the highest confidence ones"""
        if not boundaries:
            return []
        
        # Sort by confidence descending
        sorted_boundaries = sorted(boundaries, key=lambda x: x.confidence, reverse=True)
        non_overlapping = []
        
        for boundary in sorted_boundaries:
            # Check if this boundary overlaps with any already selected
            overlaps = False
            for selected in non_overlapping:
                if (boundary.start_char < selected.end_char and 
                    boundary.end_char > selected.start_char):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(boundary)
        
        return non_overlapping

def test_boundary_detector():
    """Test the intelligent boundary detector"""
    detector = IntelligentBoundaryDetector()
    
    test_text = """
    This study demonstrates the effectiveness of the proposed method. However, we need to consider
    the limitations. In conclusion, the results show significant improvement. According to previous
    research, such findings are unprecedented. The first section discusses methodology, while the
    second section presents results.
    """
    
    potential_markers = [
        "However", "In conclusion", "According to", "first section", "second section",
        "such", "demonstrates"
    ]
    
    print("Testing Intelligent Boundary Detection...")
    boundaries = detector.detect_boundaries(test_text, potential_markers)
    
    print(f"\nFound {len(boundaries)} marker boundaries:")
    for i, boundary in enumerate(boundaries, 1):
        print(f"{i}. '{boundary.text}' ({boundary.start_char}-{boundary.end_char})")
        print(f"   Type: {boundary.boundary_type.value}")
        print(f"   Confidence: {boundary.confidence:.3f}")
        print(f"   Justification: {boundary.linguistic_justification}")
        print()

if __name__ == "__main__":
    test_boundary_detector()