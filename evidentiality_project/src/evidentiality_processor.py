"""Processor for evidentiality and metadiscourse analysis."""

import os
import re
import sys
import pandas as pd
from tqdm import tqdm
import spacy
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path to import from the main project
try:
    from src.processor import TextProcessor
    from src.markers import MetadiscourseMarkers
except ModuleNotFoundError:
    import sys
    import os
    # Try adding parent directory to sys.path
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from processor import TextProcessor
        from markers import MetadiscourseMarkers
    except ModuleNotFoundError as e:
        raise ImportError("Could not import TextProcessor and MetadiscourseMarkers. Please ensure the parent src directory is in your PYTHONPATH.")

# Import evidentiality markers
from evidentiality_markers import EvidentialityMarkers, EVIDENTIALITY_MARKERS, MENTAL_SPACE_BUILDERS

class EvidentialityProcessor(TextProcessor):
    """Process and analyze texts for evidentiality and metadiscourse markers."""
    
    def __init__(self, model_name='en_core_web_trf'):
        """Initialize the processor with a spaCy model."""
        super().__init__(model_name=model_name)
        
        # Initialize evidentiality markers
        self.evidentiality_markers_obj = EvidentialityMarkers()
        self.evidentiality_markers = EVIDENTIALITY_MARKERS
        self.mental_space_builders = MENTAL_SPACE_BUILDERS
        
        # Initialize metadiscourse markers with hierarchical structure
        self.metadiscourse_markers_obj = MetadiscourseMarkers()
        
        # Regex pattern for sentence splitting
        self.sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
    
    def extract_context(self, text: str, marker_position: int, window_size: int = 2) -> Dict[str, str]:
        """Extract context around a marker.
        
        Args:
            text: Full text
            marker_position: Position of the marker in the text
            window_size: Number of sentences before and after to extract
            
        Returns:
            Dictionary with before_context, sentence, and after_context
        """
        # Split text into sentences
        sentences = self.sentence_pattern.split(text)
        
        # Find which sentence contains the marker
        current_pos = 0
        sentence_index = -1
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence) + 1  # +1 for the space after the sentence
            if current_pos <= marker_position < current_pos + sentence_length:
                sentence_index = i
                break
            current_pos += sentence_length
        
        if sentence_index == -1:
            return {
                "before_context": "",
                "sentence": text,
                "after_context": ""
            }
        
        # Extract the sentence containing the marker
        sentence = sentences[sentence_index].strip()
        
        # Extract context before
        start_idx = max(0, sentence_index - window_size)
        before_context = " ".join(sentences[start_idx:sentence_index]).strip()
        
        # Extract context after
        end_idx = min(len(sentences), sentence_index + window_size + 1)
        after_context = " ".join(sentences[sentence_index+1:end_idx]).strip()
        
        return {
            "before_context": before_context,
            "sentence": sentence,
            "after_context": after_context
        }
    
    def determine_essay_position(self, text: str, marker_position: int) -> str:
        """Determine the position of the marker in the essay (intro, body, conclusion).
        
        Args:
            text: Full text
            marker_position: Position of the marker in the text
            
        Returns:
            Position as a string: 'introduction', 'body', or 'conclusion'
        """
        # Simple heuristic based on position in text
        text_length = len(text)
        rel_position = marker_position / text_length
        
        if rel_position < 0.15:
            return "introduction"
        elif rel_position > 0.85:
            return "conclusion"
        else:
            return "body"
    
    def detect_markers_in_text(self, text: str, corpus_source: str, essay_id: str) -> List[Dict[str, Any]]:
        """Detect all evidentiality and metadiscourse markers in a text.
        
        Args:
            text: Text to analyze
            corpus_source: Source corpus (TICLE or LOCNESS)
            essay_id: ID of the essay
            
        Returns:
            List of dictionaries with marker information
        """
        if not text or not isinstance(text, str):
            return []
        
        # Clean and normalize text
        clean_text = self.clean_text(text)
        
        # Process with spaCy for lemmatization and dependency parsing
        doc = self.nlp(clean_text)
        
        # Convert to lowercase for case-insensitive matching
        text_lower = clean_text.lower()
        
        # Store all found markers
        found_markers = []
        
        # Enhanced marker detection using dependency parsing
        def detect_evidentiality_patterns(doc):
            """Detect evidentiality patterns using dependency parsing."""
            patterns = []
            
            # 1. First-person perception verbs (I saw, I noticed, etc.)
            for token in doc:
                # Check for first-person pronouns
                if token.lower_ in ["i", "we"] and token.dep_ == "nsubj":
                    # Look for perception verbs
                    if token.head.lemma_ in ["see", "observe", "notice", "witness", "hear", "feel", "sense", "experience"]:
                        # Get the full span including any modifiers
                        start = min(token.i, token.head.i)
                        end = max(token.i, token.head.i) + 1
                        
                        # Include auxiliaries and modifiers
                        for child in token.head.children:
                            if child.dep_ in ["aux", "advmod", "neg"] and child.i not in range(start, end):
                                start = min(start, child.i)
                                end = max(end, child.i + 1)
                        
                        pattern_text = doc[start:end].text.lower()
                        category = "direct_perception"
                        subcategory = "visual" if token.head.lemma_ in ["see", "observe", "notice", "witness"] else \
                                    "auditory" if token.head.lemma_ in ["hear"] else "sensory"
                        
                        patterns.append({
                            "text": pattern_text,
                            "position": doc[start].idx,
                            "category": category,
                            "subcategory": subcategory
                        })
            
            # 2. Inference markers (must have, seems, appears, etc.)
            for token in doc:
                # Modal verbs indicating inference
                if token.lemma_ in ["must", "might", "may", "could", "would", "should"] and any(child.lemma_ == "have" for child in token.children):
                    start = token.i
                    end = token.i + 1
                    
                    # Find the 'have' token
                    for child in token.children:
                        if child.lemma_ == "have":
                            end = max(end, child.i + 1)
                    
                    pattern_text = doc[start:end].text.lower()
                    category = "inference"
                    subcategory = "deductive" if token.lemma_ == "must" else "speculative"
                    
                    patterns.append({
                        "text": pattern_text,
                        "position": doc[start].idx,
                        "category": category,
                        "subcategory": subcategory
                    })
                
                # Verbs and adverbs indicating inference
                elif token.lemma_ in ["seem", "appear", "look"] or token.lower_ in ["apparently", "seemingly", "evidently", "obviously", "clearly", "certainly"]:
                    start = token.i
                    end = token.i + 1
                    
                    # Include context for verbs
                    if token.pos_ == "VERB":
                        for child in token.children:
                            if child.dep_ in ["aux", "mark", "nsubj", "ccomp", "xcomp"]:
                                start = min(start, child.i)
                                end = max(end, child.i + 1)
                    
                    pattern_text = doc[start:end].text.lower()
                    category = "inference"
                    subcategory = "assumptive" if token.lemma_ in ["seem", "appear", "look"] else \
                                "deductive" if token.lower_ in ["obviously", "clearly", "certainly"] else "assumptive"
                    
                    patterns.append({
                        "text": pattern_text,
                        "position": doc[start].idx,
                        "category": category,
                        "subcategory": subcategory
                    })
            
            # 3. Reportative markers (according to, X claims that, etc.)
            for token in doc:
                # Preposition 'according to'
                if token.lower_ == "according" and any(child.lower_ == "to" for child in token.children):
                    start = token.i
                    end = token.i + 2  # Approximate, will refine
                    
                    # Find the 'to' token and any following noun phrase
                    for child in token.children:
                        if child.lower_ == "to":
                            end = child.i + 1
                            # Include the entity being cited
                            for grandchild in child.children:
                                if grandchild.dep_ in ["pobj"]:
                                    end = grandchild.i + 1
                                    # Include any modifiers of the cited entity
                                    for great_grandchild in grandchild.children:
                                        if great_grandchild.dep_ in ["amod", "compound"]:
                                            start = min(start, great_grandchild.i)
                                            end = max(end, great_grandchild.i + 1)
                    
                    pattern_text = doc[start:end].text.lower()
                    category = "reportative"
                    subcategory = "quotative"
                    
                    patterns.append({
                        "text": pattern_text,
                        "position": doc[start].idx,
                        "category": category,
                        "subcategory": subcategory
                    })
                
                # Reporting verbs (claims, states, argues, etc.)
                elif token.lemma_ in ["claim", "state", "argue", "suggest", "report", "mention", "note", "say", "write", "point"]:
                    # Check if followed by 'that' or has a clausal complement
                    has_clausal = False
                    for child in token.children:
                        if child.lower_ == "that" or child.dep_ in ["ccomp", "xcomp"]:
                            has_clausal = True
                            break
                    
                    if has_clausal:
                        start = token.i
                        end = token.i + 1
                        
                        # Include subject (who is reporting)
                        for child in token.children:
                            if child.dep_ == "nsubj":
                                start = min(start, child.i)
                                end = max(end, child.i + 1)
                                # Include any modifiers of the subject
                                for grandchild in child.children:
                                    if grandchild.dep_ in ["amod", "compound", "det"]:
                                        start = min(start, grandchild.i)
                                        end = max(end, grandchild.i + 1)
                            
                            # Include 'that' complementizer
                            if child.lower_ == "that":
                                end = max(end, child.i + 1)
                        
                        pattern_text = doc[start:end].text.lower()
                        category = "reportative"
                        subcategory = "quotative"
                        
                        patterns.append({
                            "text": pattern_text,
                            "position": doc[start].idx,
                            "category": category,
                            "subcategory": subcategory
                        })
            
            # 4. Knowledge/belief markers (I know, I believe, etc.)
            for token in doc:
                # First-person knowledge/belief verbs
                if token.lower_ in ["i", "we"] and token.dep_ == "nsubj":
                    if token.head.lemma_ in ["know", "believe", "think", "suppose", "assume", "guess", "suspect", "doubt", "question"]:
                        start = min(token.i, token.head.i)
                        end = max(token.i, token.head.i) + 1
                        
                        # Include auxiliaries and modifiers
                        for child in token.head.children:
                            if child.dep_ in ["aux", "advmod", "neg"] and child.i not in range(start, end):
                                start = min(start, child.i)
                                end = max(end, child.i + 1)
                        
                        pattern_text = doc[start:end].text.lower()
                        category = "knowledge_belief"
                        subcategory = "personal_knowledge" if token.head.lemma_ in ["know"] else \
                                    "doubt" if token.head.lemma_ in ["doubt", "question"] else "belief"
                        
                        patterns.append({
                            "text": pattern_text,
                            "position": doc[start].idx,
                            "category": category,
                            "subcategory": subcategory
                        })
            
            return patterns
        
        # Detect evidentiality patterns using dependency parsing
        dependency_patterns = detect_evidentiality_patterns(doc)
        
        # Add dependency-based patterns to found markers
        for pattern in dependency_patterns:
            context = self.extract_context(clean_text, pattern["position"], window_size=2)
            essay_position = self.determine_essay_position(clean_text, pattern["position"])
            
            found_markers.append({
                "marker": pattern["text"],
                "marker_category": "Evidentiality",
                "marker_subcategory": f"{pattern['category']}_{pattern['subcategory']}",
                "full_sentence": context["sentence"],
                "context_before": context["before_context"],
                "context_after": context["after_context"],
                "essay_id": essay_id,
                "essay_position": essay_position,
                "corpus_source": corpus_source,
                "position_in_text": pattern["position"]
            })
        
        # Function to check if a marker is present with proper boundaries
        def find_marker_positions(marker: str, text: str) -> List[int]:
            """Find all positions of a marker in text."""
            positions = []
            
            # For single-word markers, check if they exist as whole words
            if len(marker.split()) == 1:
                # Use regex to find whole word matches
                pattern = r'\b' + re.escape(marker) + r'\b'
                for match in re.finditer(pattern, text):
                    positions.append(match.start())
            else:
                # For multi-word phrases, allow for some flexibility
                words = marker.split()
                
                # Check for exact phrase match first
                start_pos = 0
                while True:
                    pos = text.find(marker, start_pos)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start_pos = pos + 1
                
                # Check for phrases with intervening punctuation or words (up to 2)
                if len(words) > 1 and not positions:
                    first_word_pattern = r'\b' + re.escape(words[0]) + r'\b'
                    last_word_pattern = r'\b' + re.escape(words[-1]) + r'\b'
                    
                    first_matches = list(re.finditer(first_word_pattern, text))
                    last_matches = list(re.finditer(last_word_pattern, text))
                    
                    for first_match in first_matches:
                        first_pos = first_match.start()
                        
                        for last_match in last_matches:
                            last_pos = last_match.start()
                            
                            if first_pos < last_pos:
                                # Check if the intervening text is reasonable in length
                                intervening_text = text[first_pos:last_pos]
                                word_count = len(intervening_text.split())
                                
                                # Allow for reasonable number of intervening words based on phrase length
                                max_allowed = len(words) + 2  # Original words plus 2 extra
                                
                                if word_count <= max_allowed:
                                    # Check if middle words are present in order if there are more than 2 words
                                    if len(words) > 2:
                                        middle_words_present = True
                                        for middle_word in words[1:-1]:
                                            if middle_word not in intervening_text:
                                                middle_words_present = False
                                                break
                                        if middle_words_present:
                                            positions.append(first_pos)
                                    else:
                                        positions.append(first_pos)
            
            return positions
        
        # Process evidentiality markers
        for category, subcategories in self.evidentiality_markers.items():
            for subcategory, markers in subcategories.items():
                for marker in markers:
                    positions = find_marker_positions(marker, text_lower)
                    
                    for pos in positions:
                        # Extract context
                        context = self.extract_context(clean_text, pos, window_size=2)
                        
                        # Determine essay position
                        essay_position = self.determine_essay_position(clean_text, pos)
                        
                        found_markers.append({
                            "marker": marker,
                            "marker_category": "Evidentiality",
                            "marker_subcategory": f"{category}_{subcategory}",
                            "full_sentence": context["sentence"],
                            "context_before": context["before_context"],
                            "context_after": context["after_context"],
                            "essay_id": essay_id,
                            "essay_position": essay_position,
                            "corpus_source": corpus_source,
                            "position_in_text": pos
                        })
        
        # Process mental space builders
        for category, markers in self.mental_space_builders.items():
            for marker in markers:
                positions = find_marker_positions(marker, text_lower)
                
                for pos in positions:
                    # Extract context
                    context = self.extract_context(clean_text, pos, window_size=2)
                    
                    # Determine essay position
                    essay_position = self.determine_essay_position(clean_text, pos)
                    
                    found_markers.append({
                        "marker": marker,
                        "marker_category": "Mental Space",
                        "marker_subcategory": category,
                        "full_sentence": context["sentence"],
                        "context_before": context["before_context"],
                        "context_after": context["after_context"],
                        "essay_id": essay_id,
                        "essay_position": essay_position,
                        "corpus_source": corpus_source,
                        "position_in_text": pos
                    })
        
        # Process interactive metadiscourse markers
        for category, subcategories in self.metadiscourse_markers_obj.interactive_markers.items():
            for subcategory, markers in subcategories.items():
                for marker in markers:
                    positions = find_marker_positions(marker, text_lower)
                    
                    for pos in positions:
                        # Extract context
                        context = self.extract_context(clean_text, pos, window_size=2)
                        
                        # Determine essay position
                        essay_position = self.determine_essay_position(clean_text, pos)
                        
                        found_markers.append({
                            "marker": marker,
                            "marker_category": "Interactive",
                            "marker_subcategory": f"{category}_{subcategory}",
                            "full_sentence": context["sentence"],
                            "context_before": context["before_context"],
                            "context_after": context["after_context"],
                            "essay_id": essay_id,
                            "essay_position": essay_position,
                            "corpus_source": corpus_source,
                            "position_in_text": pos
                        })
        
        # Process interactional metadiscourse markers
        for category, subcategories in self.metadiscourse_markers_obj.interactional_markers.items():
            for subcategory, markers in subcategories.items():
                for marker in markers:
                    positions = find_marker_positions(marker, text_lower)
                    
                    for pos in positions:
                        # Extract context
                        context = self.extract_context(clean_text, pos, window_size=2)
                        
                        # Determine essay position
                        essay_position = self.determine_essay_position(clean_text, pos)
                        
                        found_markers.append({
                            "marker": marker,
                            "marker_category": "Interactional",
                            "marker_subcategory": f"{category}_{subcategory}",
                            "full_sentence": context["sentence"],
                            "context_before": context["before_context"],
                            "context_after": context["after_context"],
                            "essay_id": essay_id,
                            "essay_position": essay_position,
                            "corpus_source": corpus_source,
                            "position_in_text": pos
                        })
        
        return found_markers
    
    def process_corpus(self, ticle_path: str, locness_path: str = None, text_field: str = 'text') -> pd.DataFrame:
        """Process TICLE and optionally LOCNESS corpora."""
        all_results = []
        # Process TICLE corpus
        print("Processing TICLE corpus...")
        try:
            ticle_df = pd.read_csv(ticle_path)
            if text_field not in ticle_df.columns:
                raise ValueError(f"Text field '{text_field}' not found in TICLE CSV. Available columns: {', '.join(ticle_df.columns)}")
            for idx, row in tqdm(ticle_df.iterrows(), total=len(ticle_df)):
                try:
                    text = str(row[text_field])
                    if not text or text.isspace():
                        continue
                    essay_id = f"TICLE_{idx}" if "id" not in row else f"TICLE_{row['id']}"
                    markers = self.detect_markers_in_text(text, "TICLE", essay_id)
                    all_results.extend(markers)
                except Exception as e:
                    print(f"Error processing TICLE document {idx}: {str(e)}")
                    continue
        except Exception as e:
            print(f"Error reading TICLE corpus: {str(e)}")
        # Optionally process LOCNESS corpus
        if locness_path:
            print("Processing LOCNESS corpus...")
            try:
                locness_df = pd.read_csv(locness_path)
                if text_field not in locness_df.columns:
                    raise ValueError(f"Text field '{text_field}' not found in LOCNESS CSV. Available columns: {', '.join(locness_df.columns)}")
                for idx, row in tqdm(locness_df.iterrows(), total=len(locness_df)):
                    try:
                        text = str(row[text_field])
                        if not text or text.isspace():
                            continue
                        essay_id = f"LOCNESS_{idx}" if "id" not in row else f"LOCNESS_{row['id']}"
                        markers = self.detect_markers_in_text(text, "LOCNESS", essay_id)
                        all_results.extend(markers)
                    except Exception as e:
                        print(f"Error processing LOCNESS document {idx}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error reading LOCNESS corpus: {str(e)}")
        results_df = pd.DataFrame(all_results)
        return results_df
    
    def calculate_statistics(self, results_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate statistics for the results.
        
        Args:
            results_df: DataFrame with marker extraction results
            
        Returns:
            Dictionary with various statistics DataFrames
        """
        stats = {}
        
        # Calculate word counts per corpus
        corpus_stats = {}
        
        # Group by corpus and marker category
        grouped = results_df.groupby(['corpus_source', 'marker_category', 'marker_subcategory'])
        
        # Raw frequency counts
        raw_counts = grouped.size().reset_index(name='raw_frequency')
        stats['raw_counts'] = raw_counts
        
        # Calculate normalized frequencies (per 10,000 words)
        # We need to estimate total word counts for each corpus
        # For this example, we'll use approximate values - in a real scenario, you'd calculate this
        corpus_word_counts = {
            'TICLE': 500000,  # Example value
            'LOCNESS': 500000  # Example value
        }
        
        # Function to normalize frequencies
        def normalize_frequency(row):
            corpus = row['corpus_source']
            raw_freq = row['raw_frequency']
            return (raw_freq / corpus_word_counts[corpus]) * 10000
        
        # Apply normalization
        raw_counts['normalized_frequency'] = raw_counts.apply(normalize_frequency, axis=1)
        
        # Create pivot tables for easier comparison
        pivot_raw = raw_counts.pivot_table(
            index=['marker_category', 'marker_subcategory'],
            columns='corpus_source',
            values='raw_frequency',
            fill_value=0
        )
        stats['pivot_raw'] = pivot_raw
        
        pivot_norm = raw_counts.pivot_table(
            index=['marker_category', 'marker_subcategory'],
            columns='corpus_source',
            values='normalized_frequency',
            fill_value=0
        )
        stats['pivot_norm'] = pivot_norm
        
        # Calculate ratio of frequencies between corpora
        if 'TICLE' in pivot_norm.columns and 'LOCNESS' in pivot_norm.columns:
            pivot_norm['ratio_TICLE_LOCNESS'] = pivot_norm['TICLE'] / pivot_norm['LOCNESS'].replace(0, float('nan'))
            pivot_norm['overuse'] = pivot_norm['ratio_TICLE_LOCNESS'] > 1
        
        # Distribution across essay positions
        position_dist = results_df.groupby(['corpus_source', 'essay_position', 'marker_category']).size().reset_index(name='count')
        stats['position_distribution'] = position_dist
        
        return stats
