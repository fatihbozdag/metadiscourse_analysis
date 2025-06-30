from typing import Dict, List, Set
from spacy.matcher import Matcher
from spacy.tokens import Doc

"""Enhanced metadiscourse marker definitions based on Hyland's scheme with improved accuracy."""

INTERACTIVE_MARKERS = {
    "transitions": [
        # Additive
        "moreover", "furthermore", "in addition", "additionally", "besides", "similarly", 
        "likewise", "equally", "also", "further", "what is more", "apart from this",
        # Causal
        "therefore", "thus", "consequently", "hence", "as a result", "because", "since", 
        "due to", "owing to", "so", "for this reason", "accordingly", "as a consequence",
        # Adversative
        "however", "nevertheless", "nonetheless", "but", "yet", "though", "although", "even though", 
        "despite", "in spite of", "in contrast", "on the other hand", "conversely", "by contrast",
        "on the contrary", "rather", "instead", "alternatively", "whereas", "while",
        # Temporal
        "meanwhile", "simultaneously", "subsequently", "previously", "after", "before", 
        "then", "later", "formerly", "eventually", "finally", "initially", "originally"
    ],
    "frame_markers": [
        # Sequencing
        "first", "firstly", "second", "secondly", "third", "thirdly", "fourth", "finally", 
        "lastly", "to begin with", "to start with", "next", "then", "subsequently",
        # Topic shift
        "turning to", "moving on to", "back to", "with regard to", "concerning", 
        "regarding", "as for", "as to", "in terms of", "with respect to",
        # Conclusion
        "in conclusion", "to conclude", "to summarize", "in summary", "in brief", "all in all", 
        "on the whole", "so far", "at this point", "overall", "to sum up", "in short",
        # Purpose
        "aim", "purpose", "goal", "objective", "focus", "seek to", "intend to", "in order to"
    ],
    "endophoric_markers": [
        # Internal references
        "in chapter", "in section", "in part", "in figure", "in table", "figure", "table", 
        "above", "below", "earlier", "previously", "as noted above", "as mentioned earlier", 
        "see", "refer to", "page", "the following", "as follows", "aforementioned",
        "as discussed", "as shown", "as indicated", "as outlined", "as presented"
    ],
    "evidentials": [
        # Citation markers
        "according to", "cited", "quoted", "states that", "argues that", "notes that", 
        "suggests that", "reports that", "found that", "observed that", "concluded that", 
        "claims that", "proposes that", "maintains that", "asserts that", "demonstrates that",
        # Research references
        "in the literature", "previous research", "research shows", "studies indicate",
        "evidence suggests", "data show", "findings reveal", "results indicate"
    ],
    "code_glosses": [
        # Reformulation
        "in other words", "that is", "i.e.", "that is to say", "this means", "in simple terms",
        "put simply", "to put it simply", "namely", "or rather", "more precisely",
        # Exemplification
        "for example", "for instance", "such as", "e.g.", "specifically", "particularly", 
        "including", "included", "especially", "notably", "like", "as in the case of",
        # Clarification
        "called", "defined as", "referred to as", "known as", "termed"
    ]
}

INTERACTIONAL_MARKERS = {
    "hedges": [
        # Modal hedges (removed must/should to avoid overlap)
        "may", "might", "could", "would", "perhaps", "possibly", "probably", "maybe", "likely", 
        "seemingly", "apparently", "presumably", "conceivably", "potentially",
        # Approximation
        "approximately", "about", "roughly", "around", "nearly", "almost", "virtually",
        # Tentative verbs
        "suggest", "assume", "believe", "think", "appear", "seem", "indicate", "suspect", 
        "suppose", "estimate", "tend to", "incline to",
        # Epistemic phrases
        "in my opinion", "from my perspective", "to my knowledge", "it seems that", "it appears that",
        # Frequency hedges
        "generally", "usually", "sometimes", "often", "in most cases", "to some extent", 
        "sort of", "kind of", "more or less"
    ],
    "boosters": [
        # Certainty markers
        "clearly", "obviously", "certainly", "definitely", "undoubtedly", "undeniably", 
        "without doubt", "beyond doubt", "no doubt", "surely", "of course",
        # Emphatic verbs
        "demonstrate", "prove", "show", "establish", "confirm", "find", "reveal", 
        "determine", "verify", "validate", "substantiate",
        # Intensifiers
        "will", "always", "never", "absolutely", "completely", "entirely", "truly", "really",
        "indeed", "in fact", "actually" # These can also be code glosses (polyfunctional)
    ],
    "attitude_markers": [
        # Evaluative adverbs
        "unfortunately", "fortunately", "surprisingly", "remarkably", "interestingly", 
        "hopefully", "importantly", "significantly", "regrettably", "disappointingly",
        # Judgment markers
        "correctly", "appropriately", "understandably", "predictably", "inevitably",
        # Emotional responses
        "agree", "prefer", "disagree", "admire", "appreciate", "welcome",
        # Evaluative adjectives in context
        "dramatic", "unexpected", "desirable", "disappointing", "alarming", "striking",
        "remarkable", "notable", "crucial", "essential", "vital"
    ],
    "engagement_markers": [
        # Direct address (removed must/should to avoid hedge overlap)
        "you", "your", "yours", "yourself", "yourselves",
        # Imperatives
        "consider", "note", "imagine", "think about", "suppose", "assume", "remember",
        "bear in mind", "keep in mind", "notice", "observe", "see",
        # Inclusive pronouns
        "let us", "let's", "we", "us", "our" # when used inclusively
        # Questions to readers
        "what about", "how about", "why not",
        # Reader references
        "the reader", "readers", "one", "anyone", "everyone"
    ],
    "self_mentions": [
        # Organizational self-mentions
        "i argue", "i suggest", "i propose", "i claim", "i contend", "i maintain",
        "we argue", "we suggest", "we propose", "we claim", "we contend", "we maintain",
        # Methodological self-mentions  
        "i analyze", "i examine", "i investigate", "i study", "i explore", "i test",
        "we analyze", "we examine", "we investigate", "we study", "we explore", "we test",
        # Basic pronouns (context-dependent)
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves", 
        # Authorial references
        "the author", "the authors", "the researcher", "the researchers", "this author",
        "the present author", "the current study", "this study"
    ]
}

class EnhancedMetadiscourseMarkers:
    """Enhanced hierarchical structure for metadiscourse markers with improved accuracy."""
    
    def __init__(self):
        self.interactive_markers = {
            "transitions": {
                "additive": ["moreover", "furthermore", "in addition", "additionally", "besides", 
                           "similarly", "likewise", "equally", "also", "further", "what is more"],
                "causal": ["therefore", "thus", "consequently", "hence", "as a result", 
                          "because", "since", "due to", "owing to", "so", "for this reason"],
                "adversative": ["however", "nevertheless", "nonetheless", "but", "yet", 
                              "though", "although", "even though", "despite", "in spite of", 
                              "in contrast", "on the other hand", "conversely", "by contrast"],
                "temporal": ["meanwhile", "simultaneously", "subsequently", "previously", 
                           "after", "before", "then", "later", "formerly", "eventually"]
            },
            "frame_markers": {
                "sequencing": ["first", "firstly", "second", "secondly", "third", "thirdly", 
                             "fourth", "finally", "lastly", "to begin with", "to start with", 
                             "next", "subsequently"],
                "topic_shift": ["turning to", "moving on to", "back to", "with regard to", 
                              "concerning", "regarding", "as for", "in terms of"],
                "conclusion": ["in conclusion", "to conclude", "to summarize", "in summary", 
                             "in brief", "all in all", "on the whole", "overall"],
                "purpose": ["aim", "purpose", "goal", "objective", "focus", "seek to", 
                          "intend to", "in order to"]
            },
            "endophoric_markers": {
                "internal_reference": ["in chapter", "in section", "in part", "in figure", 
                                     "in table", "figure", "table", "above", "below"],
                "retrospective": ["earlier", "previously", "as noted above", "as mentioned earlier",
                                "as discussed", "as shown", "as indicated"],
                "prospective": ["see", "refer to", "the following", "as follows"]
            },
            "evidentials": {
                "attribution": ["according to", "states that", "argues that", "notes that", 
                              "suggests that", "reports that", "claims that", "maintains that"],
                "research_reference": ["found that", "observed that", "concluded that", 
                                     "demonstrated that", "showed that", "revealed that"],
                "general_reference": ["in the literature", "previous research", "research shows", 
                                    "studies indicate", "evidence suggests"]
            },
            "code_glosses": {
                "reformulation": ["in other words", "that is", "i.e.", "that is to say", 
                                "this means", "in simple terms", "put simply", "namely"],
                "exemplification": ["for example", "for instance", "such as", "e.g.", 
                                  "specifically", "particularly", "including", "notably"],
                "definition": ["called", "defined as", "referred to as", "known as", "termed"]
            }
        }
        
        self.interactional_markers = {
            "hedges": {
                "modal": ["may", "might", "could", "would", "perhaps", "possibly", 
                         "probably", "maybe", "likely"],
                "approximation": ["approximately", "about", "roughly", "around", "nearly"],
                "epistemic": ["seemingly", "apparently", "presumably", "it seems", "it appears"],
                "tentative": ["suggest", "assume", "believe", "think", "appear", "seem", 
                            "indicate", "tend to"],
                "personal": ["in my opinion", "from my perspective", "to my knowledge"],
                "frequency": ["generally", "usually", "sometimes", "often", "in most cases"]
            },
            "boosters": {
                "certainty": ["clearly", "obviously", "certainly", "definitely", 
                            "undoubtedly", "undeniably", "without doubt"],
                "demonstration": ["demonstrate", "prove", "show", "establish", "confirm", 
                                "find", "reveal", "determine"],
                "emphasis": ["will", "indeed", "in fact", "actually"],
                "absoluteness": ["always", "never", "absolutely", "completely", "entirely"]
            },
            "attitude_markers": {
                "evaluation": ["unfortunately", "fortunately", "surprisingly", "remarkably", 
                             "interestingly", "importantly", "significantly"],
                "judgment": ["correctly", "appropriately", "understandably", "predictably"],
                "emotion": ["agree", "prefer", "disagree", "dramatic", "unexpected", 
                          "disappointing", "alarming"]
            },
            "engagement_markers": {
                "direct_address": ["you", "your", "yours", "yourself"],
                "imperatives": ["consider", "note", "imagine", "think about", "suppose", 
                              "remember", "bear in mind", "notice"],
                "inclusive": ["let us", "let's"],
                "questions": ["what about", "how about", "why not"],
                "reader_reference": ["the reader", "readers", "one"]
            },
            "self_mentions": {
                "organizational": ["i argue", "i suggest", "i propose", "we argue", "we suggest",
                                 "i discuss", "i present", "we discuss", "we present"],
                "methodological": ["i analyze", "i examine", "i investigate", "we analyze", 
                                 "we examine", "i test", "we test"],
                "stance": ["i believe", "i think", "i feel", "we believe", "we think"],
                "basic": ["i", "me", "my", "we", "us", "our"],
                "authorial": ["the author", "the authors", "the researcher", "this study"]
            }
        }
        
        # Enhanced polyfunctional markers with confidence scores
        self.polyfunctional_markers = {
            # High confidence polyfunctional markers
            "in fact": [
                ("interactive", "code_glosses", "exemplification", 0.7),
                ("interactional", "boosters", "emphasis", 0.8)
            ],
            "indeed": [
                ("interactive", "code_glosses", "exemplification", 0.6),
                ("interactional", "boosters", "emphasis", 0.9)
            ],
            "actually": [
                ("interactive", "code_glosses", "exemplification", 0.5),
                ("interactional", "boosters", "emphasis", 0.8)
            ],
            # Temporal/sequential overlap
            "then": [
                ("interactive", "transitions", "temporal", 0.8),
                ("interactive", "frame_markers", "sequencing", 0.6)
            ],
            "next": [
                ("interactive", "transitions", "temporal", 0.7),
                ("interactive", "frame_markers", "sequencing", 0.8)
            ],
            "finally": [
                ("interactive", "transitions", "temporal", 0.6),
                ("interactive", "frame_markers", "conclusion", 0.9)
            ]
        }
        
        # Context-sensitive exclusions
        self.context_exclusions = {
            "must": ["hedge_context"],  # Only hedge when expressing uncertainty
            "should": ["hedge_context"], # Only hedge when expressing uncertainty  
            "will": ["future_tense_context"], # Exclude when expressing future time
            "may": ["permission_context"], # Exclude when expressing permission
            "can": ["ability_context"] # Exclude when expressing ability
        }
        
        self.matcher = None
    
    def initialize_matcher(self, nlp):
        """Initialize spaCy matcher with all patterns."""
        self.matcher = Matcher(nlp.vocab)
        
        # Add patterns for each marker category with more flexible matching
        for category, subcategories in self.interactive_markers.items():
            for subcategory, markers in subcategories.items():
                for marker in markers:
                    # Create basic pattern
                    words = marker.split()
                    
                    # For single word markers, use simple matching
                    if len(words) == 1:
                        pattern = [{"LOWER": words[0]}]
                        self.matcher.add(f"{category}_{subcategory}", [pattern])
                    else:
                        # For multi-word markers, create patterns with optional intervening tokens
                        # Basic pattern - exact match
                        basic_pattern = [{"LOWER": word} for word in words]
                        
                        # Pattern with optional intervening tokens (up to 2) for longer phrases
                        if len(words) > 2:
                            flexible_pattern = [
                                {"LOWER": words[0]}
                            ]
                            
                            # Add middle words with optional intervening tokens
                            for i in range(1, len(words) - 1):
                                # Allow up to 2 optional words between marker words
                                flexible_pattern.extend([
                                    {"OP": "?", "IS_PUNCT": True},  # Optional punctuation
                                    {"OP": "?", "IS_ALPHA": True},   # Optional word
                                    {"OP": "?", "IS_PUNCT": True},  # Optional punctuation
                                    {"LOWER": words[i]}
                                ])
                            
                            # Add last word
                            flexible_pattern.extend([
                                {"OP": "?", "IS_PUNCT": True},  # Optional punctuation
                                {"OP": "?", "IS_ALPHA": True},   # Optional word
                                {"OP": "?", "IS_PUNCT": True},  # Optional punctuation
                                {"LOWER": words[-1]}
                            ])
                            
                            # Add both patterns
                            self.matcher.add(f"{category}_{subcategory}", [basic_pattern, flexible_pattern])
                        else:
                            # For two-word phrases, use simpler pattern with optional punctuation
                            punct_pattern = [
                                {"LOWER": words[0]},
                                {"OP": "?", "IS_PUNCT": True},
                                {"LOWER": words[1]}
                            ]
                            self.matcher.add(f"{category}_{subcategory}", [basic_pattern, punct_pattern])
        
        # Same for interactional markers
        for category, subcategories in self.interactional_markers.items():
            for subcategory, markers in subcategories.items():
                for marker in markers:
                    # Create basic pattern
                    words = marker.split()
                    
                    # For single word markers, use simple matching
                    if len(words) == 1:
                        pattern = [{"LOWER": words[0]}]
                        self.matcher.add(f"{category}_{subcategory}", [pattern])
                    else:
                        # For multi-word markers, create patterns with optional intervening tokens
                        # Basic pattern - exact match
                        basic_pattern = [{"LOWER": word} for word in words]
                        
                        # Pattern with optional intervening tokens (up to 2) for longer phrases
                        if len(words) > 2:
                            flexible_pattern = [
                                {"LOWER": words[0]}
                            ]
                            
                            # Add middle words with optional intervening tokens
                            for i in range(1, len(words) - 1):
                                # Allow up to 2 optional words between marker words
                                flexible_pattern.extend([
                                    {"OP": "?", "IS_PUNCT": True},  # Optional punctuation
                                    {"OP": "?", "IS_ALPHA": True},   # Optional word
                                    {"OP": "?", "IS_PUNCT": True},  # Optional punctuation
                                    {"LOWER": words[i]}
                                ])
                            
                            # Add last word
                            flexible_pattern.extend([
                                {"OP": "?", "IS_PUNCT": True},  # Optional punctuation
                                {"OP": "?", "IS_ALPHA": True},   # Optional word
                                {"OP": "?", "IS_PUNCT": True},  # Optional punctuation
                                {"LOWER": words[-1]}
                            ])
                            
                            # Add both patterns
                            self.matcher.add(f"{category}_{subcategory}", [basic_pattern, flexible_pattern])
                        else:
                            # For two-word phrases, use simpler pattern with optional punctuation
                            punct_pattern = [
                                {"LOWER": words[0]},
                                {"OP": "?", "IS_PUNCT": True},
                                {"LOWER": words[1]}
                            ]
                            self.matcher.add(f"{category}_{subcategory}", [basic_pattern, punct_pattern])
    
    def find_markers(self, doc: Doc) -> Dict[str, Dict[str, List[str]]]:
        """Find all metadiscourse markers in a document."""
        if self.matcher is None:
            raise ValueError("Matcher not initialized. Call initialize_matcher first.")
        
        # Initialize results structure with all categories and subcategories
        results = {
            "interactive": {
                "transitions": {"additive": [], "causal": [], "adversative": [], "temporal": []},
                "frame_markers": {"sequencing": [], "conclusion": [], "topic": []},
                "endophoric_markers": {"reference": [], "citation": []},
                "evidentials": {"citation": [], "research": []},
                "code_glosses": {"reformulation": [], "exemplification": []}
            },
            "interactional": {
                "hedges": {"modality": [], "probability": [], "approximation": []},
                "boosters": {"certainty": [], "emphasis": [], "absoluteness": []},
                "attitude_markers": {"evaluation": [], "judgment": [], "emotion": []},
                "engagement_markers": {"direct_address": [], "imperatives": [], "inclusive": [], "questions": [], "reader_reference": []},
                "self_mentions": {"singular": [], "plural": [], "authorial": []}
            }
        }
        
        # Find matches
        matches = self.matcher(doc)
        
        # Process matches
        for match_id, start, end in matches:
            try:
                marker_text = doc[start:end].text
                parts = doc.vocab.strings[match_id].split("_")
                
                if len(parts) < 2:
                    print(f"Warning: Invalid match ID format: {doc.vocab.strings[match_id]}")
                    continue
                    
                category = parts[0]
                subcategory = "_".join(parts[1:])
                
                # Handle interactive markers
                if category in self.interactive_markers:
                    # Find the main category (transitions, frame_markers, etc.)
                    for main_cat, subcats in self.interactive_markers.items():
                        if subcategory in subcats or subcategory == main_cat:
                            # If subcategory is found in this main category or matches the main category
                            if subcategory in subcats:
                                results["interactive"][main_cat][subcategory].append(marker_text)
                            else:
                                # Default to first subcategory if subcategory matches main category
                                first_subcat = list(results["interactive"][main_cat].keys())[0]
                                results["interactive"][main_cat][first_subcat].append(marker_text)
                            break
                # Handle interactional markers
                elif category in self.interactional_markers:
                    # Find the main category (hedges, boosters, etc.)
                    for main_cat, subcats in self.interactional_markers.items():
                        if subcategory in subcats or subcategory == main_cat:
                            # If subcategory is found in this main category or matches the main category
                            if subcategory in subcats:
                                results["interactional"][main_cat][subcategory].append(marker_text)
                            else:
                                # Default to first subcategory if subcategory matches main category
                                first_subcat = list(results["interactional"][main_cat].keys())[0]
                                results["interactional"][main_cat][first_subcat].append(marker_text)
                            break
            except Exception as e:
                print(f"Error processing match: {str(e)}")
                continue
        
        # Flatten the results structure to match the expected format
        flattened_results = {
            "interactive": {},
            "interactional": {}
        }
        
        # Flatten interactive categories
        for main_cat, subcats in results["interactive"].items():
            flattened_results["interactive"][main_cat] = {}
            for subcat, markers in subcats.items():
                flattened_results["interactive"][main_cat][subcat] = markers
        
        # Flatten interactional categories
        for main_cat, subcats in results["interactional"].items():
            flattened_results["interactional"][main_cat] = {}
            for subcat, markers in subcats.items():
                flattened_results["interactional"][main_cat][subcat] = markers
        
        return flattened_results
    
    def get_marker_counts(self, doc: Doc) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Get counts of all metadiscourse markers in a document.
        
        This method handles polyfunctional markers by counting them in all relevant categories.
        """
        # First, find all markers using the standard approach
        results = self.find_markers(doc)
        
        # Initialize counts with the same structure as results
        counts = {
            "interactive": {},
            "interactional": {}
        }
        
        # Count interactive markers
        for category, subcategories in results["interactive"].items():
            counts["interactive"][category] = {}
            for subcategory, markers in subcategories.items():
                counts["interactive"][category][subcategory] = len(markers)
        
        # Count interactional markers
        for category, subcategories in results["interactional"].items():
            counts["interactional"][category] = {}
            for subcategory, markers in subcategories.items():
                counts["interactional"][category][subcategory] = len(markers)
        
        # Now handle polyfunctional markers
        try:
            polyfunctional_matches = self.find_polyfunctional_markers(doc)
            
            # Add polyfunctional matches to the counts
            for marker_text, categories in polyfunctional_matches.items():
                for marker_type, category, subcategory in categories:
                    try:
                        # Ensure the category and subcategory exist in the counts
                        if marker_type in counts and category in counts[marker_type]:
                            if subcategory in counts[marker_type][category]:
                                counts[marker_type][category][subcategory] += 1
                    except Exception as e:
                        print(f"Error processing polyfunctional marker {marker_text}: {str(e)}")
        except Exception as e:
            print(f"Error processing polyfunctional markers: {str(e)}")
        
        return counts
        
    def find_polyfunctional_markers(self, doc: Doc) -> Dict[str, List[tuple]]:
        """Find markers that can belong to multiple categories.
        
        Returns:
            Dictionary mapping marker text to list of (type, category, subcategory) tuples
        """
        results = {}
        
        # Check for each polyfunctional marker in the document
        for marker, categories in self.polyfunctional_markers.items():
            words = marker.split()
            
            # For single-word markers
            if len(words) == 1:
                for token in doc:
                    if token.text.lower() == words[0]:
                        if marker not in results:
                            results[marker] = categories
            # For multi-word markers
            else:
                # Simple n-gram matching for multi-word markers
                text_lower = doc.text.lower()
                if marker in text_lower:
                    if marker not in results:
                        results[marker] = categories
        
        return results

    def get_confidence_weighted_markers(self, confidence_threshold=0.7):
        """Get markers above confidence threshold for polyfunctional cases."""
        filtered_markers = {}
        for marker, functions in self.polyfunctional_markers.items():
            filtered_functions = [f for f in functions if f[3] >= confidence_threshold]
            if filtered_functions:
                filtered_markers[marker] = filtered_functions
        return filtered_markers