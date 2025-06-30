"""Evidentiality marker definitions for linguistic analysis."""

from typing import Dict, List

# Evidentiality markers categorized by type
EVIDENTIALITY_MARKERS = {
    "direct_perception": {
        "visual": [
            "i saw", "i see", "i observed", "i noticed", "i witnessed", 
            "i can see", "i have seen", "i could see", "i am seeing",
            "we saw", "we see", "we observed", "we noticed", "we witnessed",
            "we can see", "we have seen", "we could see", "we are seeing",
            "visible", "visibly", "visually", "observable", "evident"
        ],
        "auditory": [
            "i heard", "i hear", "i am hearing", "i have heard", "i could hear",
            "we heard", "we hear", "we are hearing", "we have heard", "we could hear",
            "audible", "audibly", "sound like", "sounds like", "sounded like"
        ],
        "sensory": [
            "i felt", "i feel", "i am feeling", "i have felt", "i could feel",
            "we felt", "we feel", "we are feeling", "we have felt", "we could feel",
            "i experienced", "we experienced", "i sensed", "we sensed"
        ]
    },
    "inference": {
        "deductive": [
            "must have", "must be", "certainly", "definitely", "obviously",
            "clearly", "undoubtedly", "necessarily", "inevitably", "unmistakably"
        ],
        "assumptive": [
            "seems", "appears", "looks like", "presumably", "apparently",
            "seemingly", "ostensibly", "evidently", "it seems that", "it appears that",
            "it looks like", "it would seem", "it would appear"
        ],
        "speculative": [
            "might have", "may have", "could have", "possibly", "perhaps",
            "probably", "likely", "conceivably", "plausibly", "it is possible that",
            "it is probable that", "it is likely that"
        ]
    },
    "reportative": {
        "quotative": [
            "according to", "said that", "says that", "stated that", "states that",
            "claimed that", "claims that", "reported that", "reports that",
            "mentioned that", "mentions that", "noted that", "notes that",
            "argued that", "argues that", "suggested that", "suggests that",
            "wrote that", "writes that", "pointed out that", "points out that"
        ],
        "hearsay": [
            "reportedly", "allegedly", "supposedly", "reputedly", "purportedly",
            "it is said that", "it is reported that", "it is claimed that",
            "it is alleged that", "it is rumored that", "word has it that",
            "they say that", "people say that", "some say that", "it has been said that"
        ],
        "citation": [
            "cited in", "as cited in", "referenced in", "as referenced in",
            "as shown in", "as demonstrated in", "as illustrated in",
            "as presented in", "as discussed in", "as analyzed in",
            "as explained in", "as described in", "as mentioned in"
        ]
    },
    "knowledge_belief": {
        "personal_knowledge": [
            "i know", "i knew", "i am aware", "i am certain", "i am sure",
            "i am convinced", "i am confident", "i have no doubt",
            "we know", "we knew", "we are aware", "we are certain", "we are sure",
            "we are convinced", "we are confident", "we have no doubt",
            "to my knowledge", "as far as i know", "to the best of my knowledge"
        ],
        "belief": [
            "i believe", "i think", "i suppose", "i assume", "i guess",
            "i suspect", "i presume", "i imagine", "i reckon", "i consider",
            "we believe", "we think", "we suppose", "we assume", "we guess",
            "we suspect", "we presume", "we imagine", "we reckon", "we consider",
            "in my opinion", "in my view", "from my perspective", "in our opinion",
            "in our view", "from our perspective"
        ],
        "doubt": [
            "i doubt", "i question", "i am skeptical", "i am doubtful",
            "i am uncertain", "i am not sure", "i am not convinced",
            "we doubt", "we question", "we are skeptical", "we are doubtful",
            "we are uncertain", "we are not sure", "we are not convinced",
            "it is doubtful that", "it is questionable whether"
        ]
    }
}

# Mental space builders categorized by type
MENTAL_SPACE_BUILDERS = {
    "belief_spaces": [
        "i think", "i believe", "i suppose", "i assume", "i imagine",
        "we think", "we believe", "we suppose", "we assume", "we imagine",
        "in my opinion", "in my view", "in our opinion", "in our view"
    ],
    "speech_spaces": [
        "he said", "she said", "they said", "according to", "as stated by",
        "as mentioned by", "as noted by", "as claimed by", "as argued by",
        "as suggested by", "as reported by", "as pointed out by"
    ],
    "hypothetical_spaces": [
        "if", "would", "could", "might", "may", "imagine", "let's say",
        "suppose", "assuming that", "in case", "provided that", "unless",
        "as if", "as though", "hypothetically", "theoretically", "in theory"
    ],
    "time_place_spaces": [
        "in 2010", "in 2011", "in 2012", "in 2013", "in 2014", "in 2015",
        "in 2016", "in 2017", "in 2018", "in 2019", "in 2020", "in 2021",
        "in 2022", "in 2023", "in 2024", "in 2025",
        "in january", "in february", "in march", "in april", "in may", "in june",
        "in july", "in august", "in september", "in october", "in november", "in december",
        "in turkey", "in england", "in the uk", "in the united kingdom", "in the us",
        "in the united states", "in america", "in europe", "in asia", "in africa",
        "in australia", "in canada", "in germany", "in france", "in italy", "in spain",
        "in china", "in japan", "in russia", "in brazil", "in india", "in mexico"
    ],
    "possibility_probability_spaces": [
        "maybe", "perhaps", "possibly", "probably", "likely", "unlikely",
        "it is possible that", "it is probable that", "it is likely that",
        "it is unlikely that", "there is a chance that", "there is a possibility that",
        "chances are", "the odds are", "in all probability", "in all likelihood"
    ]
}

class EvidentialityMarkers:
    """Hierarchical structure for evidentiality markers."""
    
    def __init__(self):
        self.evidentiality_markers = EVIDENTIALITY_MARKERS
        self.mental_space_builders = MENTAL_SPACE_BUILDERS
        
    def get_all_evidentiality_markers(self) -> List[str]:
        """Return a flat list of all evidentiality markers."""
        all_markers = []
        for category, subcategories in self.evidentiality_markers.items():
            for subcategory, markers in subcategories.items():
                all_markers.extend(markers)
        return all_markers
    
    def get_all_mental_space_builders(self) -> List[str]:
        """Return a flat list of all mental space builders."""
        all_builders = []
        for category, markers in self.mental_space_builders.items():
            all_builders.extend(markers)
        return all_builders
    
    def get_category_for_marker(self, marker: str) -> Dict[str, str]:
        """Return the category and subcategory for a given marker."""
        marker = marker.lower()
        
        # Check evidentiality markers
        for category, subcategories in self.evidentiality_markers.items():
            for subcategory, markers in subcategories.items():
                if marker in markers:
                    return {
                        "main_category": "evidentiality",
                        "category": category,
                        "subcategory": subcategory
                    }
        
        # Check mental space builders
        for category, markers in self.mental_space_builders.items():
            if marker in markers:
                return {
                    "main_category": "mental_space",
                    "category": category,
                    "subcategory": None
                }
        
        return None
