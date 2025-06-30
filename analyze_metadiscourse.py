import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm

# Initialize spaCy
spacy.prefer_gpu()
nlp = spacy.load('en_core_web_trf')

# Marker definitions (from original notebook)
INTERACTIVE_MARKERS = {
    "transitions": [
        "moreover", "furthermore", "in addition", "additionally", "besides", "similarly", 
        "likewise", "equally", "also", "therefore", "thus", "consequently", "hence", 
        "as a result", "because", "since", "due to", "owing to", "so", "however", 
        "nevertheless", "nonetheless", "but", "yet", "though", "although", "even though", 
        "despite", "in spite of", "in contrast", "on the other hand", "conversely", 
        "meanwhile", "simultaneously", "subsequently", "previously", "after", "before", 
        "then", "later", "formerly", "eventually"
    ],
    "frame_markers": [
        "first", "firstly", "second", "secondly", "third", "thirdly", "fourth", "finally", 
        "lastly", "to begin with", "to start with", "next", "then", "subsequently",
        "in conclusion", "to conclude", "to summarize", "in summary", "in brief", "all in all", 
        "on the whole", "so far", "at this point", "overall", "aim", "purpose", "goal", 
        "objective", "focus", "seek to", "intend to", "with regard to", "concerning", 
        "regarding", "turning to", "moving on to", "back to"
    ],
    "endophoric_markers": [
        "in chapter", "in section", "in part", "in figure", "in table", "figure", "table", 
        "above", "below", "earlier", "previously", "as noted above", "as mentioned earlier", 
        "see", "refer to", "page", "the following", "as follows", "aforementioned"
    ],
    "evidentials": [
        "according to", "cited", "quoted", "states that", "argues that", "notes that", 
        "suggests that", "reports that", "found that", "observed that", "concluded that", 
        "in the literature", "previous research", "research shows", "studies indicate"
    ],
    "code_glosses": [
        "in other words", "that is", "i.e.", "that is to say", "this means", "in simple terms",
        "put simply", "to put it simply", "namely", "for example", "for instance", "such as", 
        "e.g.", "specifically", "particularly", "in fact", "indeed", "actually", "called", 
        "defined as", "referred to as", "including", "included", "especially", "notably"
    ]
}

INTERACTIONAL_MARKERS = {
    "hedges": [
        "may", "might", "could", "would", "perhaps", "possibly", "probably", "maybe", "likely", 
        "seemingly", "apparently", "approximately", "about", "roughly", "suggest", "assume", 
        "believe", "think", "appear", "seem", "indicate", "suspect", "suppose", "estimate", 
        "in my opinion", "from my perspective", "to my knowledge", "generally", "usually", 
        "sometimes", "often", "in most cases", "to some extent", "sort of", "kind of"
    ],
    "boosters": [
        "clearly", "obviously", "certainly", "definitely", "undoubtedly", "undeniably", 
        "demonstrate", "prove", "show", "establish", "confirm", "find", "reveal", "must", 
        "will", "beyond doubt", "without doubt", "in fact", "indeed", "actually", "always", 
        "never", "absolutely", "completely", "entirely", "truly", "really"
    ],
    "attitude_markers": [
        "unfortunately", "fortunately", "surprisingly", "remarkably", "interestingly", 
        "hopefully", "importantly", "significantly", "correctly", "appropriately", "agree", 
        "prefer", "disagree", "dramatic", "unexpected", "desirable", "disappointing", "alarming"
    ],
    "engagement_markers": [
        "you", "your", "yours", "yourself", "consider", "note", "imagine", "think about", 
        "let us", "let's", "see", "must", "should", "need to", "have to", "ought to", 
        "what about", "how about", "by the way", "the reader", "readers"
    ],
    "self_mentions": [
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves", 
        "the author", "the authors", "the researcher", "the researchers", "this author"
    ]
}

def initialize_matcher():
    """Initialize spaCy matcher with patterns."""
    matcher = Matcher(nlp.vocab)
    
    # Add patterns for each marker category
    for category, markers in INTERACTIVE_MARKERS.items():
        for marker in markers:
            pattern = [{"LOWER": word} for word in marker.split()]
            matcher.add(f"interactive_{category}", [pattern])
    
    for category, markers in INTERACTIONAL_MARKERS.items():
        for marker in markers:
            pattern = [{"LOWER": word} for word in marker.split()]
            matcher.add(f"interactional_{category}", [pattern])
    
    return matcher

def process_text(text: str, matcher: Matcher):
    """Process a single text and return marker counts."""
    doc = nlp(text)
    matches = matcher(doc)
    
    # Initialize counts
    counts = {
        "interactive": {cat: 0 for cat in INTERACTIVE_MARKERS},
        "interactional": {cat: 0 for cat in INTERACTIONAL_MARKERS}
    }
    
    # Count matches
    for match_id, start, end in matches:
        category = nlp.vocab.strings[match_id]
        marker_type, marker_cat = category.split("_", 1)  # Split only on first underscore
        if marker_type == "interactive":
            counts["interactive"][marker_cat] += 1
        elif marker_type == "interactional":
            counts["interactional"][marker_cat] += 1
    
    # Calculate word count
    word_count = len([token for token in doc if not token.is_punct])
    
    # Calculate frequencies per 1000 words
    frequencies = {
        "interactive": {
            cat: (count / word_count) * 1000 
            for cat, count in counts["interactive"].items()
        },
        "interactional": {
            cat: (count / word_count) * 1000 
            for cat, count in counts["interactional"].items()
        }
    }
    
    return counts, frequencies, word_count

def main():
    # Initialize matcher
    matcher = initialize_matcher()
    
    # Read CSV file
    print("Reading data...")
    df = pd.read_csv("data/metadata_with_text.csv")
    
    # Process texts
    results = []
    print("Processing texts...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            counts, frequencies, word_count = process_text(str(row['text_field']), matcher)
            
            result = {
                'document': f"doc_{idx}",
                'word_count': word_count
            }
            
            # Add metadata
            for col in df.columns:
                if col != 'text_field':
                    result[col] = row[col]
            
            # Add counts and frequencies
            for category, count in counts["interactive"].items():
                result[f'interactive_{category}_count'] = count
                result[f'interactive_{category}_freq'] = frequencies["interactive"][category]
            
            for category, count in counts["interactional"].items():
                result[f'interactional_{category}_count'] = count
                result[f'interactional_{category}_freq'] = frequencies["interactional"][category]
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing document {idx}: {str(e)}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Save results
    results_df.to_csv("results/metadiscourse_analysis.csv", index=False)
    
    # Create basic visualizations
    print("Creating visualizations...")
    
    # Distribution of marker frequencies
    plt.figure(figsize=(15, 10))
    freq_cols = [col for col in results_df.columns if col.endswith('_freq')]
    results_df[freq_cols].boxplot()
    plt.xticks(rotation=45, ha='right')
    plt.title("Distribution of Marker Frequencies")
    plt.tight_layout()
    plt.savefig("results/marker_distributions.png")
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(results_df[freq_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation between Marker Frequencies")
    plt.tight_layout()
    plt.savefig("results/marker_correlations.png")
    plt.close()
    
    print("Analysis complete! Results saved in the 'results' directory.")

if __name__ == "__main__":
    main() 