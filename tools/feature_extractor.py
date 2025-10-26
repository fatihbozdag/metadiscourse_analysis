import pandas as pd
import spacy
from spacy import cli
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse

# Assuming MetadiscourseAnalyzer is in precision_analyzer.py
# For simplicity, I'll include a simplified version here for feature extraction
# In a real project, you would import it: from precision_analyzer import MetadiscourseAnalyzer

class SimplifiedMetadiscourseAnalyzer:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy model 'en_core_web_sm' not found. Downloading...")
            cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.academic_keywords = [
            'research', 'study', 'analysis', 'theory', 'evidence', 'data',
            'findings', 'results', 'conclusion', 'argument', 'hypothesis',
            'methodology', 'framework', 'literature', 'investigation',
            'examination', 'discussion', 'exploration', 'demonstrate',
            'establish', 'indicate', 'suggest', 'reveal', 'show',
            'academic', 'scholarly', 'empirical', 'theoretical', 'analytical',
            'argue', 'claim', 'suggest', 'propose', 'conclude', 'demonstrate',
            'scholar', 'researcher', 'scientist', 'professor', 'expert',
            'publication', 'journal', 'conference', 'paper'
        ]

        self.conversational_keywords = [
            'i went', 'we went', 'i saw', 'we saw', 'my family', 'our family',
            'my friends', 'our friends', 'at home', 'at school', 'yesterday', 'tomorrow',
            'last week', 'next week', 'my mother', 'my father',
            'go', 'come', 'see', 'do', 'have', 'be', 'like', 'love', 'want', 'need'
        ]

    def extract_features(self, text: str, marker_text: str, marker_category: str) -> Dict:
        doc = self.nlp(text)
        features = {}

        # Find the marker span in the processed document
        marker_span = None
        # Simple character-based search for the marker_text within the doc
        # This is a simplification; a more robust approach would align with tokens
        for sent in doc.sents:
            if marker_text.lower() in sent.text.lower():
                start_char = sent.text.lower().find(marker_text.lower())
                end_char = start_char + len(marker_text)
                marker_span = sent.char_span(start_char, end_char, alignment_mode="expand")
                if marker_span:
                    break
        
        if not marker_span:
            # Fallback if direct char_span fails, try to find the first token that matches
            for token in doc:
                if token.text.lower() == marker_text.lower():
                    marker_span = token.as_span()
                    break
        
        if not marker_span:
            # If still no marker_span, return empty features or handle error
            print(f"Warning: Marker '{marker_text}' not found in text: '{text[:50]}...'")
            return {}

        # 1. Lexical Features of the marker itself
        features['marker_lemma'] = marker_span.lemma_ if len(marker_span) == 1 else "_MULTI_"
        features['marker_pos'] = marker_span.root.pos_ if marker_span.root else "_NONE_"
        features['marker_dep'] = marker_span.root.dep_ if marker_span.root else "_NONE_"
        features['marker_length'] = len(marker_span.text.split())

        # 2. Contextual Features (within the sentence)
        sentence_text = marker_span.sent.text.lower()
        features['sentence_length_words'] = len(sentence_text.split())
        features['marker_position_ratio'] = marker_span.start / len(marker_span.sent) if len(marker_span.sent) > 0 else 0

        # Academic/Conversational keyword counts in sentence
        features['academic_keyword_count'] = sum(1 for kw in self.academic_keywords if kw in sentence_text)
        features['conversational_keyword_count'] = sum(1 for kw in self.conversational_keywords if kw in sentence_text)

        # 3. Surrounding Word Features (e.g., +/- 2 words)
        # Using token indices for robustness
        marker_start_token_idx = marker_span.start
        marker_end_token_idx = marker_span.end

        # Word before marker
        if marker_start_token_idx > 0:
            prev_token = doc[marker_start_token_idx - 1]
            features['prev_word_lemma'] = prev_token.lemma_
            features['prev_word_pos'] = prev_token.pos_
            features['prev_word_dep'] = prev_token.dep_
        else:
            features['prev_word_lemma'] = "_NONE_"
            features['prev_word_pos'] = "_NONE_"
            features['prev_word_dep'] = "_NONE_"

        # Word after marker
        if marker_end_token_idx < len(doc):
            next_token = doc[marker_end_token_idx]
            features['next_word_lemma'] = next_token.lemma_
            features['next_word_pos'] = next_token.pos_
            features['next_word_dep'] = next_token.dep_
        else:
            features['next_word_lemma'] = "_NONE_"
            features['next_word_pos'] = "_NONE_"
            features['next_word_dep'] = "_NONE_"

        # 4. Dependency Features (simplified)
        # Check if marker is a root or has specific dependency relations
        features['is_marker_root'] = marker_span.root.dep_ == "ROOT" if marker_span.root else False
        features['marker_head_pos'] = marker_span.root.head.pos_ if marker_span.root and marker_span.root.head else "_NONE_"
        features['marker_head_lemma'] = marker_span.root.head.lemma_ if marker_span.root and marker_span.root.head else "_NONE_"

        return features

def main():
    # ------------------------------------------------------------------
    # CLI – allow caller to override input / output paths
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Extract engineered features & labels for the metadiscourse ML pipeline."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="synthetic_metadiscourse_dataset.csv",
        help="Path to the raw synthetic dataset CSV (default: synthetic_metadiscourse_dataset.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="features_and_labels.csv",
        help="Where to write the extracted feature table (default: features_and_labels.csv)",
    )

    args = parser.parse_args()

    input_csv = args.input
    output_csv = args.output

    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} rows from {input_csv}")
    except FileNotFoundError:
        print(f"Error: {input_csv} not found. Please run generate_dataset.py first.")
        return

    analyzer = SimplifiedMetadiscourseAnalyzer()
    all_features = []

    for index, row in df.iterrows():
        text = row['text']
        marker_text = row['marker_text']
        marker_category = row['marker_category']
        is_metadiscourse = row['is_metadiscourse']

        features = analyzer.extract_features(text, marker_text, marker_category)
        if features: # Only add if features were successfully extracted
            features['marker_category_label'] = marker_category
            features['is_metadiscourse_label'] = is_metadiscourse
            all_features.append(features)
        
        if (index + 1) % 1000 == 0:
            print(f"Processed {index + 1} rows...")

    features_df = pd.DataFrame(all_features)
    
    # Handle categorical features: One-hot encoding
    categorical_cols = [
        'marker_lemma', 'marker_pos', 'marker_dep',
        'prev_word_lemma', 'prev_word_pos', 'prev_word_dep',
        'next_word_lemma', 'next_word_pos', 'next_word_dep',
        'marker_head_pos', 'marker_head_lemma'
    ]

    # Filter out columns that might not exist if some features were not extracted for all rows
    categorical_cols = [col for col in categorical_cols if col in features_df.columns]

    features_df = pd.get_dummies(features_df, columns=categorical_cols, dummy_na=False)

    features_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Features and labels saved to {output_csv}")
    print(f"Generated DataFrame shape: {features_df.shape}")
    print("Sample of generated features:")
    print(features_df.head())

if __name__ == "__main__":
    main()
