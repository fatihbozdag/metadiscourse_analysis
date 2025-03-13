# Metadiscourse Analysis Script
# This script can be run directly or converted to a notebook using nbconvert

# Import necessary libraries
import pandas as pd
import numpy as np
import os
import re
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token
from spacy.matcher import PhraseMatcher, Matcher
import torch
from typing import Dict, List, Set, Tuple
import math
from collections import Counter

# Configure patterns and pandas display
pattern = r'ICLE\-\w+\-\w+\-\d+\.\d+'
pattern = r'[^\w\s]'
pd.set_option('display.max_colwidth', None)

# Attempt to use MPS (Metal Performance Shaders) if available
try:
    torch.device('mps')  # For Apple Silicon Macs
    spacy.require_gpu()
    nlp = spacy.load("en_core_web_trf")
    print("Using GPU acceleration with the transformer model.")
except:
    try:
        nlp = spacy.load("en_core_web_sm")
        print("GPU acceleration not available, using CPU with the small model.")
    except:
        print("No spaCy models found. Please install with: python -m spacy download en_core_web_sm")

# Define comprehensive metadiscourse marker lists based on Hyland (2005)
INTERACTIVE_MARKERS = {
    # Code glosses
    "code_glosses": [
        "in other words", "that is", "i.e.", "that is to say", "this means", "in simple terms",
        "put simply", "to put it simply", "namely", "for example", "for instance", "such as", 
        "e.g.", "specifically", "particularly", "in fact", "indeed", "actually", "called", 
        "defined as", "referred to as", "including", "included", "especially", "notably"
    ],
    
    # Endophoric markers
    "endophoric_markers": [
        "in chapter", "in section", "in part", "in figure", "in table", "figure", "table", 
        "above", "below", "earlier", "previously", "as noted above", "as mentioned earlier", 
        "see", "refer to", "page", "the following", "as follows", "aforementioned"
    ],
    
    # Evidentials
    "evidentials": [
        "according to", "cited", "quoted", "states that", "argues that", "notes that", 
        "suggests that", "reports that", "found that", "observed that", "concluded that", 
        "in the literature", "previous research", "research shows", "studies indicate"
    ],
    
    # Frame markers
    "frame_markers": [
        # Sequencers
        "first", "firstly", "second", "secondly", "third", "thirdly", "fourth", "finally", 
        "lastly", "to begin with", "to start with", "next", "then", "subsequently",
        # Stage labels
        "in conclusion", "to conclude", "to summarize", "in summary", "in brief", "all in all", 
        "on the whole", "so far", "at this point", "overall",
        # Goal announcements
        "aim", "purpose", "goal", "objective", "focus", "seek to", "intend to", 
        # Topic shifters
        "with regard to", "concerning", "regarding", "turning to", "moving on to", "back to"
    ],
    
    # Transition markers
    "transition_markers": [
        # Additive
        "moreover", "furthermore", "in addition", "additionally", "besides", "similarly", 
        "likewise", "equally", "also", 
        # Causal
        "therefore", "thus", "consequently", "hence", "as a result", "because", "since", 
        "due to", "owing to", "so", 
        # Adversative
        "however", "nevertheless", "nonetheless", "but", "yet", "though", "although", "even though", 
        "despite", "in spite of", "in contrast", "on the other hand", "conversely", 
        # Temporal
        "meanwhile", "simultaneously", "subsequently", "previously", "after", "before", 
        "then", "later", "formerly", "eventually"
    ]
}

INTERACTIONAL_MARKERS = {
    # Attitude markers
    "attitude_markers": [
        "unfortunately", "fortunately", "surprisingly", "remarkably", "interestingly", 
        "hopefully", "importantly", "significantly", "correctly", "appropriately", "agree", 
        "prefer", "disagree", "dramatic", "unexpected", "desirable", "disappointing", "alarming",
        "it is surprising that", "it is important that", "it is significant that"
    ],
    
    # Self-mention
    "self_mention": [
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves", 
        "the author", "the authors", "the researcher", "the researchers", "this author"
    ],
    
    # Engagement markers
    "engagement_markers": [
        "you", "your", "yours", "yourself", "consider", "note", "imagine", "think about", 
        "let us", "let's", "see", "must", "should", "need to", "have to", "ought to", 
        "what about", "how about", "by the way", "the reader", "readers"
    ],
    
    # Hedges
    "hedges": [
        "may", "might", "could", "would", "perhaps", "possibly", "probably", "maybe", "likely", 
        "seemingly", "apparently", "approximately", "about", "roughly", "suggest", "assume", 
        "believe", "think", "appear", "seem", "indicate", "suspect", "suppose", "estimate", 
        "in my opinion", "from my perspective", "to my knowledge", "generally", "usually", 
        "sometimes", "often", "in most cases", "to some extent", "sort of", "kind of"
    ],
    
    # Boosters
    "boosters": [
        "clearly", "obviously", "certainly", "definitely", "undoubtedly", "undeniably", 
        "demonstrate", "prove", "show", "establish", "confirm", "find", "reveal", "must", 
        "will", "beyond doubt", "without doubt", "in fact", "indeed", "actually", "always", 
        "never", "absolutely", "completely", "entirely", "truly", "really", 
        "it is clear that", "we found that", "we proved that"
    ]
}

# Context rules to filter false positives
CONTEXT_PATTERNS = {
    "hedges": [
        # Modal verbs functioning as hedges (check they're not followed by nouns)
        [{"LOWER": {"IN": ["may", "might", "could", "would"]}}, {"TAG": {"NOT_IN": ["NN", "NNP"]}, "OP": "+"}, {"TAG": "VB"}],
        # "I think/believe" pattern
        [{"LOWER": {"IN": ["i", "we"]}}, {"LOWER": {"IN": ["think", "believe", "assume", "suppose"]}}],
        # "It appears/seems" pattern
        [{"LOWER": "it"}, {"LOWER": {"IN": ["appears", "seems", "looks"]}}],
        # "likely to" - epistemic rather than similarity
        [{"LOWER": "likely"}, {"LOWER": "to"}],
    ],
    
    "frame_markers": [
        # Sequencers followed by a comma often indicate discourse organization
        [{"LOWER": {"IN": ["first", "second", "third", "finally", "lastly"]}}, {"IS_PUNCT": True, "LOWER": ","}],
        # Goal statements
        [{"LOWER": {"IN": ["i", "we", "this", "the"]}}, {"LOWER": {"IN": ["aim", "intend", "focus", "purpose"]}}],
        # Topic shifters
        [{"LOWER": {"IN": ["turning", "moving"]}}, {"LOWER": "to"}],
    ],
    
    "transition_markers": [
        # Transitions at sentence start
        [{"IS_SENT_START": True}, {"LOWER": {"IN": ["however", "nevertheless", "thus", "therefore"]}}],
        # "Because of" causal pattern
        [{"LOWER": "because"}, {"LOWER": "of"}],
        # "As a result" pattern
        [{"LOWER": "as"}, {"LOWER": "a"}, {"LOWER": "result"}],
    ],
    
    "evidentials": [
        # "According to" pattern
        [{"LOWER": "according"}, {"LOWER": "to"}],
        # "X states/argues that" pattern
        [{"POS": "PROPN"}, {"LOWER": {"IN": ["states", "argues", "claims", "notes"]}}, {"LOWER": "that"}],
        # Passive citation
        [{"LOWER": {"IN": ["is", "was", "are", "were"]}}, {"LOWER": {"IN": ["cited", "reported", "claimed", "noted"]}}],
    ]
}

# Register the metadiscourse detector component
@Language.component("metadiscourse_detector")
def metadiscourse_detector(doc):
    """Improved metadiscourse detector with contextual filtering"""
    if not Doc.has_extension("metadiscourse_markers"):
        Doc.set_extension("metadiscourse_markers", default={})
    
    results = {
        # Interactive markers
        "code_glosses": [],
        "endophoric_markers": [],
        "evidentials": [],
        "frame_markers": [],
        "transition_markers": [],
        # Interactional markers
        "attitude_markers": [],
        "self_mention": [],
        "engagement_markers": [],
        "hedges": [],
        "boosters": []
    }
    
    # Create matchers for each category
    matcher = PhraseMatcher(doc.vocab, attr="LOWER")
    
    # Add interactive markers
    for category, markers in INTERACTIVE_MARKERS.items():
        patterns = [doc.vocab.strings.add(marker) for marker in markers]
        matcher.add(category, None, *patterns)
    
    # Add interactional markers
    for category, markers in INTERACTIONAL_MARKERS.items():
        patterns = [doc.vocab.strings.add(marker) for marker in markers]
        matcher.add(category, None, *patterns)
    
    # Find matches
    matches = matcher(doc)
    for match_id, start, end in matches:
        category = doc.vocab.strings[match_id]
        span = doc[start:end]
        results[category].append((span.text, (start, end)))
    
    # Apply context rules for improved accuracy
    context_matcher = Matcher(doc.vocab)
    
    # Add context patterns
    for category, patterns in CONTEXT_PATTERNS.items():
        for i, pattern in enumerate(patterns):
            context_matcher.add(f"{category}_{i}", [pattern])
    
    # Find context matches
    context_matches = context_matcher(doc)
    for match_id, start, end in context_matches:
        match_name = doc.vocab.strings[match_id]
        category = match_name.split('_')[0]
        span = doc[start:end]
        
        # Check if this is a new match or already captured
        if not any(start <= s < end or start < e <= end for _, (s, e) in results[category]):
            results[category].append((span.text, (start, end)))
    
    # Additional rule-based detection for evidentials
    for token in doc:
        if token.lemma_ in ["cite", "report", "claim", "note", "argue", "suggest", "state"]:
            # Check if it's in a reporting context
            if any(child.dep_ == "nsubj" for child in token.children) and any(
                child.dep_ == "ccomp" or child.dep_ == "xcomp" for child in token.children):
                # Check if not already captured
                if not any(start <= token.i < end for _, (start, end) in results["evidentials"]):
                    results["evidentials"].append((token.text, (token.i, token.i+1)))
    
    doc._.metadiscourse_markers = results
    return doc

def analyze_text(text, nlp=None):
    """Analyze a text for metadiscourse markers and calculate statistics."""
    if nlp is None:
        # Load spaCy model if not provided
        try:
            nlp = spacy.load("en_core_web_trf")
        except:
            nlp = spacy.load("en_core_web_sm")
            print("Warning: Using smaller spaCy model. For better results, install en_core_web_trf.")
        
        # Add the metadiscourse detector component if not already added
        if "metadiscourse_detector" not in nlp.pipe_names:
            nlp.add_pipe("metadiscourse_detector", last=True)
    
    # Process the text
    doc = nlp(text)
    
    # Get word count (excluding punctuation)
    word_count = len([token for token in doc if not token.is_punct and not token.is_space])
    
    # Extract metadiscourse markers
    markers = doc._.metadiscourse_markers
    
    # Count markers by category
    counts = {}
    for category, instances in markers.items():
        counts[category] = len(instances)
    
    # Calculate interactive and interactional totals
    interactive_categories = ["code_glosses", "endophoric_markers", "evidentials", "frame_markers", "transition_markers"]
    interactional_categories = ["attitude_markers", "self_mention", "engagement_markers", "hedges", "boosters"]
    
    interactive_total = sum(counts.get(cat, 0) for cat in interactive_categories)
    interactional_total = sum(counts.get(cat, 0) for cat in interactional_categories)
    
    # Calculate normalized frequencies (per 1000 words)
    density_factor = 1000 / word_count if word_count > 0 else 0
    
    results = {
        "interactive_total": interactive_total,
        "interactional_total": interactional_total,
        "interactive_density": interactive_total * density_factor,
        "interactional_density": interactional_total * density_factor,
        "word_count": word_count
    }
    
    # Add individual category counts and densities
    for category in interactive_categories + interactional_categories:
        results[category] = counts.get(category, 0)
        results[f"{category}_density"] = counts.get(category, 0) * density_factor
    
    return results

def load_corpus(corpus_path, language_map=None):
    """Load corpus texts and analyze them for metadiscourse markers."""
    results = []
    
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_trf")
    except:
        nlp = spacy.load("en_core_web_sm")
        print("Warning: Using smaller spaCy model. For better results, install en_core_web_trf.")
    
    # Add the metadiscourse detector component
    if "metadiscourse_detector" not in nlp.pipe_names:
        nlp.add_pipe("metadiscourse_detector", last=True)
    
    # Process all text files in the corpus directory
    for file_path in os.listdir(corpus_path):
        if file_path.endswith(".txt"):
            full_path = os.path.join(corpus_path, file_path)
            print(f"Processing {file_path}...")
            
            # Read text file
            with open(full_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Get native language if available
            native_language = language_map.get(file_path, "Unknown") if language_map else "Unknown"
            
            # Analyze metadiscourse markers
            analysis_result = analyze_text(text, nlp)
            
            # Add filename and native language to results
            analysis_result["Filename"] = file_path
            analysis_result["Native_Language"] = native_language
            
            results.append(analysis_result)
    
    return pd.DataFrame(results)

def load_language_map(file_path):
    """Load mapping between filenames and native languages."""
    language_map = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                filename, language = parts[0].strip(), parts[1].strip()
                language_map[filename] = language
    
    return language_map

# Function to generate LaTeX tables for academic publication
def generate_latex_tables(df, output_dir='results'):
    """Generate enhanced LaTeX tables for academic publication."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating language summary table...")
    # Generate summary statistics by language group
    if 'Native_Language' in df.columns:
        language_summary = df.groupby('Native_Language').agg({
            'interactive_density': ['mean', 'std', 'count'],
            'interactional_density': ['mean', 'std', 'count'],
            'word_count': ['mean', 'std']
        })
        
        # Format the summary table for LaTeX
        language_summary_latex = language_summary.to_latex(
            float_format="{:.2f}".format,
            caption="Summary of Metadiscourse Markers by Native Language",
            label="tab:language_summary",
            position="h!",
            multirow=True
        )
        
        # Add notes and enhance formatting
        language_summary_latex = language_summary_latex.replace(
            "\\end{tabular}",
            "\\bottomrule\n\\end{tabular}\n\\caption*{Note: Values represent mean and standard deviation per 1000 words.}"
        )
        
        # Save to file
        with open(os.path.join(output_dir, 'language_summary_table_full.tex'), 'w') as f:
            f.write(language_summary_latex)
        print(f"Language summary table saved to {os.path.join(output_dir, 'language_summary_table_full.tex')}")
    
    print("Generating detailed marker statistics table...")
    # Generate detailed statistics for interactive markers
    interactive_cols = [col for col in df.columns if col.endswith('_density') and 
                        any(col.startswith(cat) for cat in ['code_glosses', 'endophoric_markers', 'evidentials', 'frame_markers', 'transition_markers'])]
    
    if interactive_cols:
        interactive_stats = df[interactive_cols].describe().T.reset_index()
        interactive_stats.columns = ['Marker', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
        
        # Format for LaTeX
        interactive_latex = interactive_stats.to_latex(
            index=False,
            float_format="{:.3f}".format,
            caption="Descriptive Statistics for Interactive Metadiscourse Markers",
            label="tab:interactive_markers",
            position="h!"
        )
        
        # Enhance formatting
        interactive_latex = interactive_latex.replace(
            "\\begin{tabular}",
            "\\begin{tabular}{lrrrrrrr}"
        ).replace(
            "\\end{tabular}",
            "\\bottomrule\n\\end{tabular}\n\\caption*{Note: Values represent frequency per 1000 words.}"
        )
        
        # Save to file
        with open(os.path.join(output_dir, 'interactive_markers_table_full.tex'), 'w') as f:
            f.write(interactive_latex)
        print(f"Interactive markers table saved to {os.path.join(output_dir, 'interactive_markers_table_full.tex')}")
    
    # Generate detailed statistics for interactional markers
    interactional_cols = [col for col in df.columns if col.endswith('_density') and 
                          any(col.startswith(cat) for cat in ['attitude_markers', 'self_mention', 'engagement_markers', 'hedges', 'boosters'])]
    
    if interactional_cols:
        interactional_stats = df[interactional_cols].describe().T.reset_index()
        interactional_stats.columns = ['Marker', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
        
        # Format for LaTeX
        interactional_latex = interactional_stats.to_latex(
            index=False,
            float_format="{:.3f}".format,
            caption="Descriptive Statistics for Interactional Metadiscourse Markers",
            label="tab:interactional_markers",
            position="h!"
        )
        
        # Enhance formatting
        interactional_latex = interactional_latex.replace(
            "\\begin{tabular}",
            "\\begin{tabular}{lrrrrrrr}"
        ).replace(
            "\\end{tabular}",
            "\\bottomrule\n\\end{tabular}\n\\caption*{Note: Values represent frequency per 1000 words.}"
        )
        
        # Save to file
        with open(os.path.join(output_dir, 'interactional_markers_table_full.tex'), 'w') as f:
            f.write(interactional_latex)
        print(f"Interactional markers table saved to {os.path.join(output_dir, 'interactional_markers_table_full.tex')}")
    
    print("Generating correlation matrix table...")
    # Generate correlation matrix
    density_cols = [col for col in df.columns if col.endswith('_density')]
    if density_cols:
        corr_matrix = df[density_cols].corr()
        
        # Format for LaTeX
        corr_latex = corr_matrix.to_latex(
            float_format="{:.3f}".format,
            caption="Correlation Matrix of Metadiscourse Markers",
            label="tab:correlation_matrix",
            position="h!"
        )
        
        # Enhance formatting
        corr_latex = corr_latex.replace(
            "\\end{tabular}",
            "\\bottomrule\n\\end{tabular}\n\\caption*{Note: Values represent Pearson correlation coefficients.}"
        )
        
        # Save to file
        with open(os.path.join(output_dir, 'correlation_matrix_full.tex'), 'w') as f:
            f.write(corr_latex)
        print(f"Correlation matrix table saved to {os.path.join(output_dir, 'correlation_matrix_full.tex')}")
    
    # Create a combined LaTeX file with all tables
    print("Creating combined LaTeX file with all tables...")
    combined_latex = (
        "\\documentclass{article}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{caption}\n"
        "\\usepackage{multirow}\n"
        "\\usepackage{float}\n"
        "\\begin{document}\n"
        "\\title{Metadiscourse Analysis Results}\n"
        "\\author{Metadiscourse Analyzer}\n"
        "\\date{\\today}\n"
        "\\maketitle\n"
        "\\section{Summary Statistics}\n"
    )
    
    # Add language summary table
    if 'Native_Language' in df.columns:
        combined_latex += (
            "\\subsection{Language Group Summary}\n"
            "\\input{" + os.path.join(output_dir, 'language_summary_table_full').replace('\\', '/') + "}\n"
        )
    
    # Add marker statistics tables
    combined_latex += (
        "\\subsection{Metadiscourse Marker Statistics}\n"
        "\\input{" + os.path.join(output_dir, 'interactive_markers_table_full').replace('\\', '/') + "}\n"
        "\\input{" + os.path.join(output_dir, 'interactional_markers_table_full').replace('\\', '/') + "}\n"
        "\\subsection{Correlation Analysis}\n"
        "\\input{" + os.path.join(output_dir, 'correlation_matrix_full').replace('\\', '/') + "}\n"
        "\\end{document}\n"
    )
    
    # Save combined file
    with open(os.path.join(output_dir, 'metadiscourse_analysis_tables.tex'), 'w') as f:
        f.write(combined_latex)

def perform_language_group_statistics(df, output_dir='results'):
    """Perform statistical tests to compare language groups."""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'Native_Language' not in df.columns or len(df['Native_Language'].unique()) <= 1:
        return
    
    # Measures to test for language group differences
    measures_to_test = [
        'interactional_density',
        'interactive_endophoric_markers',
        'interactive_frame_markers',
        'interactive_transition_markers',
        'interactional_boosters'
    ]
    
    results = []
    
    for measure in measures_to_test:
        if measure in df.columns:
            try:
                # Perform ANOVA
                formula = f"{measure} ~ C(Native_Language)"
                model = ols(formula, data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                # Extract F-value and p-value
                f_value = anova_table['F'][0]
                p_value = anova_table['PR(>F)'][0]
                
                # Perform post-hoc test if significant
                posthoc_results = None
                if p_value < 0.05:
                    mc = MultiComparison(df[measure], df['Native_Language'])
                    posthoc_results = mc.tukeyhsd()
                    
                    # Save posthoc results to file
                    posthoc_file = os.path.join(output_dir, f"{measure}_posthoc.txt")
                    with open(posthoc_file, 'w') as f:
                        f.write(str(posthoc_results))
                
                # Add to results
                results.append({
                    'Measure': measure,
                    'F_value': f_value,
                    'p_value': p_value,
                    'Significant': p_value < 0.05,
                    'Posthoc_file': f"{measure}_posthoc.txt" if p_value < 0.05 else None
                })
                
            except Exception as e:
                print(f"Error performing ANOVA for {measure}: {e}")
    
    # Create a summary table
    if results:
        results_df = pd.DataFrame(results)
        
        # Format for LaTeX
        results_latex = results_df.to_latex(
            index=False,
            float_format="{:.3f}".format,
            caption="ANOVA Results for Language Group Differences",
            label="tab:anova_results",
            position="h!"
        )
        
        # Enhance formatting
        results_latex = results_latex.replace(
            "\\end{tabular}",
            "\\bottomrule\n\\end{tabular}\n\\caption*{Note: * indicates significant at p < 0.05.}"
        )
        
        # Save to file
        with open(os.path.join(output_dir, 'anova_results.tex'), 'w') as f:
            f.write(results_latex)
        
        # Save as CSV for easier viewing
        results_df.to_csv(os.path.join(output_dir, 'anova_results.csv'), index=False)

def generate_visualizations(df, output_dir='results'):
    """Generate visualizations for metadiscourse analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style for plots
    sns.set(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': (12, 8)
    })
    
    # 1. Interactive vs. Interactional Markers Scatter Plot
    print("Generating interactive vs. interactional markers scatter plot...")
    plt.figure(figsize=(10, 8))
    
    if 'Native_Language' in df.columns:
        scatter = sns.scatterplot(
            data=df,
            x='interactive_density',
            y='interactional_density',
            hue='Native_Language',
            s=100,
            alpha=0.7
        )
    else:
        scatter = sns.scatterplot(
            data=df,
            x='interactive_density',
            y='interactional_density',
            s=100,
            alpha=0.7
        )
    
    plt.title('Interactive vs. Interactional Metadiscourse Markers')
    plt.xlabel('Interactive Markers (per 1000 words)')
    plt.ylabel('Interactional Markers (per 1000 words)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add regression line
    sns.regplot(
        x='interactive_density',
        y='interactional_density',
        data=df,
        scatter=False,
        ci=None,
        line_kws={"color": "red", "lw": 2, "linestyle": "--"}
    )
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'interactive_vs_interactional.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'interactive_vs_interactional.pdf'))
    plt.close()
    
    # 2. Boxplots of marker categories
    print("Generating boxplots of marker categories...")
    # Get all density columns
    density_cols = [col for col in df.columns if col.endswith('_density') and not col.startswith('interactive_') and not col.startswith('interactional_')]
    
    if density_cols:
        # Melt the dataframe for easier plotting
        melted_df = pd.melt(
            df,
            id_vars=['Filename'] + (['Native_Language'] if 'Native_Language' in df.columns else []),
            value_vars=density_cols,
            var_name='Marker_Category',
            value_name='Density'
        )
        
        # Clean up marker category names
        melted_df['Marker_Category'] = melted_df['Marker_Category'].str.replace('_density', '')
        
        plt.figure(figsize=(14, 8))
        box = sns.boxplot(
            data=melted_df,
            x='Marker_Category',
            y='Density',
            palette='viridis'
        )
        
        plt.title('Distribution of Metadiscourse Marker Categories')
        plt.xlabel('Marker Category')
        plt.ylabel('Frequency (per 1000 words)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'marker_categories_boxplot.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'marker_categories_boxplot.pdf'))
        plt.close()
    
    # 3. Language group comparison (if available)
    if 'Native_Language' in df.columns and len(df['Native_Language'].unique()) > 1:
        print("Generating language group comparison plots...")
        
        # Interactive markers by language
        plt.figure(figsize=(12, 8))
        bar = sns.barplot(
            data=df,
            x='Native_Language',
            y='interactive_density',
            palette='Blues_d',
            errorbar=('ci', 95),
            capsize=0.2
        )
        
        plt.title('Interactive Metadiscourse Markers by Native Language')
        plt.xlabel('Native Language')
        plt.ylabel('Interactive Markers (per 1000 words)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'interactive_by_language.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'interactive_by_language.pdf'))
        plt.close()
        
        # Interactional markers by language
        plt.figure(figsize=(12, 8))
        bar = sns.barplot(
            data=df,
            x='Native_Language',
            y='interactional_density',
            palette='Reds_d',
            errorbar=('ci', 95),
            capsize=0.2
        )
        
        plt.title('Interactional Metadiscourse Markers by Native Language')
        plt.xlabel('Native Language')
        plt.ylabel('Interactional Markers (per 1000 words)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'interactional_by_language.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'interactional_by_language.pdf'))
        plt.close()
        
        # Heatmap of marker categories by language
        print("Generating heatmap of marker categories by language...")
        # Calculate mean densities by language
        heatmap_data = df.groupby('Native_Language')[density_cols].mean()
        
        plt.figure(figsize=(16, 10))
        heatmap = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            cmap='viridis',
            linewidths=0.5,
            cbar_kws={'label': 'Mean Frequency (per 1000 words)'}
        )
        
        plt.title('Metadiscourse Marker Categories by Native Language')
        plt.ylabel('Native Language')
        plt.xlabel('Marker Category')
        
        # Clean up x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.xticks(
            [x + 0.5 for x in range(len(density_cols))],
            [col.replace('_density', '') for col in density_cols],
            rotation=45,
            ha='right'
        )
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'marker_heatmap_by_language.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'marker_heatmap_by_language.pdf'))
        plt.close()
    
    # 4. Correlation heatmap
    print("Generating correlation heatmap...")
    density_cols = [col for col in df.columns if col.endswith('_density')]
    if density_cols:
        corr_matrix = df[density_cols].corr()
        
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        heatmap = sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'}
        )
        
        plt.title('Correlation Matrix of Metadiscourse Markers')
        
        # Clean up axis labels
        clean_labels = [col.replace('_density', '') for col in density_cols]
        plt.xticks(ticks=[i + 0.5 for i in range(len(clean_labels))], labels=clean_labels, rotation=45, ha='right')
        plt.yticks(ticks=[i + 0.5 for i in range(len(clean_labels))], labels=clean_labels, rotation=0)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.pdf'))
        plt.close()
    
    print(f"All visualizations saved to {output_dir}")

def analyze_metadiscourse_distribution(df, output_dir='results'):
    """Analyze the distribution of metadiscourse markers and generate visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all marker categories
    interactive_categories = ["code_glosses", "endophoric_markers", "evidentials", "frame_markers", "transition_markers"]
    interactional_categories = ["attitude_markers", "self_mention", "engagement_markers", "hedges", "boosters"]
    
    # Calculate total markers for each category
    total_markers = {}
    for category in interactive_categories + interactional_categories:
        if category in df.columns:
            total_markers[category] = df[category].sum()
    
    # Calculate percentages
    total_count = sum(total_markers.values())
    percentages = {category: (count / total_count) * 100 for category, count in total_markers.items()}
    
    # Create a DataFrame for easier plotting
    distribution_df = pd.DataFrame({
        'Category': list(percentages.keys()),
        'Count': list(total_markers.values()),
        'Percentage': list(percentages.values())
    })
    
    # Add category type (interactive or interactional)
    distribution_df['Type'] = distribution_df['Category'].apply(
        lambda x: 'Interactive' if x in interactive_categories else 'Interactional'
    )
    
    # Sort by percentage
    distribution_df = distribution_df.sort_values('Percentage', ascending=False)
    
    # Save the distribution data
    distribution_df.to_csv(os.path.join(output_dir, 'metadiscourse_distribution.csv'), index=False)
    
    # Generate pie chart
    plt.figure(figsize=(12, 10))
    plt.pie(
        distribution_df['Percentage'],
        labels=distribution_df['Category'],
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        explode=[0.05] * len(distribution_df),
        colors=sns.color_palette('viridis', len(distribution_df))
    )
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Distribution of Metadiscourse Marker Categories')
    
    # Save the pie chart
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metadiscourse_distribution_pie.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'metadiscourse_distribution_pie.pdf'))
    plt.close()
    
    # Generate grouped bar chart by type
    plt.figure(figsize=(14, 8))
    bar = sns.barplot(
        data=distribution_df,
        x='Category',
        y='Percentage',
        hue='Type',
        palette={'Interactive': 'skyblue', 'Interactional': 'salmon'}
    )
    
    plt.title('Distribution of Metadiscourse Marker Categories by Type')
    plt.xlabel('Category')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Marker Type')
    
    # Save the bar chart
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metadiscourse_distribution_bar.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'metadiscourse_distribution_bar.pdf'))
    plt.close()
    
    # Calculate and return summary statistics
    interactive_total = sum(total_markers.get(cat, 0) for cat in interactive_categories)
    interactional_total = sum(total_markers.get(cat, 0) for cat in interactional_categories)
    
    summary = {
        'interactive_percentage': (interactive_total / total_count) * 100 if total_count > 0 else 0,
        'interactional_percentage': (interactional_total / total_count) * 100 if total_count > 0 else 0,
        'most_frequent_category': distribution_df.iloc[0]['Category'],
        'most_frequent_percentage': distribution_df.iloc[0]['Percentage'],
        'least_frequent_category': distribution_df.iloc[-1]['Category'],
        'least_frequent_percentage': distribution_df.iloc[-1]['Percentage']
    }
    
    return summary

def calculate_shannon_entropy(df, output_dir='results'):
    """Calculate Shannon entropy for metadiscourse markers to measure diversity."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all marker categories
    categories = [
        "code_glosses", "endophoric_markers", "evidentials", "frame_markers", "transition_markers",
        "attitude_markers", "self_mention", "engagement_markers", "hedges", "boosters"
    ]
    
    # Filter to only include categories present in the dataframe
    categories = [cat for cat in categories if cat in df.columns]
    
    if not categories:
        print("No marker categories found in the dataframe.")
        return {}
    
    # Calculate entropy for each text
    entropy_results = []
    
    for _, row in df.iterrows():
        # Get counts for each category
        counts = [row[cat] for cat in categories]
        total = sum(counts)
        
        # Calculate probabilities
        probabilities = [count / total if total > 0 else 0 for count in counts]
        
        # Calculate entropy
        entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probabilities)
        
        # Calculate maximum possible entropy for this number of categories
        max_entropy = math.log2(len(categories))
        
        # Calculate normalized entropy (0-1 scale)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Add to results
        result = {
            'Filename': row['Filename'] if 'Filename' in row else f"Text_{_}",
            'Shannon_Entropy': entropy,
            'Normalized_Entropy': normalized_entropy,
            'Max_Entropy': max_entropy
        }
        
        # Add native language if available
        if 'Native_Language' in row:
            result['Native_Language'] = row['Native_Language']
        
        entropy_results.append(result)
    
    # Create DataFrame
    entropy_df = pd.DataFrame(entropy_results)
    
    # Save to CSV
    entropy_df.to_csv(os.path.join(output_dir, 'entropy_analysis.csv'), index=False)
    
    # Generate visualization
    plt.figure(figsize=(12, 8))
    
    if 'Native_Language' in entropy_df.columns:
        # Box plot by language
        box = sns.boxplot(
            data=entropy_df,
            x='Native_Language',
            y='Normalized_Entropy',
            palette='viridis'
        )
        
        # Add individual points
        swarm = sns.swarmplot(
            data=entropy_df,
            x='Native_Language',
            y='Normalized_Entropy',
            color='black',
            alpha=0.5,
            size=8
        )
        
        plt.title('Metadiscourse Diversity (Normalized Shannon Entropy) by Native Language')
        plt.xlabel('Native Language')
    else:
        # Histogram
        hist = sns.histplot(
            data=entropy_df,
            x='Normalized_Entropy',
            kde=True,
            bins=15,
            color='skyblue'
        )
        
        plt.title('Distribution of Metadiscourse Diversity (Normalized Shannon Entropy)')
        plt.xlabel('Normalized Shannon Entropy')
    
    plt.ylabel('Normalized Shannon Entropy (0-1)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_analysis.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'entropy_analysis.pdf'))
    plt.close()
    
    # Calculate summary statistics
    summary = {
        'mean_entropy': entropy_df['Shannon_Entropy'].mean(),
        'mean_normalized_entropy': entropy_df['Normalized_Entropy'].mean(),
        'max_entropy': entropy_df['Max_Entropy'].iloc[0],  # Should be the same for all texts
        'min_normalized_entropy': entropy_df['Normalized_Entropy'].min(),
        'max_normalized_entropy': entropy_df['Normalized_Entropy'].max()
    }
    
    # If we have language groups, add language-specific statistics
    if 'Native_Language' in entropy_df.columns:
        language_stats = entropy_df.groupby('Native_Language')['Normalized_Entropy'].agg(['mean', 'std', 'min', 'max'])
        language_stats_dict = language_stats.to_dict(orient='index')
        summary['language_stats'] = language_stats_dict
        
        # Perform ANOVA to test for language group differences
        try:
            formula = "Normalized_Entropy ~ C(Native_Language)"
            model = ols(formula, data=entropy_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            summary['entropy_anova_f'] = anova_table['F'][0]
            summary['entropy_anova_p'] = anova_table['PR(>F)'][0]
            summary['entropy_significant_diff'] = anova_table['PR(>F)'][0] < 0.05
        except Exception as e:
            print(f"Error performing ANOVA for entropy: {e}")
    
    return summary

def main():
    """Main function to run the metadiscourse analysis."""
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Analyze metadiscourse markers in texts.')
    parser.add_argument('--input', '-i', type=str, help='Input CSV file with text data')
    parser.add_argument('--corpus', '-c', type=str, help='Directory containing corpus text files')
    parser.add_argument('--language_map', '-l', type=str, help='File mapping filenames to native languages')
    parser.add_argument('--output_dir', '-o', type=str, default='analysis_results', help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize results dataframe
    results_df = None
    
    # Process input based on provided arguments
    if args.input:
        print(f"Loading data from CSV file: {args.input}")
        try:
            # Load data from CSV
            meta = pd.read_csv(args.input)
            text = pd.read_csv(args.input)
            
            # Preprocess text
            text['text_field'] = text['text_field'].apply(lambda x: re.sub(pattern, '', str(x)).replace('\n', ''))
            text['text_field'] = text['text_field'].str.lower()
            
            # Create list of (text, metadata) tuples
            meta_x = meta.to_dict('records')
            text_only = text['text_field'].values.tolist()
            corpus_data = list(zip(text_only, meta_x))
            
            # Analyze each text
            results = []
            for text, metadata in corpus_data:
                print(f"Analyzing text: {metadata.get('Filename', 'Unknown')}")
                analysis = analyze_text(text)
                
                # Add metadata
                for key, value in metadata.items():
                    analysis[key] = value
                
                results.append(analysis)
            
            # Create DataFrame
            results_df = pd.DataFrame(results)
            
        except Exception as e:
            print(f"Error processing CSV file: {e}")
            return
    
    elif args.corpus:
        print(f"Loading corpus from directory: {args.corpus}")
        try:
            # Load language map if provided
            language_map = None
            if args.language_map:
                language_map = load_language_map(args.language_map)
                print(f"Loaded language map with {len(language_map)} entries")
            
            # Load and analyze corpus
            results_df = load_corpus(args.corpus, language_map)
            
        except Exception as e:
            print(f"Error processing corpus: {e}")
            return
    
    else:
        print("No input specified. Please provide either --input or --corpus.")
        parser.print_help()
        return
    
    # Save raw results
    if results_df is not None:
        results_csv = os.path.join(args.output_dir, 'metadiscourse_analysis_results.csv')
        results_df.to_csv(results_csv, index=False)
        print(f"Raw analysis results saved to {results_csv}")
        
        # Generate LaTeX tables
        print("\nGenerating LaTeX tables...")
        generate_latex_tables(results_df, args.output_dir)
        
        # Perform statistical analysis
        print("\nPerforming statistical analysis...")
        perform_language_group_statistics(results_df, args.output_dir)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        generate_visualizations(results_df, args.output_dir)
        
        # Analyze metadiscourse distribution
        print("\nAnalyzing metadiscourse distribution...")
        distribution_summary = analyze_metadiscourse_distribution(results_df, args.output_dir)
        
        # Calculate Shannon entropy
        print("\nCalculating Shannon entropy...")
        entropy_summary = calculate_shannon_entropy(results_df, args.output_dir)
        
        # Generate summary report
        print("\nGenerating summary report...")
        report_path = os.path.join(args.output_dir, 'analysis_summary.txt')
        with open(report_path, 'w') as f:
            f.write("=== METADISCOURSE ANALYSIS SUMMARY ===\n\n")
            
            # Basic statistics
            f.write(f"Total texts analyzed: {len(results_df)}\n")
            if 'Native_Language' in results_df.columns:
                f.write(f"Language groups: {', '.join(results_df['Native_Language'].unique())}\n")
            
            # Distribution summary
            f.write("\n--- Metadiscourse Distribution ---\n")
            f.write(f"Interactive markers: {distribution_summary['interactive_percentage']:.2f}%\n")
            f.write(f"Interactional markers: {distribution_summary['interactional_percentage']:.2f}%\n")
            f.write(f"Most frequent category: {distribution_summary['most_frequent_category']} ({distribution_summary['most_frequent_percentage']:.2f}%)\n")
            f.write(f"Least frequent category: {distribution_summary['least_frequent_category']} ({distribution_summary['least_frequent_percentage']:.2f}%)\n")
            
            # Entropy summary
            f.write("\n--- Metadiscourse Diversity (Shannon Entropy) ---\n")
            f.write(f"Mean normalized entropy: {entropy_summary['mean_normalized_entropy']:.4f} (0-1 scale)\n")
            f.write(f"Max possible entropy: {entropy_summary['max_entropy']:.4f} bits\n")
            
            # Language-specific entropy if available
            if 'language_stats' in entropy_summary:
                f.write("\nEntropy by language group:\n")
                for lang, stats in entropy_summary['language_stats'].items():
                    f.write(f"  {lang}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                
                if 'entropy_anova_p' in entropy_summary:
                    f.write(f"\nANOVA test for entropy differences: F={entropy_summary['entropy_anova_f']:.2f}, p={entropy_summary['entropy_anova_p']:.4f}")
                    if entropy_summary.get('entropy_significant_diff', False):
                        f.write(" (significant)\n")
                    else:
                        f.write(" (not significant)\n")
        
        print(f"Summary report saved to {report_path}")
        print(f"\nAll analysis results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
