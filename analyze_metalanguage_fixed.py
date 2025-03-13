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

# Load and preprocess data
# Configure patterns and pandas display
pattern = r'ICLE\-\w+\-\w+\-\d+\.\d+'
pattern = r'[^\w\s]'
pd.set_option('display.max_colwidth', None)
# Load and preprocess data
meta = pd.read_csv('/Users/fatihbozdag/Documents/Studies/AI Library/Metadata/metadata_with_text.csv')  # Update with your path
text = pd.read_csv('/Users/fatihbozdag/Documents/Studies/AI Library/Metadata/metadata_with_text.csv')  # Update with your path
text['text_field'] = text['text_field'].apply(lambda x: re.sub(pattern, '', x).replace('\n', ''))
text['text_field'] = text['text_field'].apply(lambda x: re.sub(pattern, '', x).replace('\n', ''))
text['text_field'] = text['text_field'].str.lower()
meta_x = meta.to_dict('records')
text_only = text['text_field'].values.tolist()
icle = list(zip(text_only, meta_x))

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

# Configure patterns and pandas display
pattern = r'ICLE\-\w+\-\w+\-\d+\.\d+'
pattern = r'[^\w\s]'
pd.set_option('display.max_colwidth', None)



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

# Function to generate LaTeX tables for academic publication
def generate_latex_tables(df, output_dir='results'):
    """
    Generate LaTeX tables for academic publication from the analysis results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing analysis results
    output_dir : str, optional
        Directory to save the LaTeX tables
        
    Returns:
    --------
    None
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # LaTeX preamble for standalone tables
        latex_preamble = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"  # For professional tables with proper spacing
            "\\usepackage{siunitx}\n"   # For proper alignment of numbers
            "\\usepackage{caption}\n"   # For better captions
            "\\usepackage{threeparttable}\n"  # For table notes
            "\\usepackage[table]{xcolor}\n"  # For table colors
            "\\usepackage{multirow}\n"  # For merged rows
            "\\begin{document}\n"
        )
        
        # LaTeX closing
        latex_closing = "\\end{document}"
        
        # Table 1: Summary statistics by native language
        if 'Native_Language' in df.columns and len(df['Native_Language'].unique()) > 1:
            print("Generating language summary table...")
            language_summary = df.groupby('Native_Language')[
                ['interactive_density', 'interactional_density', 'interactive_entropy', 'interactional_entropy']
            ].agg(['mean', 'std'])
            
            # Format the table with proper column names
            language_summary.columns = [
                f"{col[0].split('_')[0].capitalize()} {col[0].split('_')[1].capitalize()} ({col[1].capitalize()})"
                for col in language_summary.columns
            ]
            
            # Generate LaTeX table with booktabs style
            latex_table = language_summary.to_latex(float_format="%.2f", escape=False)
            
            # Enhance the LaTeX table with proper formatting
            enhanced_table = (
                "\\begin{table}[htbp]\n"
                "\\centering\n"
                "\\caption{Summary Statistics of Metadiscourse Markers by Native Language}\n"
                "\\label{tab:language_summary}\n"
                "\\begin{threeparttable}\n"
                + latex_table +
                "\\begin{tablenotes}\n"
                "\\small\n"
                "\\item Note: Values represent means and standard deviations of marker densities (per 1000 words) "
                "and entropy measures across different native language groups.\n"
                "\\end{tablenotes}\n"
                "\\end{threeparttable}\n"
                "\\end{table}\n"
            )
            
            # Save to file - both standalone and embedded versions
            with open(f"{output_dir}/language_summary_table.tex", "w") as f:
                f.write(latex_table)
                
            with open(f"{output_dir}/language_summary_table_full.tex", "w") as f:
                f.write(latex_preamble + enhanced_table + latex_closing)
            
            print(f"Language summary table saved to {output_dir}/language_summary_table_full.tex")
        
        # Table 2: Detailed marker statistics by native language
        if 'Native_Language' in df.columns and len(df['Native_Language'].unique()) > 1:
            print("Generating detailed marker statistics table...")
            
            # Get interactive and interactional marker columns
            interactive_markers = [col for col in df.columns if col.startswith('interactive_') 
                                 and col not in ['interactive_total', 'interactive_density', 'interactive_entropy',
                                                'interactive_density_log', 'interactive_density_sqrt']]
            
            interactional_markers = [col for col in df.columns if col.startswith('interactional_') 
                                   and col not in ['interactional_total', 'interactional_density', 'interactional_entropy',
                                                  'interactional_density_log', 'interactional_density_sqrt']]
            
            # Create summary tables for each marker type
            if interactive_markers:
                interactive_summary = df.groupby('Native_Language')[interactive_markers].agg(['mean', 'std'])
                
                # Format column names
                interactive_summary.columns = [
                    f"{col[0].split('_', 1)[1].replace('_', ' ').title()} ({col[1].capitalize()})"
                    for col in interactive_summary.columns
                ]
                
                # Generate LaTeX table
                latex_table = interactive_summary.to_latex(float_format="%.2f", escape=False)
                
                # Enhance the LaTeX table
                enhanced_table = (
                    "\\begin{table}[htbp]\n"
                    "\\centering\n"
                    "\\caption{Detailed Statistics of Interactive Metadiscourse Markers by Native Language}\n"
                    "\\label{tab:interactive_markers}\n"
                    "\\begin{threeparttable}\n"
                    + latex_table +
                    "\\begin{tablenotes}\n"
                    "\\small\n"
                    "\\item Note: Values represent means and standard deviations of interactive marker densities "
                    "(per 1000 words) across different native language groups.\n"
                    "\\end{tablenotes}\n"
                    "\\end{threeparttable}\n"
                    "\\end{table}\n"
                )
                
                # Save to file
                with open(f"{output_dir}/interactive_markers_table.tex", "w") as f:
                    f.write(latex_table)
                    
                with open(f"{output_dir}/interactive_markers_table_full.tex", "w") as f:
                    f.write(latex_preamble + enhanced_table + latex_closing)
                
                print(f"Interactive markers table saved to {output_dir}/interactive_markers_table_full.tex")
            
            if interactional_markers:
                interactional_summary = df.groupby('Native_Language')[interactional_markers].agg(['mean', 'std'])
                
                # Format column names
                interactional_summary.columns = [
                    f"{col[0].split('_', 1)[1].replace('_', ' ').title()} ({col[1].capitalize()})"
                    for col in interactional_summary.columns
                ]
                
                # Generate LaTeX table
                latex_table = interactional_summary.to_latex(float_format="%.2f", escape=False)
                
                # Enhance the LaTeX table
                enhanced_table = (
                    "\\begin{table}[htbp]\n"
                    "\\centering\n"
                    "\\caption{Detailed Statistics of Interactional Metadiscourse Markers by Native Language}\n"
                    "\\label{tab:interactional_markers}\n"
                    "\\begin{threeparttable}\n"
                    + latex_table +
                    "\\begin{tablenotes}\n"
                    "\\small\n"
                    "\\item Note: Values represent means and standard deviations of interactional marker densities "
                    "(per 1000 words) across different native language groups.\n"
                    "\\end{tablenotes}\n"
                    "\\end{threeparttable}\n"
                    "\\end{table}\n"
                )
                
                # Save to file
                with open(f"{output_dir}/interactional_markers_table.tex", "w") as f:
                    f.write(latex_table)
                    
                with open(f"{output_dir}/interactional_markers_table_full.tex", "w") as f:
                    f.write(latex_preamble + enhanced_table + latex_closing)
                
                print(f"Interactional markers table saved to {output_dir}/interactional_markers_table_full.tex")
        
        # Table 3: Correlation matrix for metadiscourse measures
        print("Generating correlation matrix table...")
        
        # Select relevant columns for correlation analysis
        corr_columns = [
            'interactive_density', 'interactional_density', 'interactive_entropy', 'interactional_entropy'
        ]
        
        # Add individual marker categories if available
        interactive_cats = [col for col in df.columns if col.startswith('interactive_') 
                           and col not in ['interactive_total', 'interactive_density', 'interactive_entropy',
                                          'interactive_density_log', 'interactive_density_sqrt']]
        
        interactional_cats = [col for col in df.columns if col.startswith('interactional_') 
                             and col not in ['interactional_total', 'interactional_density', 'interactional_entropy',
                                            'interactional_density_log', 'interactional_density_sqrt']]
        
        # Add a subset of the individual categories to avoid making the table too large
        if len(interactive_cats) > 0:
            corr_columns.extend(interactive_cats[:min(3, len(interactive_cats))])
        
        if len(interactional_cats) > 0:
            corr_columns.extend(interactional_cats[:min(3, len(interactional_cats))])
        
        # Filter columns that exist in the dataframe
        corr_columns = [col for col in corr_columns if col in df.columns]
        
        if len(corr_columns) > 1:  # Need at least 2 columns for correlation
            # Calculate correlation matrix
            corr_matrix = df[corr_columns].corr()
            
            # Format column names for display
            formatted_columns = [col.replace('_', ' ').title() for col in corr_columns]
            corr_matrix.columns = formatted_columns
            corr_matrix.index = formatted_columns
            
            # Generate LaTeX table
            latex_table = corr_matrix.to_latex(float_format="%.2f", escape=False)
            
            # Enhance the LaTeX table
            enhanced_table = (
                "\\begin{table}[htbp]\n"
                "\\centering\n"
                "\\caption{Correlation Matrix of Metadiscourse Measures}\n"
                "\\label{tab:correlation_matrix}\n"
                "\\begin{threeparttable}\n"
                + latex_table +
                "\\begin{tablenotes}\n"
                "\\small\n"
                "\\item Note: Pearson correlation coefficients between different metadiscourse measures. "
                "Values closer to 1 indicate strong positive correlation, values closer to -1 indicate "
                "strong negative correlation, and values close to 0 indicate little or no correlation.\n"
                "\\end{tablenotes}\n"
                "\\end{threeparttable}\n"
                "\\end{table}\n"
            )
            
            # Save to file
            with open(f"{output_dir}/correlation_matrix.tex", "w") as f:
                f.write(latex_table)
                
            with open(f"{output_dir}/correlation_matrix_full.tex", "w") as f:
                f.write(latex_preamble + enhanced_table + latex_closing)
            
            print(f"Correlation matrix table saved to {output_dir}/correlation_matrix_full.tex")
        
        # Create a combined LaTeX file with all tables
        print("Creating combined LaTeX file with all tables...")
        combined_latex = latex_preamble
        
        # Add all tables
        for table_file in [
            "language_summary_table_full.tex", 
            "interactive_markers_table_full.tex", 
            "interactional_markers_table_full.tex", 
            "correlation_matrix_full.tex"
        ]:
            full_file = f"{output_dir}/{table_file}"
            if os.path.exists(full_file):
                with open(full_file, 'r') as f:
                    content = f.read()
                    # Extract just the table part (between \begin{document} and \end{document})
                    if '\\begin{document}\n' in content and '\\end{document}' in content:
                        table_part = content.split('\\begin{document}\n')[1].split('\\end{document}')[0]
                        combined_latex += table_part
        
        combined_latex += latex_closing
        
        # Save combined file
        with open(f"{output_dir}/all_tables.tex", "w") as f:
            f.write(combined_latex)
        
        print(f"Enhanced LaTeX tables have been saved to the '{output_dir}' directory")
    except Exception as e:
        print(f"Error generating LaTeX tables: {str(e)}")
        import traceback
        traceback.print_exc()

# Function to perform statistical tests between language groups and generate publication-ready tables
def perform_language_group_statistics(df, output_dir='results'):
    """
    Perform statistical significance tests to compare differences between language groups
    and generate publication-ready LaTeX tables for academic journals.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing analysis results
    output_dir : str, optional
        Directory to save statistical test results
        
    Returns:
    --------
    dict
        Dictionary with statistical test results
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if we have language groups to compare
        if 'Native_Language' not in df.columns or len(df['Native_Language'].unique()) <= 1:
            print("Not enough language groups to perform statistical tests.")
            return {}
        
        # Variables to test
        test_variables = [
            'interactive_density', 'interactional_density',
            'interactive_entropy', 'interactional_entropy'
        ]
        
        # Add individual marker categories
        interactive_cats = [col for col in df.columns if col.startswith('interactive_') 
                           and col not in ['interactive_total', 'interactive_density', 'interactive_entropy',
                                          'interactive_density_log', 'interactive_density_sqrt']]
        
        interactional_cats = [col for col in df.columns if col.startswith('interactional_') 
                             and col not in ['interactional_total', 'interactional_density', 'interactional_entropy',
                                            'interactional_density_log', 'interactional_density_sqrt']]
        
        test_variables.extend(interactive_cats)
        test_variables.extend(interactional_cats)
        
        # Dictionary to store results
        results = {}
        
        # LaTeX preamble for standalone tables
        latex_preamble = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"  # For professional tables with proper spacing
            "\\usepackage{siunitx}\n"   # For proper alignment of numbers
            "\\usepackage{caption}\n"   # For better captions
            "\\usepackage{threeparttable}\n"  # For table notes
            "\\usepackage[table]{xcolor}\n"  # For table colors
            "\\usepackage{multirow}\n"  # For merged rows
            "\\begin{document}\n"
        )
        
        # LaTeX closing
        latex_closing = "\\end{document}"
        
        # Perform ANOVA for each variable
        anova_results = []
        for var in test_variables:
            if var not in df.columns:
                continue
            
            # Create formula for ANOVA
            formula = f"{var} ~ C(Native_Language)"
            
            try:
                # Fit the model
                model = ols(formula, data=df).fit()
                
                # Perform ANOVA
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                # Get p-value
                p_value = anova_table.loc['C(Native_Language)', 'PR(>F)']
                
                # Format variable name for display
                display_name = var.replace('_', ' ').title()
                
                # Add to results
                anova_results.append({
                    'Variable': display_name,
                    'F-value': anova_table.loc['C(Native_Language)', 'F'],
                    'p-value': p_value,
                    'Significant': p_value < 0.05
                })
                
                # If significant, perform post-hoc tests
                if p_value < 0.05:
                    # Perform Tukey's HSD test
                    mc = MultiComparison(df[var], df['Native_Language'])
                    tukey_result = mc.tukeyhsd()
                    
                    # Store post-hoc results
                    posthoc_df = pd.DataFrame(data=tukey_result._results_table.data[1:], 
                                             columns=tukey_result._results_table.data[0])
                    
                    results[f'{var}_posthoc'] = posthoc_df
                    
                    # Save post-hoc results to CSV
                    posthoc_df.to_csv(f"{output_dir}/{var}_posthoc_tests.csv", index=False)
                    
                    # Generate LaTeX table for post-hoc tests
                    latex_table = posthoc_df.to_latex(index=False, float_format="%.3f", escape=False)
                    
                    # Enhance the LaTeX table with proper formatting
                    enhanced_table = (
                        "\\begin{table}[htbp]\n"
                        "\\centering\n"
                        f"\\caption{{Post-hoc Tests (Tukey's HSD) for {display_name}}}\n"
                        f"\\label{{tab:posthoc_{var}}}\n"
                        "\\begin{threeparttable}\n"
                        + latex_table +
                        "\\begin{tablenotes}\n"
                        "\\small\n"
                        "\\item Note: Results of Tukey's HSD post-hoc tests comparing language groups. "
                        "'meandiff' represents the mean difference between groups, "
                        "'p-adj' is the adjusted p-value, and 'reject' indicates whether the null hypothesis "
                        "of no difference is rejected at the 0.05 significance level.\n"
                        "\\end{tablenotes}\n"
                        "\\end{threeparttable}\n"
                        "\\end{table}\n"
                    )
                    
                    # Save to file - both standalone and embedded versions
                    with open(f"{output_dir}/{var}_posthoc_tests.tex", "w") as f:
                        f.write(latex_table)
                        
                    with open(f"{output_dir}/{var}_posthoc_tests_full.tex", "w") as f:
                        f.write(latex_preamble + enhanced_table + latex_closing)
            except Exception as e:
                print(f"Error performing ANOVA for {var}: {e}")
        
        # Create DataFrame for ANOVA results
        anova_df = pd.DataFrame(anova_results)
        
        # Save ANOVA results to CSV
        anova_df.to_csv(f"{output_dir}/anova_results.csv", index=False)
        
        # Generate LaTeX table for ANOVA results
        latex_table = anova_df.to_latex(index=False, float_format="%.3f", escape=False)
        
        # Enhance the LaTeX table with proper formatting
        enhanced_table = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{ANOVA Results for Differences Between Language Groups}\n"
            "\\label{tab:anova_results}\n"
            "\\begin{threeparttable}\n"
            + latex_table +
            "\\begin{tablenotes}\n"
            "\\small\n"
            "\\item Note: Results of one-way ANOVA tests comparing metadiscourse measures across "
            "different native language groups. Significant results (p < 0.05) indicate "
            "differences between at least two language groups.\n"
            "\\end{tablenotes}\n"
            "\\end{threeparttable}\n"
            "\\end{table}\n"
        )
        
        # Save to file - both standalone and embedded versions
        with open(f"{output_dir}/anova_results.tex", "w") as f:
            f.write(latex_table)
            
        with open(f"{output_dir}/anova_results_full.tex", "w") as f:
            f.write(latex_preamble + enhanced_table + latex_closing)
        
        # Add ANOVA results to the main results dictionary
        results['anova'] = anova_df
        
        # Create a summary text file
        with open(f"{output_dir}/statistical_tests_summary.txt", "w") as f:
            f.write("Statistical Tests for Language Group Differences\n")
            f.write("==============================================\n\n")
            
            f.write("ANOVA Results:\n")
            f.write("-------------\n")
            for _, row in anova_df.iterrows():
                sig_str = "Significant" if row['Significant'] else "Not significant"
                f.write(f"{row['Variable']}: F = {row['F-value']:.3f}, p = {row['p-value']:.3f} ({sig_str})\n")
            
            f.write("\nPost-hoc Tests (Tukey's HSD):\n")
            f.write("---------------------------\n")
            for var in test_variables:
                if f'{var}_posthoc' in results:
                    f.write(f"\n{var.replace('_', ' ').title()}:\n")
                    posthoc_df = results[f'{var}_posthoc']
                    for _, row in posthoc_df.iterrows():
                        sig_str = "Significant" if row['reject'] else "Not significant"
                        f.write(f"{row['group1']} vs {row['group2']}: diff = {row['meandiff']:.3f}, p = {row['p-adj']:.3f} ({sig_str})\n")
        
        # Create a combined LaTeX file with all statistical tables
        combined_latex = latex_preamble
        
        # Add ANOVA table
        if os.path.exists(f"{output_dir}/anova_results_full.tex"):
            with open(f"{output_dir}/anova_results_full.tex", 'r') as f:
                content = f.read()
                # Extract just the table part (between \begin{document} and \end{document})
                if '\\begin{document}\n' in content and '\\end{document}' in content:
                    table_part = content.split('\\begin{document}\n')[1].split('\\end{document}')[0]
                    combined_latex += table_part
        
        # Add all post-hoc tables
        for var in test_variables:
            full_file = f"{output_dir}/{var}_posthoc_tests_full.tex"
            if os.path.exists(full_file):
                with open(full_file, 'r') as f:
                    content = f.read()
                    # Extract just the table part (between \begin{document} and \end{document})
                    if '\\begin{document}\n' in content and '\\end{document}' in content:
                        table_part = content.split('\\begin{document}\n')[1].split('\\end{document}')[0]
                        combined_latex += table_part
        
        combined_latex += latex_closing
        
        # Save combined file
        with open(f"{output_dir}/all_statistical_tables.tex", "w") as f:
            f.write(combined_latex)
        
        print(f"Statistical test results have been saved to the '{output_dir}' directory")
        return results
    except Exception as e:
        print(f"Error performing statistical tests: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def analyze_metadiscourse_distribution(df, output_dir='results', z_score_threshold=3.0):
    """
    Analyze the distribution of metadiscourse markers and detect outliers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing analysis results
    output_dir : str, optional
        Directory to save analysis results
    z_score_threshold : float, optional
        Threshold for outlier detection using z-scores
        
    Returns:
    --------
    tuple
        (filtered_df, outliers_df) - DataFrames with filtered data and outliers
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("Analyzing metadiscourse distribution and detecting outliers...")
        
        # Check if we have sufficient data for analysis
        if len(df) < 3:
            print("Warning: Not enough data for meaningful distribution analysis.")
            return df, None
        
        # Get category columns for outlier detection
        category_cols = [col for col in df.columns if col.endswith('_density') 
                        and col not in ['interactive_density', 'interactional_density']]
        
        # Add main density measures
        analysis_cols = ['interactive_density', 'interactional_density']
        if category_cols:
            analysis_cols.extend(category_cols)
        
        # Calculate z-scores for each column
        z_scores = pd.DataFrame()
        for col in analysis_cols:
            if col in df.columns:
                z_scores[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
        
        # Identify outliers
        outlier_mask = (z_scores.abs() > z_score_threshold).any(axis=1)
        outliers_df = df[outlier_mask].copy()
        filtered_df = df[~outlier_mask].copy()
        
        # Save outlier information
        if len(outliers_df) > 0:
            outliers_df.to_csv(os.path.join(output_dir, 'outliers.csv'), index=False)
            
            # Create a summary of outliers
            with open(os.path.join(output_dir, 'outlier_summary.txt'), 'w') as f:
                f.write("Outlier Analysis Summary\n")
                f.write("=======================\n\n")
                f.write(f"Total texts analyzed: {len(df)}\n")
                f.write(f"Number of outliers detected: {len(outliers_df)} ({len(outliers_df)/len(df)*100:.1f}%)\n\n")
                
                f.write("Outlier details:\n")
                for idx, row in outliers_df.iterrows():
                    f.write(f"\nText: {row.get('Filename', f'ID: {idx}')}\n")
                    if 'Native_Language' in row:
                        f.write(f"Native Language: {row['Native_Language']}\n")
                    
                    # Find which measures are outliers
                    outlier_measures = []
                    for col in analysis_cols:
                        z_col = f'{col}_zscore'
                        if z_col in z_scores.columns and abs(z_scores.loc[idx, z_col]) > z_score_threshold:
                            outlier_measures.append(f"{col}: {row[col]:.2f} (z-score: {z_scores.loc[idx, z_col]:.2f})")
                    
                    f.write("Outlier measures:\n  " + "\n  ".join(outlier_measures) + "\n")
            
            print(f"Outlier analysis saved to {os.path.join(output_dir, 'outlier_summary.txt')}")
            
            # Visualize outliers
            if len(df) > 5:  # Need enough data for meaningful visualization
                # Box plots with outliers highlighted
                for measure in ['interactive_density', 'interactional_density']:
                    if measure in df.columns:
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(y=df[measure])
                        # Add outlier points in red
                        outlier_points = outliers_df[measure]
                        if not outlier_points.empty:
                            plt.scatter(x=np.zeros_like(outlier_points), y=outlier_points, 
                                       color='red', s=50, label='Outliers')
                        plt.title(f'Distribution of {measure.replace("_", " ").title()} with Outliers')
                        plt.ylabel('Frequency per 1000 words')
                        if outlier_points.empty is False:
                            plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f'{measure}_outliers.png'))
                        plt.close()
        else:
            print("No outliers detected with the current threshold.")
        
        # Calculate and save distribution statistics
        stats_df = df[analysis_cols].describe()
        stats_df.to_csv(os.path.join(output_dir, 'distribution_statistics.csv'))
        
        # Save filtered dataset
        if len(filtered_df) < len(df):
            filtered_df.to_csv(os.path.join(output_dir, 'filtered_data.csv'), index=False)
            print(f"Filtered dataset (with outliers removed) saved to {os.path.join(output_dir, 'filtered_data.csv')}")
        
        # Create distribution plots for key measures
        for measure in ['interactive_density', 'interactional_density']:
            if measure in df.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[measure], kde=True)
                plt.axvline(df[measure].mean(), color='red', linestyle='--', label='Mean')
                plt.axvline(df[measure].median(), color='green', linestyle='-.', label='Median')
                plt.title(f'Distribution of {measure.replace("_", " ").title()}')
                plt.xlabel('Frequency per 1000 words')
                plt.ylabel('Count')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{measure}_distribution.png'))
                plt.close()
        
        # If we have language groups, analyze distribution by group
        if 'Native_Language' in df.columns and len(df['Native_Language'].unique()) > 1:
            # Violin plots for main measures by language
            for measure in ['interactive_density', 'interactional_density']:
                if measure in df.columns:
                    plt.figure(figsize=(12, 6))
                    lang_means = df.groupby('Native_Language')[measure].mean()
                    lang_means.plot(kind='bar')
                    plt.title(f'Distribution of {measure.replace("_", " ").title()} by Native Language')
                    plt.ylabel('Frequency per 1000 words')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'{measure}_by_language_violin.png'))
                    plt.close()
        
        print(f"Distribution analysis has been saved to the '{output_dir}' directory")
        return filtered_df, outliers_df
    
    except Exception as e:
        print(f"Error analyzing metadiscourse distribution: {str(e)}")
        import traceback
        traceback.print_exc()
        return df, None

def generate_visualizations(df, output_dir='results'):
    """
    Generate visualizations for metadiscourse analysis results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing analysis results
    output_dir : str, optional
        Directory to save visualizations
        
    Returns:
    --------
    None
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating visualizations...")
        
        # Ensure we have Native_Language column
        native_lang_col = "Native_Language"
        if native_lang_col not in df.columns:
            if "l1" in df.columns:
                native_lang_col = "l1"
                df["Native_Language"] = df["l1"]
            else:
                print(f"Warning: '{native_lang_col}' column not found. Creating dummy column.")
                df["Native_Language"] = "unknown"
        
        # Check if we have sufficient data for analysis
        if len(df) < 2:
            print("Warning: Not enough data for meaningful analysis.")
            return
        
        # Get category columns
        category_cols = [col for col in df.columns if col.endswith('_density') 
                        and not col in ['interactive_density', 'interactional_density']]
        
        # 1. Interactive vs. Interactional overall
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[['interactive_density', 'interactional_density']])
        plt.title('Distribution of Interactive vs. Interactional Metadiscourse')
        plt.ylabel('Frequency per 1000 words')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metadiscourse_type_distribution.png'))
        plt.close()
        
        # 2. Metadiscourse categories overall
        if category_cols:
            plt.figure(figsize=(12, 8))
            plot_data = df[[col for col in category_cols if col in df.columns]]
            # Reorder columns by mean value for better visualization
            plot_data = plot_data[plot_data.mean().sort_values(ascending=False).index]
            sns.boxplot(data=plot_data)
            plt.title('Distribution of Metadiscourse Categories')
            plt.ylabel('Frequency per 1000 words')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metadiscourse_categories_distribution.png'))
            plt.close()
        
        # 3. By Native Language (if available)
        if len(df[native_lang_col].unique()) > 1:
            # Interactive vs. Interactional by Native Language
            plt.figure(figsize=(12, 6))
            lang_means = df.groupby(native_lang_col)[['interactive_density', 'interactional_density']].mean()
            lang_means.plot(kind='bar')
            plt.title('Interactive vs. Interactional Metadiscourse by Native Language')
            plt.ylabel('Frequency per 1000 words')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metadiscourse_type_by_language.png'))
            plt.close()
            
            # Top 5 categories by Native Language
            if category_cols:
                plt.figure(figsize=(14, 8))
                # Get top 5 categories by overall mean
                top_categories = df[category_cols].mean().sort_values(ascending=False).head(5).index
                if len(top_categories) > 0:
                    top_cat_data = df.groupby(native_lang_col)[top_categories].mean()
                    top_cat_data.plot(kind='bar')
                    plt.title('Top 5 Metadiscourse Categories by Native Language')
                    plt.ylabel('Frequency per 1000 words')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'top_metadiscourse_categories.png'))
                    plt.close()
            
            # Heatmap of all categories
            if category_cols and len(category_cols) > 1:
                plt.figure(figsize=(16, 10))
                category_means = df.groupby(native_lang_col)[category_cols].mean()
                sns.heatmap(category_means, annot=True, cmap='YlGnBu', fmt='.2f')
                plt.title('Metadiscourse Markers by Category and Native Language')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'metadiscourse_heatmap.png'))
                plt.close()
        
        # 4. Entropy visualization
        if 'interactive_entropy' in df.columns and 'interactional_entropy' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df[['interactive_entropy', 'interactional_entropy']])
            plt.title('Distribution of Entropy Values for Metadiscourse Types')
            plt.ylabel('Shannon Entropy')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metadiscourse_entropy_distribution.png'))
            plt.close()
            
            # Entropy by language group
            if len(df[native_lang_col].unique()) > 1:
                plt.figure(figsize=(12, 6))
                entropy_means = df.groupby(native_lang_col)[['interactive_entropy', 'interactional_entropy']].mean()
                entropy_means.plot(kind='bar')
                plt.title('Metadiscourse Entropy by Native Language')
                plt.ylabel('Shannon Entropy')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'metadiscourse_entropy_by_language.png'))
                plt.close()
        
        # 5. Correlation heatmap
        if len(df) > 3:  # Need at least a few samples for correlation
            # Select columns for correlation analysis
            corr_cols = ['interactive_density', 'interactional_density']
            if 'interactive_entropy' in df.columns:
                corr_cols.append('interactive_entropy')
            if 'interactional_entropy' in df.columns:
                corr_cols.append('interactional_entropy')
            if 'word_count' in df.columns:
                corr_cols.append('word_count')
            
            # Add top marker categories
            if category_cols:
                top_cats = df[category_cols].mean().sort_values(ascending=False).head(5).index.tolist()
                corr_cols.extend(top_cats)
            
            # Generate correlation matrix
            plt.figure(figsize=(12, 10))
            corr_matrix = df[corr_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Correlation Matrix of Metadiscourse Measures')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metadiscourse_correlation_matrix.png'))
            plt.close()
        
        # 6. Scatter plot of word count vs. metadiscourse density
        if 'word_count' in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Interactive density vs. word count
            sns.scatterplot(x='word_count', y='interactive_density', hue=native_lang_col, data=df, ax=axes[0])
            axes[0].set_title('Interactive Metadiscourse vs. Document Length')
            axes[0].set_xlabel('Word Count')
            axes[0].set_ylabel('Interactive Density (per 1000 words)')
            
            # Interactional density vs. word count
            sns.scatterplot(x='word_count', y='interactional_density', hue=native_lang_col, data=df, ax=axes[1])
            axes[1].set_title('Interactional Metadiscourse vs. Document Length')
            axes[1].set_xlabel('Word Count')
            axes[1].set_ylabel('Interactional Density (per 1000 words)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metadiscourse_vs_word_count.png'))
            plt.close()
        
        print(f"Visualizations have been saved to the '{output_dir}' directory")
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()

def calculate_shannon_entropy(df, output_dir='results'):
    """
    Calculate Shannon entropy for metadiscourse markers to analyze diversity and distribution patterns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing analysis results
    output_dir : str, optional
        Directory to save entropy analysis results
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with entropy values added
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("Calculating Shannon entropy for metadiscourse markers...")
        
        # Check if we have sufficient data for analysis
        if len(df) < 2:
            print("Warning: Not enough data for meaningful entropy analysis.")
            return df
        
        # Create a copy of the dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Get interactive and interactional category columns
        interactive_cols = [col for col in df.columns if col.endswith('_density') 
                           and col not in ['interactive_density', 'interactional_density']
                           and any(cat in col for cat in ['transitions', 'frame_markers', 'endophoric_markers', 
                                                        'evidentials', 'code_glosses'])]
        
        interactional_cols = [col for col in df.columns if col.endswith('_density') 
                             and col not in ['interactive_density', 'interactional_density']
                             and any(cat in col for cat in ['hedges', 'boosters', 'attitude_markers', 
                                                          'self_mentions', 'engagement_markers'])]
        
        # Function to calculate entropy from a series of values
        def entropy(values):
            # Filter out zeros and NaNs
            values = [v for v in values if v > 0 and not pd.isna(v)]
            if not values:
                return 0.0
                
            # Normalize to get probabilities
            total = sum(values)
            probabilities = [v / total for v in values]
            
            # Calculate Shannon entropy: -sum(p * log(p))
            return -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Calculate entropy for each text's interactive markers
        if interactive_cols:
            result_df['interactive_entropy'] = df[interactive_cols].apply(
                lambda row: entropy(row.values), axis=1
            )
        
        # Calculate entropy for each text's interactional markers
        if interactional_cols:
            result_df['interactional_entropy'] = df[interactional_cols].apply(
                lambda row: entropy(row.values), axis=1
            )
        
        # Calculate overall metadiscourse entropy
        all_marker_cols = interactive_cols + interactional_cols
        if all_marker_cols:
            result_df['total_metadiscourse_entropy'] = df[all_marker_cols].apply(
                lambda row: entropy(row.values), axis=1
            )
        
        # Calculate entropy statistics by language group if available
        if 'Native_Language' in df.columns and len(df['Native_Language'].unique()) > 1:
            entropy_stats = result_df.groupby('Native_Language')[
                [col for col in ['interactive_entropy', 'interactional_entropy', 'total_metadiscourse_entropy'] 
                 if col in result_df.columns]
            ].agg(['mean', 'std', 'min', 'max'])
            
            # Save entropy statistics by language
            entropy_stats.to_csv(os.path.join(output_dir, 'entropy_by_language.csv'))
            
            # Create a summary report
            with open(os.path.join(output_dir, 'entropy_analysis.txt'), 'w') as f:
                f.write("Entropy Analysis Summary\n")
                f.write("======================\n\n")
                f.write("Shannon entropy measures the diversity of metadiscourse markers.\n")
                f.write("Higher values indicate more even distribution across marker categories.\n\n")
                
                f.write("Overall Entropy Statistics:\n")
                for col in ['interactive_entropy', 'interactional_entropy', 'total_metadiscourse_entropy']:
                    if col in result_df.columns:
                        f.write(f"\n{col.replace('_', ' ').title()}:\n")
                        f.write(f"  Mean: {result_df[col].mean():.3f}\n")
                        f.write(f"  Std Dev: {result_df[col].std():.3f}\n")
                        f.write(f"  Min: {result_df[col].min():.3f}\n")
                        f.write(f"  Max: {result_df[col].max():.3f}\n")
                
                f.write("\nEntropy by Language Group:\n")
                f.write(entropy_stats.to_string())
                
                # Add interpretation
                f.write("\n\nInterpretation:\n")
                f.write("- Higher entropy values suggest more diverse use of metadiscourse markers\n")
                f.write("- Lower entropy values suggest concentration on fewer marker types\n")
                f.write("- Comparing entropy across language groups can reveal differences in metadiscourse strategies\n")
        
        print(f"Entropy analysis has been saved to the '{output_dir}' directory")
        return result_df
    
    except Exception as e:
        print(f"Error calculating Shannon entropy: {str(e)}")
        import traceback
        traceback.print_exc()
        return df

def main():
    parser = argparse.ArgumentParser(description='Analyze metadiscourse markers in texts.')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with analysis results or directory with text files')
    parser.add_argument('--language_map', type=str, help='CSV file mapping filenames to native languages')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if input is a CSV file or a directory
    if args.input.endswith('.csv'):
        print(f"Loading analysis results from {args.input}...")
        df = pd.read_csv(args.input)
        print(f"Loaded data for {len(df)} texts.")
    else:
        # Check if input is a directory with text files
        if os.path.isdir(args.input):
            print(f"Processing text files in {args.input}...")
            
            # Load language map if provided
            language_map = None
            if args.language_map:
                language_map = load_language_map(args.language_map)
                print(f"Loaded language map with {len(language_map)} entries.")
            
            # Process corpus
            df = load_corpus(args.input, language_map)
            print(f"Processed {len(df)} texts.")
            
            # Save analysis results to CSV
            df.to_csv(os.path.join(args.output_dir, 'metalanguage_analysis_results.csv'), index=False)
            print(f"Analysis results saved to {os.path.join(args.output_dir, 'metalanguage_analysis_results.csv')}")
        else:
            print(f"Error: Input {args.input} is not a valid CSV file or directory.")
            return
    
    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    generate_latex_tables(df, args.output_dir)
    print(f"Enhanced LaTeX tables have been saved to the '{args.output_dir}' directory")
    
    # Generate visualizations
    print("Generating visualizations...")
    generate_visualizations(df, args.output_dir)
    print(f"Visualizations have been saved to the '{args.output_dir}' directory")
    
    # Perform statistical tests
    print("Performing statistical tests for language group differences...")
    perform_language_group_statistics(df, args.output_dir)
    print(f"Statistical test results have been saved to the '{args.output_dir}' directory")
    
    # Analyze metadiscourse distribution
    print("Analyzing metadiscourse distribution...")
    analyze_metadiscourse_distribution(df, args.output_dir)
    print(f"Distribution analysis has been saved to the '{args.output_dir}' directory")
    
    # Calculate Shannon entropy
    print("Calculating Shannon entropy...")
    calculate_shannon_entropy(df, args.output_dir)
    print(f"Entropy analysis has been saved to the '{args.output_dir}' directory")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()

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
