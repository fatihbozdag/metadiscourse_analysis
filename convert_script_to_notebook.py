import nbformat as nbf
import re

def create_notebook_from_script(script_path, notebook_path):
    """Convert a Python script to a Jupyter notebook with markdown cells."""
    # Read the script
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Add title and description
    cells.append(nbf.v4.new_markdown_cell("""
# Metadiscourse Analysis Notebook

This notebook provides a comprehensive framework for analyzing metadiscourse markers in academic texts based on Hyland's (2005) framework. It includes:

- Detection of interactive and interactional metadiscourse markers
- Statistical analysis of language group differences
- Visualization of metadiscourse patterns
- Shannon entropy calculation for metadiscourse diversity
- Generation of publication-ready LaTeX tables

## How to Use This Notebook

1. Run the cells in order
2. Provide your input data either as a CSV file or a directory of text files
3. Examine the visualizations and statistical results

The analysis will be saved in the `analysis_results` directory.
    """))
    
    # Define sections to split the script
    sections = [
        ("## Import Libraries", r"# Import necessary libraries[\s\S]*?(?=\n# Configure patterns)"),
        ("## Configure Settings", r"# Configure patterns[\s\S]*?(?=\n# Attempt to use MPS)"),
        ("## GPU Setup", r"# Attempt to use MPS[\s\S]*?(?=\n# Define comprehensive metadiscourse marker)"),
        ("## Define Metadiscourse Markers", r"# Define comprehensive metadiscourse marker[\s\S]*?(?=\n# Context rules)"),
        ("## Context Rules for Filtering", r"# Context rules[\s\S]*?(?=\n# Register the metadiscourse detector)"),
        ("## Metadiscourse Detector Component", r"# Register the metadiscourse detector[\s\S]*?(?=\ndef analyze_text)"),
        ("## Text Analysis Function", r"def analyze_text[\s\S]*?(?=\ndef load_corpus)"),
        ("## Corpus Loading Function", r"def load_corpus[\s\S]*?(?=\ndef load_language_map)"),
        ("## Language Map Loading Function", r"def load_language_map[\s\S]*?(?=\n# Function to generate LaTeX tables)"),
        ("## LaTeX Table Generation", r"# Function to generate LaTeX tables[\s\S]*?(?=\ndef perform_language_group_statistics)"),
        ("## Statistical Analysis", r"def perform_language_group_statistics[\s\S]*?(?=\ndef generate_visualizations)"),
        ("## Visualization Functions", r"def generate_visualizations[\s\S]*?(?=\ndef analyze_metadiscourse_distribution)"),
        ("## Distribution Analysis", r"def analyze_metadiscourse_distribution[\s\S]*?(?=\ndef calculate_shannon_entropy)"),
        ("## Shannon Entropy Calculation", r"def calculate_shannon_entropy[\s\S]*?(?=\ndef main\(\))"),
        ("## Main Function", r"def main\(\)[\s\S]*?(?=\nif __name__)"),
        ("## Example Usage", r"if __name__[\s\S]*")
    ]
    
    # Extract each section and add to the notebook
    for title, pattern in sections:
        # Add section title as markdown
        cells.append(nbf.v4.new_markdown_cell(title))
        
        # Extract code for this section
        match = re.search(pattern, script_content)
        if match:
            code = match.group(0)
            cells.append(nbf.v4.new_code_cell(code))
        else:
            cells.append(nbf.v4.new_code_cell("# Section not found"))
    
    # Add a cell for loading data
    cells.append(nbf.v4.new_markdown_cell("""
## Data Loading and Analysis

Run the following cell to load your data and perform the analysis. You can modify the parameters as needed.
    """))
    
    cells.append(nbf.v4.new_code_cell("""
# Example: Load data from a CSV file
# Replace with your own CSV file path
csv_path = 'your_data.csv'

# Load and preprocess data
meta = pd.read_csv(csv_path)
text = pd.read_csv(csv_path)
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

# Save raw results
output_dir = 'analysis_results'
os.makedirs(output_dir, exist_ok=True)
results_csv = os.path.join(output_dir, 'metadiscourse_analysis_results.csv')
results_df.to_csv(results_csv, index=False)
print(f"Raw analysis results saved to {results_csv}")

# Generate all analyses
generate_latex_tables(results_df, output_dir)
perform_language_group_statistics(results_df, output_dir)
generate_visualizations(results_df, output_dir)
distribution_summary = analyze_metadiscourse_distribution(results_df, output_dir)
entropy_summary = calculate_shannon_entropy(results_df, output_dir)

print(f"All analysis results saved to {output_dir}")
    """))
    
    # Add a cell for displaying results
    cells.append(nbf.v4.new_markdown_cell("""
## Displaying Results

Run the following cells to display some of the results directly in the notebook.
    """))
    
    cells.append(nbf.v4.new_code_cell("""
# Display summary statistics
print("=== METADISCOURSE ANALYSIS SUMMARY ===\n")

# Basic statistics
print(f"Total texts analyzed: {len(results_df)}")
if 'Native_Language' in results_df.columns:
    print(f"Language groups: {', '.join(results_df['Native_Language'].unique())}")

# Distribution summary
print("\n--- Metadiscourse Distribution ---")
print(f"Interactive markers: {distribution_summary['interactive_percentage']:.2f}%")
print(f"Interactional markers: {distribution_summary['interactional_percentage']:.2f}%")
print(f"Most frequent category: {distribution_summary['most_frequent_category']} ({distribution_summary['most_frequent_percentage']:.2f}%)")
print(f"Least frequent category: {distribution_summary['least_frequent_category']} ({distribution_summary['least_frequent_percentage']:.2f}%)")

# Entropy summary
print("\n--- Metadiscourse Diversity (Shannon Entropy) ---")
print(f"Mean normalized entropy: {entropy_summary['mean_normalized_entropy']:.4f} (0-1 scale)")
print(f"Max possible entropy: {entropy_summary['max_entropy']:.4f} bits")

# Language-specific entropy if available
if 'language_stats' in entropy_summary:
    print("\nEntropy by language group:")
    for lang, stats in entropy_summary['language_stats'].items():
        print(f"  {lang}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    if 'entropy_anova_p' in entropy_summary:
        print(f"\nANOVA test for entropy differences: F={entropy_summary['entropy_anova_f']:.2f}, p={entropy_summary['entropy_anova_p']:.4f}", end='')
        if entropy_summary.get('entropy_significant_diff', False):
            print(" (significant)")
        else:
            print(" (not significant)")
    """))
    
    # Add a cell for displaying visualizations
    cells.append(nbf.v4.new_markdown_cell("""
## Displaying Visualizations

Run the following cells to display some of the visualizations directly in the notebook.
    """))
    
    cells.append(nbf.v4.new_code_cell("""
# Display interactive vs. interactional scatter plot
plt.figure(figsize=(10, 8))

if 'Native_Language' in results_df.columns:
    scatter = sns.scatterplot(
        data=results_df,
        x='interactive_density',
        y='interactional_density',
        hue='Native_Language',
        s=100,
        alpha=0.7
    )
else:
    scatter = sns.scatterplot(
        data=results_df,
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
    data=results_df,
    scatter=False,
    ci=None,
    line_kws={"color": "red", "lw": 2, "linestyle": "--"}
)

plt.tight_layout()
plt.show()
    """))
    
    cells.append(nbf.v4.new_code_cell("""
# Display marker distribution pie chart
# Calculate total markers for each category
interactive_categories = ["code_glosses", "endophoric_markers", "evidentials", "frame_markers", "transition_markers"]
interactional_categories = ["attitude_markers", "self_mention", "engagement_markers", "hedges", "boosters"]

# Get all marker categories that exist in the dataframe
categories = [cat for cat in interactive_categories + interactional_categories if cat in results_df.columns]

# Calculate total markers for each category
total_markers = {category: results_df[category].sum() for category in categories}

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
plt.show()
    """))
    
    # Add the cells to the notebook
    nb['cells'] = cells
    
    # Write the notebook to a file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Notebook created: {notebook_path}")

if __name__ == "__main__":
    script_path = "metadiscourse_analysis.py"
    notebook_path = "metadiscourse_analysis_notebook.ipynb"
    create_notebook_from_script(script_path, notebook_path)
