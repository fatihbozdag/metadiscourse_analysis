import json
import re

def python_to_notebook(python_file, notebook_file):
    """Convert a Python file to a Jupyter notebook."""
    with open(python_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the content into cells based on logical sections
    cells = []
    
    # Add a markdown cell with title and description
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Metadiscourse Analysis Notebook\n",
            "\n",
            "This notebook provides a comprehensive analysis of metadiscourse markers in academic texts, based on Hyland's (2005) framework.\n",
            "\n",
            "The analysis includes:\n",
            "- Detection of interactive and interactional metadiscourse markers\n",
            "- Statistical analysis of language group differences\n",
            "- Visualization of metadiscourse patterns\n",
            "- Shannon entropy calculation for metadiscourse diversity\n",
            "- Generation of publication-ready LaTeX tables"
        ]
    })
    
    # Split the Python file into logical sections
    sections = [
        # Imports section
        r"import.*?(?=\n\n# Load and preprocess data)",
        # Data loading section
        r"# Load and preprocess data.*?(?=\n\n# Attempt to use MPS)",
        # GPU setup section
        r"# Attempt to use MPS.*?(?=\n\n# Define comprehensive metadiscourse marker)",
        # Interactive markers definition
        r"# Define comprehensive metadiscourse marker.*?(?=\nINTERACTIONAL_MARKERS)",
        # Interactional markers definition
        r"INTERACTIONAL_MARKERS.*?(?=\n\n# Context rules)",
        # Context patterns definition
        r"# Context rules.*?(?=\n\n# Function to generate LaTeX tables)",
        # LaTeX tables function
        r"# Function to generate LaTeX tables.*?(?=\n\ndef perform_language_group_statistics)",
        # Statistical analysis function
        r"def perform_language_group_statistics.*?(?=\n\ndef generate_visualizations)",
        # Visualizations function
        r"def generate_visualizations.*?(?=\n\ndef analyze_metadiscourse_distribution)",
        # Distribution analysis function
        r"def analyze_metadiscourse_distribution.*?(?=\n\ndef calculate_shannon_entropy)",
        # Entropy calculation function
        r"def calculate_shannon_entropy.*?(?=\n\ndef main\(\))",
        # Main function
        r"def main\(\).*?(?=\n\nif __name__)",
        # Entry point
        r"if __name__.*?(?=\n\n# Register the metadiscourse detector)",
        # Metadiscourse detector component
        r"# Register the metadiscourse detector.*?(?=\n\ndef analyze_text)",
        # Text analysis function
        r"def analyze_text.*?(?=\n\ndef load_corpus)",
        # Corpus loading function
        r"def load_corpus.*?(?=\n\ndef load_language_map)",
        # Language map loading function
        r"def load_language_map.*?$"
    ]
    
    # Use regex to extract each section
    for pattern in sections:
        try:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                section_content = match.group(0)
                # Add as a code cell
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": section_content.split('\n')
                })
        except Exception as e:
            print(f"Error processing section {pattern}: {e}")
    
    # Create the notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write the notebook to a file
    with open(notebook_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Converted {python_file} to {notebook_file}")

if __name__ == "__main__":
    python_file = "analyze_metalanguage_fixed.py"
    notebook_file = "analyze_metalanguage_notebook.ipynb"
    python_to_notebook(python_file, notebook_file)
