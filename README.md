# Metadiscourse Analysis

A Python project for analyzing metadiscourse markers in academic texts using Hyland's (2005) framework, with enhanced pattern matching and accuracy improvements.

## Features

- Identifies interactive and interactional metadiscourse markers with hierarchical categorization
- Processes large corpora of academic texts
- Generates statistical analysis and visualizations
- Supports both CSV and text file inputs
- **Enhanced pattern matching** for improved marker detection
- **Handles polyfunctional markers** that belong to multiple categories
- **Improved text preprocessing** for better handling of contractions and special cases
- **Robust error handling** with detailed error reporting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/metadiscourse-analysis.git
cd metadiscourse-analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the spaCy model:
```bash
python -m spacy download en_core_web_trf
```

## Usage

1. Place your input data in the `data` directory:
   - For CSV files: name it `metadata_with_text.csv` and ensure it has a text column
   - For text files: place them directly in the `data` directory

2. Run the analysis:
```bash
python src/main.py
```

Optional arguments:
- `--input_dir`: Directory containing input data (default: 'data')
- `--output_dir`: Directory for saving results (default: 'results')
- `--model`: spaCy model to use (default: 'en_core_web_trf')
- `--text_field`: Name of the column containing text to analyze (default: 'text_field')

## Output

The analysis produces the following outputs in the specified output directory:

1. `metadiscourse_analysis.csv`: Contains the analysis results for each document, including:
   - Document identifier
   - Word count
   - Counts and frequencies of each marker category
   - Any metadata from the original input file

2. Visualizations in the `visualizations` subdirectory:
   - Distribution plots of marker frequencies
   - Correlation heatmaps between marker categories
   - Other statistical visualizations

3. `processing_errors.csv`: If any errors occur during processing, details are saved in this file

## Marker Categories

The analysis follows Hyland's (2005) framework, which categorizes metadiscourse markers into:

### Interactive Markers
- **Transitions**: Express semantic relations between main clauses (e.g., "moreover", "therefore", "however")
- **Frame Markers**: Signal text boundaries or discourse acts (e.g., "firstly", "to conclude", "my purpose is")
- **Endophoric Markers**: Refer to other parts of the text (e.g., "in section 2", "as noted above")
- **Evidentials**: Refer to sources of information from other texts (e.g., "according to X", "Z states")
- **Code Glosses**: Help readers grasp meanings of ideational material (e.g., "namely", "such as", "in other words")

### Interactional Markers
- **Hedges**: Withhold writer's full commitment to proposition (e.g., "might", "perhaps", "possible")
- **Boosters**: Emphasize force or writer's certainty (e.g., "clearly", "obviously", "demonstrate")
- **Attitude Markers**: Express writer's attitude to proposition (e.g., "unfortunately", "surprisingly")
- **Engagement Markers**: Explicitly address readers (e.g., "consider", "note that", "you can see")
- **Self Mentions**: Explicit reference to author(s) (e.g., "I", "we", "my", "our")

## Advanced Features

### Polyfunctional Markers

Some markers can serve multiple functions depending on context. The system now handles these cases by counting them in all relevant categories. Examples include:

- "in fact": Both a code gloss and a booster
- "then": Both a transition and a frame marker
- "must": Both a booster and an engagement marker

### Flexible Pattern Matching

The system now employs more sophisticated pattern matching that can detect markers even when they contain:

- Intervening punctuation (e.g., "in, for example, this case")
- Additional words (e.g., "in my personal opinion" matching "in my opinion")
- Contractions and special cases (e.g., "can't", "i.e.", "et al.")

## References

Hyland, K. (2005). Metadiscourse: Exploring Interaction in Writing. Continuum.

## License

MIT