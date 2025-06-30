# Evidentiality and Metadiscourse Marker Extraction

This project extracts evidentiality and metadiscourse markers from the TICLE (Turkish International Corpus of Learner English) and LOCNESS (Louvain Corpus of Native English Essays) corpora to support comparative analysis of how Turkish learners of English and native English writers signal information sources and express epistemic stance in academic writing.

## Project Overview

This tool builds upon the existing metadiscourse analysis framework to extract and analyze:

1. **Evidentiality Markers**:
   - Direct perception markers (visual, auditory, sensory)
   - Inference markers (deductive, assumptive, speculative)
   - Reportative markers (quotative, hearsay, citation)
   - Knowledge/belief markers (personal knowledge, belief, doubt)

2. **Metadiscourse Markers** (following Hyland's framework):
   - Interactive markers (transitions, frame markers, endophoric markers, evidentials, code glosses)
   - Interactional markers (hedges, boosters, attitude markers, self-mentions, engagement markers)

3. **Mental Space Builders**:
   - Belief spaces
   - Speech spaces
   - Hypothetical spaces
   - Time/place spaces
   - Possibility/probability spaces

## Installation

1. Clone the repository or navigate to the project directory:
   ```
   cd evidentiality_project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the spaCy model:
   ```
   python -m spacy download en_core_web_trf
   ```

## Usage

### Preparing Your Data

The tool expects your TICLE and LOCNESS corpora to be in CSV format with a column containing the essay text. If your data is in a different format, you can use the `data_utils.py` module to preprocess it:

```python
from data_utils import preprocess_corpus_file

# Convert TICLE corpus to standardized format
preprocess_corpus_file(
    file_path="path/to/ticle_corpus.txt",
    output_path="data/ticle_processed.csv",
    corpus_type="TICLE"
)

# Convert LOCNESS corpus to standardized format
preprocess_corpus_file(
    file_path="path/to/locness_corpus.txt",
    output_path="data/locness_processed.csv",
    corpus_type="LOCNESS"
)
```

### Running the Analysis

Run the main script with the paths to your TICLE and LOCNESS corpus files:

```bash
python src/main.py --ticle_path data/ticle_processed.csv --locness_path data/locness_processed.csv --text_field text
```

Optional arguments:
- `--output_dir`: Directory for saving results (default: "../results")
- `--model`: spaCy model to use (default: "en_core_web_trf")
- `--text_field`: Name of the column containing text to analyze (default: "text")

### Output

The tool generates the following outputs in the specified output directory:

1. **Full extraction data** (`marker_extraction_[timestamp].csv`):
   - Marker (the exact text found)
   - Marker Category (e.g., Evidentiality, Interactive, Interactional, Mental Space)
   - Marker Subcategory (e.g., direct_perception_visual, hedges_modality)
   - Full Sentence (containing the marker)
   - Context Before and After (1-2 sentences)
   - Essay ID
   - Essay Position (intro/body/conclusion)
   - Corpus Source (TICLE/LOCNESS)

2. **Summary statistics**:
   - Raw frequency counts by marker category (`raw_counts_[timestamp].csv`)
   - Normalized frequencies per 10,000 words (`pivot_norm_[timestamp].csv`)
   - Distribution across essay positions (`position_distribution_[timestamp].csv`)

3. **Visualizations**:
   - Distribution of marker categories by corpus
   - Distribution of markers across essay positions
   - Comparison of normalized frequencies between corpora
   - Heatmap of marker subcategories

## Project Structure

```
evidentiality_project/
├── src/
│   ├── main.py                     # Main script to run the analysis
│   ├── evidentiality_processor.py  # Core processor for marker extraction
│   ├── evidentiality_markers.py    # Definitions of evidentiality markers
│   └── data_utils.py               # Utilities for data preparation
├── data/                           # Directory for corpus data
├── results/                        # Directory for analysis results
└── requirements.txt                # Project dependencies
```

## Extending the Project

You can extend this project by:

1. Adding new marker categories or subcategories in `evidentiality_markers.py`
2. Enhancing the context extraction in `evidentiality_processor.py`
3. Adding more sophisticated statistical analyses
4. Implementing machine learning approaches for marker classification

## References

- Hyland, K. (2005). Metadiscourse: Exploring interaction in writing. Continuum.
- Aikhenvald, A. Y. (2004). Evidentiality. Oxford University Press.
- Fauconnier, G. (1994). Mental spaces: Aspects of meaning construction in natural language. Cambridge University Press.
