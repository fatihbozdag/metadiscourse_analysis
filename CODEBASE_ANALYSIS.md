# Metalinguistics Project - Comprehensive Codebase Analysis

## Project Overview

This is a comprehensive linguistic analysis framework for studying **metadiscourse markers** and **evidentiality** in academic texts, with a primary focus on comparing native English and learner English (Turkish L1) writing patterns. The project implements Hyland's (2005) metadiscourse framework with advanced computational linguistics techniques for marker detection and analysis.

## Project Structure

### Main Project Components

```
metalinguistics/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── analyze_metadiscourse.py     # Standalone analysis script
├── l1_tables_and_plots.py      # L1-based analysis and visualization
├── metadiscourse_final.ipynb   # Jupyter notebook for analysis workflow
├── data/                       # Input datasets
│   ├── metadata_with_text.csv  # Main corpus data (51MB)
│   ├── TICLE_sample.csv        # Turkish learner corpus sample
│   └── processing_errors.csv   # Error logs
├── src/                        # Core library modules
│   ├── main.py                 # Main entry point
│   ├── processor.py            # Text processing engine
│   ├── markers.py              # Marker definitions
│   ├── viz.py                  # Visualization tools
│   └── stats.py                # Statistical analysis
├── results/                    # Analysis outputs
└── evidentiality_project/      # Specialized subproject
    ├── src/                    # Evidentiality-specific modules
    ├── data/                   # Corpus-specific data
    ├── results/                # Evidentiality analysis results
    └── requirements.txt        # Subproject dependencies
```

## Core Framework Components

### 1. Metadiscourse Analysis Engine (`src/processor.py`)

**Primary Class**: `TextProcessor`

**Key Features**:
- Advanced spaCy-based NLP processing with GPU acceleration
- Sophisticated text cleaning and normalization
- Enhanced pattern matching for metadiscourse markers
- Polyfunctional marker handling (markers serving multiple categories)
- Context-aware marker detection with flexible boundaries
- Statistical frequency calculation (per 1000 words)

**Advanced Processing Capabilities**:
- Handles contractions and linguistic edge cases
- Preserves hyphenated words and special linguistic markers
- Implements fuzzy matching for multi-word phrases
- Processes intervening punctuation and words

### 2. Marker Taxonomy (`src/markers.py`)

**Interactive Markers** (Hyland's Framework):
- **Transitions**: Semantic relations (additive, causal, adversative, temporal)
- **Frame Markers**: Text boundaries (sequencing, conclusion, topic)
- **Endophoric Markers**: Internal text references
- **Evidentials**: External source citations
- **Code Glosses**: Meaning clarification (reformulation, exemplification)

**Interactional Markers**:
- **Hedges**: Uncertainty expression (modality, approximation, tentative)
- **Boosters**: Certainty emphasis (demonstration, absoluteness)
- **Attitude Markers**: Evaluative stance (evaluation, judgment, emotion)
- **Engagement Markers**: Reader interaction (direct address, imperatives)
- **Self Mentions**: Author presence (singular, plural, authorial)

**Advanced Features**:
- Hierarchical categorization with subcategories
- Polyfunctional marker support
- Context-sensitive pattern matching
- spaCy Matcher integration for efficient processing

### 3. Visualization Suite (`src/viz.py`)

**Visualization Capabilities**:
- Distribution analysis (box plots, bar charts)
- Correlation analysis (heatmaps)
- Group-based comparisons
- Summary statistics visualization
- Multi-level marker analysis (type, category, subcategory)

### 4. Statistical Analysis (`src/stats.py`)

**Statistical Methods**:
- ANOVA and post-hoc testing (Tukey's HSD)
- Correlation analysis
- Multiple regression analysis
- Distribution analysis with normality testing
- Marker co-occurrence analysis
- Comprehensive summary reporting

## Specialized Subprojects

### Evidentiality Project

**Purpose**: Extended analysis framework for evidentiality markers and mental space builders in comparative corpus analysis (TICLE vs LOCNESS).

**Key Features**:
- Evidentiality marker extraction (direct perception, inference, reportative, knowledge/belief)
- Mental space builder identification
- Cross-corpus comparative analysis
- Position-based analysis (intro/body/conclusion)
- Normalized frequency calculations

**Output Formats**:
- Full extraction with context
- Statistical summaries
- Comparative visualizations
- Academic-ready tables (Markdown/LaTeX)

## Data Processing Pipeline

### 1. Input Processing
- CSV format support with flexible text field specification
- Large corpus handling (51MB+ datasets)
- Error handling and logging
- Metadata preservation

### 2. Text Processing
- spaCy transformer model (`en_core_web_trf`)
- GPU acceleration support
- Advanced tokenization and normalization
- Context preservation for marker detection

### 3. Marker Detection
- Pattern-based matching with spaCy Matcher
- Fuzzy matching for multi-word expressions
- Context-aware filtering
- Polyfunctional marker resolution

### 4. Analysis and Output
- Statistical computation
- Visualization generation
- Multiple export formats (CSV, Markdown, LaTeX, PNG)
- Academic table formatting

## Key Algorithms and Techniques

### Enhanced Marker Detection Algorithm

```python
def is_marker_present(marker, text):
    # Single-word whole-word matching
    if len(marker.split()) == 1:
        pattern = r'\b' + re.escape(marker) + r'\b'
        return bool(re.search(pattern, text))
    
    # Multi-word flexible matching
    # Allows intervening words and punctuation
    # Maintains word order and proximity constraints
```

### Polyfunctional Marker Handling

- Markers counted in all applicable categories
- Examples: "in fact" → both code gloss AND booster
- Context-sensitive disambiguation
- Weighted scoring for overlapping functions

### Statistical Robustness

- Multiple normalization methods (per 1000 words, per 10,000 words)
- Distribution testing and appropriate statistical tests
- Effect size calculations (eta-squared)
- Multiple comparison corrections

## Research Applications

### Current Analysis Capabilities

1. **L1 Influence Analysis**: Comparing native language backgrounds
2. **Genre Analysis**: Academic writing patterns
3. **Developmental Analysis**: Learner progression
4. **Cross-cultural Rhetoric**: Cultural writing preferences
5. **Evidentiality Patterns**: Information source strategies

### Output Products

- **Academic Tables**: Publication-ready Markdown and LaTeX
- **Statistical Reports**: Comprehensive analysis summaries
- **Visualizations**: Research-quality plots and charts
- **Raw Data**: Full extraction with context for manual analysis

## Technical Specifications

### Dependencies
- **NLP**: spaCy 3.7.2+ with transformer models
- **Data**: pandas 2.0+, numpy 1.24+
- **Statistics**: statsmodels 0.13+, scipy
- **Visualization**: matplotlib 3.7+, seaborn 0.12+
- **ML**: torch 2.0+ (GPU acceleration), scikit-learn 1.0+

### Performance Optimizations
- GPU acceleration for transformer models
- Efficient pattern matching with spaCy
- Parallel processing capabilities
- Memory-efficient data handling for large corpora

### Extensibility Features
- Modular architecture for easy extension
- Configurable marker definitions
- Pluggable analysis modules
- Multiple output format support

## Research Impact and Applications

This framework enables:
- Large-scale corpus linguistic analysis
- Cross-linguistic comparative studies
- Automated metadiscourse analysis for writing assessment
- Pedagogical applications for academic writing instruction
- Computational rhetoric research

The codebase represents a sophisticated implementation of computational metadiscourse analysis with strong foundations in linguistic theory and practical applications in corpus linguistics research. 