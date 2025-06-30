# Metalinguistics Project - Executive Summary

## What This Codebase Does

This is a **comprehensive computational linguistics framework** for analyzing metadiscourse markers in academic texts. It implements Ken Hyland's influential metadiscourse taxonomy with advanced NLP techniques to study how writers interact with their readers and position their arguments.

## Key Capabilities

### 🔍 **Advanced Marker Detection**
- Identifies 10 categories of metadiscourse markers across 100+ specific markers
- Handles polyfunctional markers (words serving multiple rhetorical functions)
- Uses sophisticated pattern matching with context awareness
- Processes large corpora (50MB+ datasets) efficiently

### 📊 **Statistical Analysis**
- Calculates normalized frequencies (per 1000 words)
- Performs ANOVA, correlation, and regression analyses
- Generates publication-ready statistical reports
- Handles multiple comparison corrections

### 🎯 **Specialized Research Applications**
- **L1 Influence Studies**: Compare native vs. learner language patterns
- **Cross-cultural Rhetoric**: Analyze cultural differences in academic writing
- **Evidentiality Analysis**: Study how writers signal information sources
- **Genre Analysis**: Examine disciplinary writing conventions

### 📈 **Research-Quality Outputs**
- Academic tables in Markdown and LaTeX formats
- Statistical visualizations (heatmaps, distributions, correlations)
- Raw extraction data with full context
- Error logging and quality assurance

## Quick Start

### Basic Analysis
```bash
# Analyze a corpus
python src/main.py --input_dir data --output_dir results

# Generate L1-based analysis
python l1_tables_and_plots.py
```

### Evidentiality Analysis
```bash
cd evidentiality_project
python src/main.py --ticle_path data/ticle.csv --locness_path data/locness.csv
```

## Core Technologies

- **NLP Engine**: spaCy with transformer models (`en_core_web_trf`)
- **Statistical Computing**: pandas, numpy, statsmodels
- **Visualization**: matplotlib, seaborn
- **Performance**: GPU acceleration, efficient pattern matching

## Research Impact

This framework has been designed for:
- **Corpus Linguistics Research**: Large-scale text analysis
- **Second Language Writing Studies**: Learner language analysis
- **Computational Rhetoric**: Automated discourse analysis
- **Writing Assessment**: Metadiscourse profiling tools

## Data Requirements

- **Input Format**: CSV files with text column
- **Corpus Size**: Handles datasets from small samples to 50MB+
- **Languages**: Optimized for English academic writing
- **Metadata**: Preserves all original metadata for analysis

## Project Structure at a Glance

```
📁 Main Analysis Framework (src/)
   ├── processor.py    # Core NLP engine
   ├── markers.py      # Metadiscourse taxonomy
   ├── viz.py          # Visualization suite
   └── stats.py        # Statistical analysis

📁 Specialized Modules
   ├── analyze_metadiscourse.py     # Standalone analysis
   ├── l1_tables_and_plots.py      # L1 comparison tools
   └── evidentiality_project/       # Evidentiality analysis

📁 Data & Results
   ├── data/           # Input corpora
   └── results/        # Analysis outputs
```

## Scientific Foundation

Based on established linguistic frameworks:
- **Hyland (2005)**: Metadiscourse taxonomy
- **Aikhenvald (2004)**: Evidentiality theory
- **Fauconnier (1994)**: Mental spaces theory

The codebase bridges theoretical linguistics with computational methods, enabling scalable analysis of rhetorical patterns in academic discourse.

---

**Ready for Research**: This codebase is production-ready for linguistic research, with robust error handling, comprehensive documentation, and validated output formats for academic publication. 