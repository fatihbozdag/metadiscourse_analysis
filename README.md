# Metadiscourse Analysis Tool

A comprehensive Python-based tool for analyzing metadiscourse markers in academic texts based on Hyland's (2005) framework. This tool provides a complete workflow for detecting, analyzing, and visualizing metadiscourse patterns in text corpora.

## Features

- **Comprehensive Marker Detection**: Identifies all interactive and interactional metadiscourse markers
- **Context-Aware Filtering**: Uses linguistic rules to improve detection accuracy
- **Statistical Analysis**: ANOVA and post-hoc tests for language group differences
- **Visualization Suite**: Scatter plots, bar charts, heatmaps, and more
- **Shannon Entropy Calculation**: Measures metadiscourse diversity
- **LaTeX Table Generation**: Creates publication-ready tables for academic papers

## Usage

The main script for this tool is `metadiscourse_analysis.py`:

```bash
python metadiscourse_analysis.py --input your_data.csv --output_dir results
```

Or for analyzing a corpus of text files:

```bash
python metadiscourse_analysis.py --corpus /path/to/corpus --language_map language_mapping.txt --output_dir results
```

### Jupyter Notebook

A Jupyter notebook version is also available as `metadiscourse_analysis_notebook.ipynb` for interactive analysis.

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- spaCy (with en_core_web_trf or en_core_web_sm model)
- torch (optional, for GPU acceleration)
- jupyter and nbconvert (for notebook conversion)

## Author

Fatih Ünal Bozdağ - fatihbozdag@osmaniye.edu.tr

## License

MIT License
