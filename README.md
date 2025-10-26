# 🔬 Metalinguistics: Advanced Metadiscourse Analysis Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A state-of-the-art Python library for detecting and analyzing metadiscourse markers in academic texts using transformer models and machine learning.

## 🚀 Overview

**Metalinguistics** provides sophisticated tools for metadiscourse analysis in academic writing. It combines **transformer-based NLP** (RoBERTa), **machine learning classification**, and **linguistic feature engineering** to achieve high accuracy in detecting and categorizing metadiscourse markers.

### Key Features

- 🧠 **Transformer-based NLP**: Uses `en_core_web_trf` (RoBERTa) for deep linguistic understanding
- 🎯 **ML Classification**: Random Forest classifier with 20+ linguistic features
- 📊 **Eight Category Detection**: Comprehensive coverage of Hyland's metadiscourse framework
- 🔍 **Context-Aware**: Distinguishes academic from conversational usage patterns
- ⚙️ **Highly Configurable**: External JSON/YAML configuration for easy customization
- 🧪 **Well-Tested**: Comprehensive validation with 90.8% accuracy
- 📦 **Easy Integration**: Simple API for batch processing and corpus analysis

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install from Source

```bash
git clone https://github.com/fatihbozdag/metadiscourse_analysis.git
cd metadiscourse_analysis
pip install -e .
```

### Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

> **Note**: The transformer model (`en_core_web_trf`) is ~560MB. For lighter installations, see [Alternative Models](#alternative-models).

## 🎯 Quick Start

```python
from metalinguistics.analyzers import EnhancedMetadiscourseAnalyzer

# Initialize analyzer
analyzer = EnhancedMetadiscourseAnalyzer()

# Analyze academic text
text = """
This study demonstrates the effectiveness of the proposed methodology.
However, further research is needed to validate these findings.
Our results clearly show significant improvement in performance.
"""

results = analyzer.analyze_text(text)

# Display detected markers
print(f"Found {len(results['markers'])} metadiscourse markers")
for marker in results['markers']:
    print(f"  '{marker.text}' ({marker.category}) - confidence: {marker.confidence:.2f}")
```

**Output:**
```
Found 5 metadiscourse markers
  'demonstrates' (evidentials) - confidence: 0.85
  'However' (transitions) - confidence: 0.92
  'Our' (self_mentions) - confidence: 0.78
  'clearly' (boosters) - confidence: 0.88
  'show' (evidentials) - confidence: 0.83
```

## 📊 Metadiscourse Categories

The library detects **eight metadiscourse categories** based on Hyland's (2005) framework:

| Category | Function | Examples |
|----------|----------|----------|
| **Transitions** | Logical connectors | however, therefore, moreover |
| **Frame Markers** | Discourse organizers | first, in conclusion, finally |
| **Evidentials** | Source references | according to, demonstrate, show |
| **Code Glosses** | Clarifications | namely, such as, for example |
| **Engagement Markers** | Reader address | note that, consider, observe |
| **Self Mentions** | Academic self-reference | we argue, our study |
| **Boosters** | Certainty markers | clearly, obviously, definitely |
| **Hedges** | Cautious phrasing | might, possibly, seem |

## 🏗️ Project Structure

```
metalinguistics/
├── src/metalinguistics/        # Main library code
│   ├── analyzers/              # Analysis engines
│   │   └── enhanced_analyzer.py
│   ├── features/               # Feature extraction
│   │   └── spacy_extractor.py
│   ├── ml/                     # ML classifiers
│   │   └── classifier.py
│   ├── processing/             # Post-processing
│   │   ├── deduplicator.py
│   │   ├── boundary_detector.py
│   │   └── calibrator.py
│   └── config/                 # Configuration management
│       └── manager.py
├── config/                     # External configurations
│   ├── patterns/               # Detection patterns
│   ├── models/                 # Model configs
│   └── analysis/               # Analysis settings
├── models/                     # Trained models (download separately)
│   ├── production/
│   └── experimental/
├── scripts/                    # Utility scripts
│   ├── train_model.py
│   └── analyze_corpus.py
├── tests/                      # Test suite
│   └── test_comprehensive.py
└── examples/                   # Usage examples
    └── basic_usage.py
```

## 🔧 Advanced Usage

### Custom Configuration

```python
from metalinguistics.config import ConfigManager

# Load custom patterns
config = ConfigManager()
config.load_patterns('config/patterns/metadiscourse_patterns.json')

# Adjust confidence threshold
analyzer = EnhancedMetadiscourseAnalyzer()
results = analyzer.analyze_text(text, confidence_threshold=0.8)
```

### Batch Corpus Analysis

```python
import pandas as pd
from metalinguistics.analyzers import EnhancedMetadiscourseAnalyzer

# Load corpus
corpus_df = pd.read_csv('my_corpus.csv')  # columns: doc_id, text

# Analyze all documents
analyzer = EnhancedMetadiscourseAnalyzer()
results = []

for idx, row in corpus_df.iterrows():
    analysis = analyzer.analyze_text(row['text'])
    for marker in analysis['markers']:
        results.append({
            'doc_id': row['doc_id'],
            'marker': marker.text,
            'category': marker.category,
            'confidence': marker.confidence
        })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('analysis_results.csv', index=False)
```

### Training Custom Models

```python
from metalinguistics.ml import MetadiscourseClassifier
import pandas as pd

# Prepare training data
training_df = pd.DataFrame({
    'text': [...],           # Full sentences
    'marker_text': [...],    # Candidate markers
    'is_metadiscourse': [...],  # True/False labels
    'category': [...]        # Category labels
})

# Train classifier
classifier = MetadiscourseClassifier(model_type='random_forest')
results = classifier.train(training_df, test_size=0.2)

# Save model
classifier.save_model('models/my_custom_model.joblib')
```

## 📈 Performance

### Validation Results
- **True Positive Rate**: 89.6%
- **False Positive Avoidance**: 92.1%
- **Overall Accuracy**: 90.8%
- **Test Cases**: 86 manually annotated samples

### Technical Specifications
- **NLP Model**: `en_core_web_trf` (RoBERTa-based, 560MB)
- **ML Classifier**: Random Forest with 100 estimators
- **Features**: 20+ linguistic features (lexical, syntactic, contextual)
- **Processing Speed**: ~1-2 seconds per 100 words
- **Memory**: ~2GB RAM with transformer model loaded

### Alternative Models

For lighter installations, you can use smaller spaCy models:

```bash
python -m spacy download en_core_web_sm  # 12MB
python -m spacy download en_core_web_md  # 40MB
```

Note: Accuracy may decrease with smaller models.

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_comprehensive.py

# Run with coverage
python -m pytest --cov=src/metalinguistics tests/
```

## 📚 Documentation

- **API Reference**: See module docstrings and `docs/api/`
- **Configuration Guide**: `config/README.md`
- **Examples**: `examples/`

## 🛠️ Development

### Setup Development Environment

```bash
git clone https://github.com/yourusername/metalinguistics.git
cd metalinguistics
pip install -e .[dev]
pre-commit install
```

### Code Quality Tools

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure all tests pass (`pytest`)
5. Format code (`black src/ tests/`)
6. Submit a Pull Request

## 📋 Requirements

Core dependencies:
- `spacy >= 3.0.0`
- `scikit-learn >= 1.0.0`
- `pandas >= 1.3.0`
- `numpy >= 1.20.0`
- `pyyaml >= 6.0`

See `requirements.txt` for complete list.

## 🏆 Research Applications

This library is designed for:
- **Corpus Linguistics**: Large-scale metadiscourse pattern analysis
- **L2 Writing Research**: Second language acquisition studies
- **Writing Assessment**: Automated evaluation of academic writing
- **Educational Technology**: Intelligent tutoring systems
- **Comparative Rhetoric**: Cross-cultural discourse analysis

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Citation

If you use this library in your research, please cite:

```bibtex
@software{metalinguistics2025,
  title={Metalinguistics: Advanced Metadiscourse Analysis Library},
  author={Bozdag, Fatih},
  year={2025},
  url={https://github.com/fatihbozdag/metadiscourse_analysis},
  note={Python library for metadiscourse analysis in academic texts}
}
```

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/fatihbozdag/metadiscourse_analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fatihbozdag/metadiscourse_analysis/discussions)

## 🙏 Acknowledgments

This library implements metadiscourse detection based on:
- Hyland, K. (2005). *Metadiscourse: Exploring Interaction in Writing*. Continuum.
- Transformer models from [spaCy](https://spacy.io/)
- Machine learning with [scikit-learn](https://scikit-learn.org/)

---

**Built for academic research and NLP applications**
