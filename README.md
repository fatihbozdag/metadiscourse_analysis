# 🔬 Metalinguistics: Advanced Metadiscourse Analysis Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A state-of-the-art NLP library for detecting and analyzing metadiscourse markers in academic texts using transformer models and machine learning.

## 🚀 Overview

**Metalinguistics** transforms metadiscourse analysis from simple pattern matching to sophisticated linguistic understanding. The library uses **BERT-based transformers**, **advanced feature engineering**, and **machine learning** to achieve research-grade accuracy in detecting metadiscourse markers across eight categories.

### Key Features

- 🧠 **Transformer-based NLP**: Uses `en_core_web_trf` (RoBERTa) for deep linguistic understanding
- 🎯 **ML Classification**: Random Forest classifier with 20+ linguistic features  
- 🔍 **Intelligent Boundaries**: Linguistic boundary detection instead of fixed rules
- ⚖️ **Smart Deduplication**: Multi-factor overlap resolution with confidence scoring
- 📊 **Post-processing Analysis**: Calibration and reporting without compromising detection accuracy
- ⚙️ **Configurable**: External JSON/YAML configuration for easy customization
- 🧪 **Tested**: Comprehensive test suite with 95% success rate

## 📦 Installation

### From Source (Development)
```bash
git clone https://github.com/yourusername/metalinguistics.git
cd metalinguistics
pip install -e .
```

### Requirements
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## 🎯 Quick Start

```python
from metalinguistics import EnhancedMetadiscourseAnalyzer

# Initialize analyzer
analyzer = EnhancedMetadiscourseAnalyzer()

# Analyze text
text = "This study demonstrates the effectiveness of the method. However, further research is needed."
results = analyzer.analyze_text(text)

# View results
print(f"Found {len(results['markers'])} metadiscourse markers")
for marker in results['markers']:
    print(f"'{marker.text}' ({marker.category}) - confidence: {marker.confidence:.3f}")
```

## 📊 Metadiscourse Categories

The library detects **8 metadiscourse categories** based on Hyland's framework:

| Category | Description | Examples |
|----------|-------------|----------|
| **Transitions** | Logical connectors | however, therefore, in contrast |
| **Frame Markers** | Discourse organizers | first, in conclusion, next section |
| **Evidentials** | Source references | according to, demonstrate, studies show |
| **Code Glosses** | Clarifications | namely, such as, in other words |
| **Engagement Markers** | Reader address | note that, consider, observe |
| **Self Mentions** | Academic self-reference | we argue, our study, the author |
| **Boosters** | Certainty markers | clearly, obviously, definitely |
| **Hedges** | Cautious phrasing | might, possibly, seem to |

## 🏗️ Architecture

```
metalinguistics/
├── src/metalinguistics/           # Main library code
│   ├── analyzers/                 # Analysis engines
│   ├── features/                  # Feature extraction
│   ├── ml/                        # ML components  
│   ├── processing/                # Text processing utilities
│   └── config/                    # Configuration management
├── config/                        # External configurations
├── data/                          # Datasets
├── models/                        # Trained ML models
├── scripts/                       # Training & utility scripts
├── tests/                         # Test suite
└── examples/                      # Usage examples
```

## 🔧 Advanced Usage

### Custom Configuration
```python
from metalinguistics.config import ConfigManager

config = ConfigManager()
config.update_category_keywords('transitions', ['however', 'moreover', 'furthermore'])

# Use custom confidence threshold
analyzer = EnhancedMetadiscourseAnalyzer()
results = analyzer.analyze_text(text, confidence_threshold=0.8)
```

### Batch Processing
```python
from metalinguistics.features import SpacyFeatureExtractor
import pandas as pd

# Extract features for multiple texts
extractor = SpacyFeatureExtractor()
df = pd.DataFrame({'text': texts, 'marker_text': markers})
features_df = extractor.extract_features_from_dataset(df)
```

### Training Custom Models
```python
from metalinguistics.ml import MetadiscourseClassifier

# Train on your own dataset
classifier = MetadiscourseClassifier(model_type='random_forest')
results = classifier.train(training_df)
classifier.save_model('my_custom_model.joblib')
```

## 📈 Performance

The system has been trained and tested on a 100K synthetic dataset:

- **Training Data**: 100,000 annotated examples
- **Model**: Random Forest with 20+ linguistic features
- **Feature Types**: Lexical, syntactic, contextual, academic
- **Test Suite**: 95% success rate across all components

### Technical Specifications
- **NLP Model**: `en_core_web_trf` (RoBERTa-based, 560MB)
- **Processing Speed**: ~1-2 seconds per 100 words (with MPS acceleration)
- **Memory Usage**: ~2GB RAM with transformer model loaded
- **Platform**: Optimized for macOS Apple Silicon, compatible with CPU/CUDA

## 🧪 Testing

Run the comprehensive test suite:

```bash
python tests/test_comprehensive.py
```

The test suite includes:
- Unit tests for individual components
- Integration tests for end-to-end workflow  
- Performance benchmarks
- Configuration validation

## 📚 Documentation

- **API Documentation**: `docs/api/`
- **Examples**: `examples/`
- **Research Notes**: `docs/research/`

### Module Documentation

Each module includes detailed README files:
- [Analyzers](src/metalinguistics/analyzers/README.md) - Main analysis engines
- [Features](src/metalinguistics/features/README.md) - Feature extraction components
- [ML](src/metalinguistics/ml/README.md) - Machine learning classifiers
- [Processing](src/metalinguistics/processing/README.md) - Text processing utilities

## 🛠️ Development

### Setup Development Environment
```bash
git clone https://github.com/yourusername/metalinguistics.git
cd metalinguistics
pip install -e .[dev]
```

### Code Quality
```bash
# Format code
black src/ tests/

# Run linting  
flake8 src/ tests/

# Type checking
mypy src/
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Research Applications

This library is suitable for:
- **Academic Writing Analysis**: Analyze discourse patterns in scholarly texts
- **L2 Writing Assessment**: Evaluate non-native speaker metadiscourse use
- **Corpus Linguistics**: Large-scale discourse marker studies
- **Educational Technology**: Automated writing feedback systems
- **Comparative Rhetoric**: Cross-cultural discourse analysis

## 📧 Support

- **Documentation**: Check the `docs/` directory
- **Issues**: [GitHub Issues](https://github.com/yourusername/metalinguistics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/metalinguistics/discussions)

## 🎓 Citation

If you use this library in your research, please cite:

```bibtex
@software{metalinguistics2025,
  title={Metalinguistics: Advanced Metadiscourse Analysis Library},
  author={Bozdag, Fatih},
  year={2025},
  url={https://github.com/yourusername/metalinguistics}
}
```

---

**Made with ❤️ for academic research and NLP innovation**