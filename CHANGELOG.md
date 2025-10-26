# Changelog

All notable changes to the Metalinguistics library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-10-27

### Added
- Initial public release of Metalinguistics library
- EnhancedMetadiscourseAnalyzer with hybrid rule-based + ML detection
- Support for eight metadiscourse categories based on Hyland (2005):
  - Transitions
  - Frame markers
  - Evidentials
  - Code glosses
  - Engagement markers
  - Self mentions
  - Boosters
  - Hedges
- Transformer-based NLP using spaCy's `en_core_web_trf` (RoBERTa)
- Random Forest classifier with 20+ linguistic features
- Context-aware pattern matching with ambiguity resolution
- Comprehensive validation framework with 86 test cases
- External JSON/YAML configuration system
- Batch corpus processing capabilities
- Confidence score calibration
- Duplicate detection and boundary analysis
- API for custom pattern definitions
- Training pipeline for custom models
- Performance metrics: 90.8% overall accuracy
  - True Positive Rate: 89.6%
  - False Positive Avoidance: 92.1%

### Documentation
- Complete API documentation with examples
- Installation and quick start guide
- Configuration guide for custom patterns
- Training guide for custom models
- Research methodology documentation

### Configuration
- Pattern definitions in `config/patterns/metadiscourse_patterns.json`
- Model configuration templates
- Analysis settings for different use cases

### Testing
- 86 manually annotated test cases
- Comprehensive test suite with pytest
- Pattern validation framework
- Edge case coverage for ambiguous contexts

### Examples
- Basic usage examples
- Batch corpus analysis examples
- Custom configuration examples

---

## Release Notes

### Version 1.0.0

This is the first public release of Metalinguistics, a production-ready library for metadiscourse analysis in academic texts. The library has been validated on multiple corpora and achieves state-of-the-art accuracy in distinguishing academic metadiscourse from conversational language patterns.

**Key Features:**
- Hybrid detection combining linguistic rules with machine learning
- Context-aware disambiguation to reduce false positives
- Highly configurable with external pattern files
- Suitable for corpus linguistics research and writing assessment applications

**Validated Performance:**
- Tested on learner and native speaker academic writing corpora
- 90.8% accuracy on independent validation set
- Robust handling of ambiguous markers (e.g., "however" as transition vs. temporal adverb)

**Research Applications:**
- Second language acquisition studies
- Corpus linguistics analysis
- Academic writing assessment
- Cross-linguistic discourse analysis
- Educational technology applications

For detailed usage instructions, see the [README](README.md) and [documentation](docs/).

**Citation:**
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

[Unreleased]: https://github.com/fatihbozdag/metadiscourse_analysis/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/fatihbozdag/metadiscourse_analysis/releases/tag/v1.0.0
