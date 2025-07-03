# Analyzers Module

This module contains the main analysis engines for metadiscourse detection and analysis.

## Purpose

The analyzers module provides high-level interfaces for performing metadiscourse analysis on academic texts. It integrates ML classification, feature extraction, and post-processing components into user-friendly analysis engines.

## Structure

```
analyzers/
├── __init__.py                    # Module exports
├── enhanced_analyzer.py           # Main ML-integrated analyzer
└── README.md                      # This documentation
```

## Usage

### Basic Analysis
```python
from metalinguistics.analyzers import EnhancedMetadiscourseAnalyzer

analyzer = EnhancedMetadiscourseAnalyzer()
results = analyzer.analyze_text("Your academic text here...")
```

### Advanced Configuration
```python
# Use custom model path
analyzer = EnhancedMetadiscourseAnalyzer("path/to/custom/model.joblib")

# Configure analysis parameters
results = analyzer.analyze_text(
    text,
    use_ml=True,
    confidence_threshold=0.7
)
```

## Components

### EnhancedMetadiscourseAnalyzer

The main analysis engine that combines:
- **ML Classification**: Trained Random Forest classifier
- **Rule-based Fallback**: Pattern matching when ML unavailable
- **Confidence Scoring**: Per-marker confidence assessment
- **Category Detection**: 8 metadiscourse categories
- **Export Capabilities**: JSON/CSV result export

#### Key Methods

- `analyze_text(text, use_ml=True, confidence_threshold=0.6)`: Analyze single text
- `export_results(results, format='json')`: Export analysis results

#### Output Format

Returns analysis results with:
```python
{
    'text': str,                    # Original text
    'markers': List[EnhancedMarker], # Detected markers
    'summary': Dict[str, Any],      # Analysis statistics
    'analysis_method': str          # Method used
}
```

### EnhancedMarker

Dataclass representing detected metadiscourse markers:

```python
@dataclass
class EnhancedMarker:
    text: str                      # Marker text
    category: str                  # Metadiscourse category
    start_pos: int                 # Character start position
    end_pos: int                   # Character end position
    context: str                   # Surrounding context
    confidence: float              # Detection confidence
    ml_prediction: bool            # Whether ML was used
    ml_confidence: float           # ML prediction confidence
    linguistic_features: Dict      # Extracted features
    validation_reason: str         # Why marker was accepted
```

## Dependencies

- **ML Module**: `MetadiscourseClassifier` for machine learning
- **Processing**: Deduplication and boundary detection
- **Config**: Configuration management

## Examples

### Batch Processing
```python
analyzer = EnhancedMetadiscourseAnalyzer()

texts = ["Text 1...", "Text 2...", "Text 3..."]
all_results = []

for text in texts:
    results = analyzer.analyze_text(text)
    all_results.append(results)
```

### Export Results
```python
results = analyzer.analyze_text(text)

# Export as JSON
json_output = analyzer.export_results(results, 'json')
with open('analysis.json', 'w') as f:
    f.write(json_output)

# Export as CSV
csv_output = analyzer.export_results(results, 'csv')
with open('analysis.csv', 'w') as f:
    f.write(csv_output)
```

## Contributing

When adding new analyzers:

1. **Inherit common patterns** from existing analyzers
2. **Use consistent interfaces** for analysis methods
3. **Include comprehensive docstrings** and type hints
4. **Add unit tests** in `tests/test_analyzers.py`
5. **Update module exports** in `__init__.py`