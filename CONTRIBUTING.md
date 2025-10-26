# Contributing to Metalinguistics

Thank you for your interest in contributing to the Metalinguistics library! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/metalinguistics.git
   cd metalinguistics
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]
   python -m spacy download en_core_web_trf
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-category-X` for new features
- `fix/issue-123` for bug fixes
- `docs/improve-readme` for documentation

### 2. Make Your Changes

- Follow PEP 8 style guidelines
- Write clear, descriptive commit messages
- Add docstrings to all functions and classes
- Include type hints where appropriate

### 3. Test Your Changes

Run the test suite:
```bash
pytest tests/
```

Run specific tests:
```bash
pytest tests/test_comprehensive.py -v
```

Check code quality:
```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "Brief description of changes"
```

Commit message guidelines:
- Use present tense ("Add feature" not "Added feature")
- Keep first line under 50 characters
- Reference issues: "Fix #123: Description"

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Pull Request Guidelines

### Before Submitting

- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`black`)
- [ ] No linting errors (`flake8`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated

### PR Description Should Include

- Clear description of changes
- Motivation and context
- Related issue numbers (e.g., "Fixes #123")
- Screenshots (if UI changes)
- Testing methodology

## Code Style

### Python Style
- Follow PEP 8
- Use `black` for formatting (line length: 88)
- Use `flake8` for linting
- Use type hints for function signatures

### Documentation Style
- Use NumPy-style docstrings
- Include parameter types and return types
- Provide usage examples for public APIs

Example:
```python
def analyze_text(text: str, confidence_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Analyze text for metadiscourse markers.

    Parameters
    ----------
    text : str
        The text to analyze.
    confidence_threshold : float, optional
        Minimum confidence score for marker detection (default: 0.7).

    Returns
    -------
    Dict[str, Any]
        Analysis results containing detected markers and statistics.

    Examples
    --------
    >>> analyzer = EnhancedMetadiscourseAnalyzer()
    >>> results = analyzer.analyze_text("However, this shows...")
    >>> len(results['markers'])
    2
    """
```

## Testing Guidelines

### Writing Tests

- Use pytest for all tests
- Organize tests to mirror source structure
- Use descriptive test names: `test_detect_transitions_in_academic_text()`
- Include both positive and negative test cases
- Test edge cases and error conditions

### Test Coverage

Aim for high test coverage:
```bash
pytest --cov=src/metalinguistics --cov-report=html
```

## Adding New Features

### New Metadiscourse Categories

If adding a new category:
1. Update `config/patterns/metadiscourse_patterns.json`
2. Add test cases in `tests/test_comprehensive.py`
3. Update documentation
4. Retrain classifier if needed

### New Detection Patterns

1. Add patterns to appropriate category in JSON config
2. Include examples of true positives and false positives
3. Test against validation corpus
4. Document ambiguity rules

## Reporting Issues

### Bug Reports Should Include

- Python version and OS
- Metalinguistics version
- Minimal reproducible example
- Expected vs actual behavior
- Error messages and stack traces

### Feature Requests Should Include

- Clear use case description
- Proposed API or interface
- Example usage
- Potential implementation approach

## Code Review Process

1. Maintainers will review your PR
2. Address feedback and push updates
3. Once approved, your PR will be merged
4. Your contribution will be acknowledged in CHANGELOG.md

## Research Contributions

### Theoretical Contributions
If proposing changes based on metadiscourse theory:
- Cite relevant research
- Explain theoretical justification
- Provide corpus examples

### Corpus-Based Improvements
If suggesting pattern changes based on corpus analysis:
- Share methodology
- Provide frequency statistics
- Include inter-rater reliability metrics

## Community

- Be respectful and constructive
- Help others in discussions
- Share your use cases and applications

## Questions?

- Open an issue for questions
- Check existing issues first
- Use GitHub Discussions for general questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to Metalinguistics!
