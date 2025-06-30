# Metadiscourse Validation Implementation Roadmap

## Conceptual Overview

The validation approach transforms our system from assumption-based to evidence-based by:
1. **Comparing against human annotations** instead of arbitrary benchmarks
2. **Setting empirical thresholds** based on actual performance data
3. **Identifying systematic errors** through detailed analysis
4. **Optimizing category-specific rules** based on evidence

## Phase 1: Dataset Acquisition (Week 1-2)

### Immediate Actions
```bash
# 1. Clone TED-MDB repository
git clone https://github.com/MurathanKurfali/Ted-MDB-Annotations.git
cd Ted-MDB-Annotations

# 2. Explore dataset structure
ls -la
head English/*.txt  # Check annotation format
```

### Contact Strategy
- **metaTED Authors**: Email Rui Correia, Nuno Mamede (INESC-ID, Portugal)
- **OCWMD Authors**: Email Ghada Alharbi, Thomas Hain (University of Sheffield)
- **Request**: Access to annotated datasets for academic research validation

## Phase 2: Data Integration (Week 3-4)

### Category Mapping
Create mapping between dataset categories and our system:

```python
# category_mapping.py
CATEGORY_MAPPINGS = {
    'metaTED': {
        'DEF': 'Interactive_Code_Glosses',      # Definitions
        'EMPH': 'Interactional_Boosters',      # Emphasizing  
        'INTRO': 'Interactive_Transitions',    # Introducing
        'CONC': 'Interactive_Transitions',     # Concluding
        'EXMPL': 'Interactive_Code_Glosses',   # Examples
        # ... map all 16 categories
    },
    'TED_MDB': {
        # Map PDTB-style annotations to our categories
        'Expansion': 'Interactive_Transitions',
        'Comparison': 'Interactive_Transitions',
        # ... PDTB mappings
    }
}
```

### Evaluation Pipeline
```python
# validation_pipeline.py
class MetadiscourseValidator:
    def __init__(self, human_dataset, system_output):
        self.human_data = self.load_human_annotations(human_dataset)
        self.system_data = self.load_system_output(system_output)
        
    def align_annotations(self):
        """Align human and system annotations by text position"""
        # Match annotations to same text spans
        pass
    
    def calculate_metrics(self):
        """Calculate precision, recall, F1 for each category"""
        metrics = {}
        for category in self.categories:
            tp, fp, fn = self.count_matches(category)
            metrics[category] = {
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'f1': self.f1_score(tp, fp, fn)
            }
        return metrics
```

## Phase 3: Baseline Evaluation (Week 5-6)

### Performance Analysis
```python
# Run current system on human-annotated data
def evaluate_current_system():
    # 1. Load human annotations
    human_annotations = load_human_dataset('metaTED')
    
    # 2. Run our system on same texts
    system_results = run_metadiscourse_analysis(human_annotations.texts)
    
    # 3. Compare results
    validator = MetadiscourseValidator(human_annotations, system_results)
    metrics = validator.calculate_metrics()
    
    # 4. Analyze errors
    error_patterns = validator.analyze_errors()
    
    return metrics, error_patterns
```

### Expected Findings
- **Over-detection categories**: Likely self-mentions, engagement markers
- **Under-detection categories**: Complex transitions, hedges
- **Context patterns**: Where false positives occur most

## Phase 4: System Refinement (Week 7-8)

### Empirical Threshold Setting
```python
def optimize_thresholds(validation_data):
    """Find optimal confidence thresholds using ROC analysis"""
    optimal_thresholds = {}
    
    for category in categories:
        # Get confidence scores and ground truth
        scores = get_confidence_scores(category, validation_data)
        labels = get_ground_truth_labels(category, validation_data)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Find optimal threshold (maximize F1)
        optimal_idx = find_optimal_f1_threshold(labels, scores, thresholds)
        optimal_thresholds[category] = thresholds[optimal_idx]
    
    return optimal_thresholds
```

### Context-Sensitive Rules
```python
def add_context_filters(error_analysis):
    """Add filters based on systematic error patterns"""
    filters = {}
    
    # Example: Filter pronouns in non-metadiscourse contexts
    if error_analysis['self_mentions']['false_positive_contexts']:
        filters['self_mentions'] = {
            'avoid_contexts': ['narrative_sequences', 'quoted_speech'],
            'require_contexts': ['argument_structure', 'topic_introduction']
        }
    
    return filters
```

## Phase 5: Validation Testing (Week 9-10)

### Cross-Validation Setup
```python
def cross_validate_system(annotated_data, k_folds=5):
    """Perform k-fold cross-validation"""
    results = []
    
    for train_idx, test_idx in kfold_split(annotated_data, k_folds):
        # Train thresholds on training set
        train_data = annotated_data[train_idx]
        optimal_thresholds = optimize_thresholds(train_data)
        
        # Test on validation set
        test_data = annotated_data[test_idx]
        performance = evaluate_with_thresholds(test_data, optimal_thresholds)
        results.append(performance)
    
    return aggregate_results(results)
```

### Domain Transfer Testing
```python
def test_domain_transfer():
    """Test system across different domains"""
    domains = {
        'academic_lectures': 'OCWMD_corpus',
        'presentations': 'metaTED_corpus', 
        'student_essays': 'TICLE_sample'
    }
    
    results = {}
    for source_domain, target_domain in domain_pairs:
        # Train on source domain
        model = train_on_domain(domains[source_domain])
        
        # Test on target domain
        performance = test_on_domain(model, domains[target_domain])
        results[f"{source_domain}_to_{target_domain}"] = performance
    
    return results
```

## Expected Outcomes

### Quantified Performance
- **Precision/Recall by category**: Replace "high accuracy" with specific numbers
- **Confidence intervals**: Statistical significance of improvements
- **Benchmark comparison**: How we compare to published systems

### Evidence-Based System
- **Validated thresholds**: Replace arbitrary 0.8/0.85 with empirical values
- **Context rules**: Based on actual error patterns, not assumptions
- **Category-specific approaches**: Different strategies per metadiscourse type

### Documentation
- **Error taxonomy**: Systematic understanding of failure modes
- **Performance limits**: Known boundaries of system capability
- **Domain applicability**: Where system works well vs. poorly

## Success Metrics

1. **Reduced over-detection**: Bring marker density to validated range
2. **Improved precision**: Higher accuracy on positive detections
3. **Maintained recall**: Don't lose true positive detections
4. **Empirical validation**: All decisions backed by human annotation data

This roadmap transforms the fundamental approach from "filtering based on assumptions" to "optimizing based on evidence" - addressing your core concern about the reliability of our current methodology. 