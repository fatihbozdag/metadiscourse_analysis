"""
Comprehensive Testing & Evaluation Framework
Phase 3.8: Complete testing suite for the metadiscourse analysis system
"""

import unittest
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import tempfile
import time

# Import all our modules
from spacy_feature_extractor import SpacyFeatureExtractor
from ml_metadiscourse_classifier import MetadiscourseClassifier
from enhanced_metadiscourse_analyzer import EnhancedMetadiscourseAnalyzer
from intelligent_boundary_detector import IntelligentBoundaryDetector
from post_processing_calibrator import PostProcessingCalibrator
from enhanced_deduplicator import EnhancedDeduplicator
from config_manager import ConfigManager

class TestSpacyFeatureExtractor(unittest.TestCase):
    """Test cases for SpacyFeatureExtractor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = SpacyFeatureExtractor()
        self.test_text = "This study demonstrates the effectiveness of the proposed method. However, further research is needed."
        self.test_marker = "However"
        self.test_start = self.test_text.find(self.test_marker)
    
    def test_feature_extraction(self):
        """Test basic feature extraction"""
        features = self.extractor.extract_features(self.test_text, self.test_marker, self.test_start)
        
        self.assertEqual(features.marker_text, self.test_marker)
        self.assertGreater(features.marker_length, 0)
        self.assertIsInstance(features.confidence, float)
        self.assertIsInstance(features.pos_tag, str)
    
    def test_batch_processing(self):
        """Test batch processing of multiple samples"""
        test_data = pd.DataFrame({
            'text': [self.test_text, "Moreover, this approach is innovative."],
            'marker_text': ["However", "Moreover"]
        })
        
        result_df = self.extractor.extract_features_from_dataset(test_data)
        
        self.assertEqual(len(result_df), 2)
        self.assertTrue(any(col.startswith('feat_') for col in result_df.columns))
    
    def test_academic_context_detection(self):
        """Test academic context scoring"""
        academic_text = "This research demonstrates significant findings in the analysis."
        non_academic_text = "I went to the store to buy groceries."
        
        academic_features = self.extractor.extract_features(academic_text, "demonstrates", 13)
        non_academic_features = self.extractor.extract_features(non_academic_text, "went", 2)
        
        self.assertGreater(academic_features.academic_context_score, 
                          non_academic_features.academic_context_score)

class TestMLClassifier(unittest.TestCase):
    """Test cases for ML Classifier"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create minimal test dataset
        self.test_data = pd.DataFrame({
            'text': [
                "This study demonstrates the effectiveness.",
                "However, further research is needed.",
                "I went to the store yesterday.",
                "The results clearly show improvement."
            ],
            'marker_text': ["demonstrates", "However", "went", "clearly"],
            'marker_category': ["evidentials", "transitions", "transitions", "boosters"],
            'is_metadiscourse': [True, True, False, True]
        })
    
    def test_classifier_initialization(self):
        """Test classifier initialization"""
        classifier = MetadiscourseClassifier(model_type='random_forest')
        self.assertEqual(classifier.model_type, 'random_forest')
        self.assertIsNotNone(classifier.feature_extractor)
    
    def test_feature_preparation(self):
        """Test feature preparation"""
        classifier = MetadiscourseClassifier(model_type='logistic_regression')
        X, y = classifier.prepare_features(self.test_data)
        
        self.assertEqual(len(X), len(self.test_data))
        self.assertEqual(len(y), len(self.test_data))
        self.assertIsInstance(X, np.ndarray)
    
    def test_prediction_interface(self):
        """Test prediction interface (without full training)"""
        classifier = MetadiscourseClassifier(model_type='random_forest')
        
        # Test that prediction interface exists
        self.assertTrue(hasattr(classifier, 'predict'))
        self.assertTrue(hasattr(classifier, 'train'))

class TestBoundaryDetector(unittest.TestCase):
    """Test cases for Intelligent Boundary Detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = IntelligentBoundaryDetector()
        self.test_text = "In conclusion, the results show significant improvement. However, more research is needed."
        self.potential_markers = ["In conclusion", "However", "show"]
    
    def test_boundary_detection(self):
        """Test boundary detection"""
        boundaries = self.detector.detect_boundaries(self.test_text, self.potential_markers)
        
        self.assertGreater(len(boundaries), 0)
        for boundary in boundaries:
            self.assertIsInstance(boundary.confidence, float)
            self.assertGreater(boundary.confidence, 0)
            self.assertLessEqual(boundary.confidence, 1)
    
    def test_overlap_classification(self):
        """Test overlap type classification"""
        # Test with overlapping markers
        overlapping_markers = ["conclusion", "In conclusion"]
        boundaries = self.detector.detect_boundaries(self.test_text, overlapping_markers)
        
        # Should handle overlaps appropriately
        self.assertIsInstance(boundaries, list)
    
    def test_confidence_scoring(self):
        """Test confidence scoring mechanism"""
        boundaries = self.detector.detect_boundaries(self.test_text, ["However"])
        
        if boundaries:
            boundary = boundaries[0]
            self.assertGreater(boundary.confidence, 0)
            self.assertIsInstance(boundary.linguistic_justification, str)

class TestDeduplicator(unittest.TestCase):
    """Test cases for Enhanced Deduplicator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.deduplicator = EnhancedDeduplicator()
        
        # Mock marker class
        class MockMarker:
            def __init__(self, text, category, start_pos, end_pos, confidence):
                self.text = text
                self.category = category
                self.start_pos = start_pos
                self.end_pos = end_pos
                self.confidence = confidence
                self.ml_prediction = True
                self.context = "academic context"
                self.linguistic_features = {'feat_academic_context_score': confidence * 0.8}
        
        # Create overlapping markers
        self.test_markers = [
            MockMarker("however", "transitions", 0, 7, 0.7),
            MockMarker("However,", "transitions", 0, 8, 0.8),  # Overlaps
            MockMarker("conclusion", "frame_markers", 50, 60, 0.6),
            MockMarker("clearly", "boosters", 100, 107, 0.9)
        ]
    
    def test_deduplication(self):
        """Test deduplication process"""
        deduplicated, log = self.deduplicator.deduplicate_markers(self.test_markers)
        
        # Should remove some overlapping markers
        self.assertLessEqual(len(deduplicated), len(self.test_markers))
        self.assertIsInstance(log, list)
    
    def test_confidence_preservation(self):
        """Test that deduplication preserves higher confidence markers"""
        deduplicated, log = self.deduplicator.deduplicate_markers(self.test_markers)
        
        if len(deduplicated) < len(self.test_markers):
            # Average confidence should improve or stay the same
            orig_conf = np.mean([m.confidence for m in self.test_markers])
            dedup_conf = np.mean([m.confidence for m in deduplicated])
            self.assertGreaterEqual(dedup_conf, orig_conf - 0.1)  # Allow small tolerance

class TestCalibrator(unittest.TestCase):
    """Test cases for Post-Processing Calibrator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calibrator = PostProcessingCalibrator()
        
        # Mock marker class
        class MockMarker:
            def __init__(self, category, confidence):
                self.category = category
                self.confidence = confidence
                self.ml_prediction = True
                self.linguistic_features = {'feat_academic_context_score': confidence * 0.8}
        
        self.test_markers = [
            MockMarker('transitions', 0.9),
            MockMarker('frame_markers', 0.8),
            MockMarker('evidentials', 0.7)
        ]
        
        self.test_text = "This is a test text with fifty words to evaluate the calibration " \
                        "system and its ability to analyze marker density and distribution " \
                        "patterns in academic texts for testing purposes and validation."
    
    def test_marker_analysis(self):
        """Test marker analysis functionality"""
        stats = self.calibrator.analyze_markers(self.test_markers, self.test_text)
        
        self.assertEqual(stats.total_count, len(self.test_markers))
        self.assertGreater(stats.density_per_100_words, 0)
        self.assertIsInstance(stats.category_distribution, dict)
    
    def test_calibration_report(self):
        """Test calibration report generation"""
        report = self.calibrator.calibrate_for_purpose(self.test_markers, self.test_text, 'balanced')
        
        self.assertIsInstance(report.original_markers, int)
        self.assertIsInstance(report.calibrated_markers, int)
        self.assertIsInstance(report.recommendations, list)

class TestConfigManager(unittest.TestCase):
    """Test cases for Configuration Manager"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Use temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(config_dir=self.temp_dir)
    
    def test_config_loading(self):
        """Test configuration loading"""
        # Should create default config if none exists
        patterns = self.config_manager.load_config('patterns')
        self.assertIsInstance(patterns, dict)
    
    def test_category_access(self):
        """Test category configuration access"""
        categories = self.config_manager.get_all_categories()
        self.assertIsInstance(categories, list)
        
        if categories:
            keywords = self.config_manager.get_keywords_for_category(categories[0])
            self.assertIsInstance(keywords, list)
    
    def test_config_validation(self):
        """Test configuration validation"""
        validation = self.config_manager.validate_config('patterns')
        self.assertIsInstance(validation, dict)
        self.assertIn('valid', validation)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.test_text = """
        This study aims to demonstrate the effectiveness of the proposed methodology. 
        However, further research is needed to validate these findings. 
        In conclusion, the results clearly show significant improvement.
        I went to the store to buy groceries yesterday.
        """
    
    def test_end_to_end_analysis(self):
        """Test complete analysis pipeline"""
        try:
            # Initialize analyzer (may fail if model not available)
            analyzer = EnhancedMetadiscourseAnalyzer()
            
            # Run analysis
            results = analyzer.analyze_text(self.test_text, use_ml=False)  # Use rule-based fallback
            
            self.assertIsInstance(results, dict)
            self.assertIn('markers', results)
            self.assertIn('summary', results)
            
        except FileNotFoundError:
            # Skip if ML model not available
            self.skipTest("ML model not available for integration test")
    
    def test_config_integration(self):
        """Test configuration integration"""
        config_manager = ConfigManager()
        
        # Test that configurations are accessible
        patterns = config_manager.load_config('patterns')
        self.assertIn('categories', patterns)
        
        model_params = config_manager.get_model_parameters()
        self.assertIsInstance(model_params, dict)

class PerformanceTests(unittest.TestCase):
    """Performance and benchmark tests"""
    
    def test_feature_extraction_performance(self):
        """Test feature extraction performance"""
        extractor = SpacyFeatureExtractor()
        
        # Test with multiple samples
        test_data = pd.DataFrame({
            'text': ["Sample text for testing."] * 100,
            'marker_text': ["testing"] * 100
        })
        
        start_time = time.time()
        result_df = extractor.extract_features_from_dataset(test_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process reasonably quickly
        self.assertLess(processing_time, 60)  # Less than 60 seconds for 100 samples
        self.assertEqual(len(result_df), 100)
    
    def test_boundary_detection_performance(self):
        """Test boundary detection performance"""
        detector = IntelligentBoundaryDetector()
        
        long_text = "This is a test sentence. " * 100  # Long text
        markers = ["This", "test", "sentence"]
        
        start_time = time.time()
        boundaries = detector.detect_boundaries(long_text, markers)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process reasonably quickly
        self.assertLess(processing_time, 10)  # Less than 10 seconds
        self.assertIsInstance(boundaries, list)

def run_comprehensive_tests():
    """Run all test suites and generate report"""
    
    print("=" * 60)
    print("COMPREHENSIVE METADISCOURSE ANALYSIS SYSTEM TESTS")
    print("=" * 60)
    
    # Test suites to run
    test_suites = [
        TestSpacyFeatureExtractor,
        TestMLClassifier,
        TestBoundaryDetector,
        TestDeduplicator,
        TestCalibrator,
        TestConfigManager,
        TestIntegration,
        PerformanceTests
    ]
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for test_suite in test_suites:
        print(f"\nRunning {test_suite.__name__}...")
        print("-" * 40)
        
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(test_suite)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Update counters
        total_tests += result.testsRun
        total_passed += result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
        total_failed += len(result.failures) + len(result.errors)
        total_skipped += len(result.skipped)
        
        # Print results for this suite
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped)}")
    
    # Print overall summary
    print("\n" + "=" * 60)
    print("OVERALL TEST RESULTS")
    print("=" * 60)
    print(f"Total tests run: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Skipped: {total_skipped}")
    print(f"Success rate: {(total_passed/total_tests)*100:.1f}%" if total_tests > 0 else "N/A")
    
    # Generate test report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_tests': total_tests,
        'passed': total_passed,
        'failed': total_failed,
        'skipped': total_skipped,
        'success_rate': (total_passed/total_tests)*100 if total_tests > 0 else 0
    }
    
    with open('test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nTest report saved to 'test_report.json'")
    
    return total_failed == 0

if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)