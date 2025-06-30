#!/usr/bin/env python3
"""
Test script to validate enhanced metalinguistics system achieves >90% accuracy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')

from processor import EnhancedTextProcessor
from markers import EnhancedMetadiscourseMarkers
from main import EnhancedMetalinguisticsAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_corpus():
    """Create a test corpus with known markers for accuracy validation."""
    
    test_texts = [
        # Text 1: Heavy interactive markers
        """First, we need to examine the evidence. However, the results are not conclusive. 
        According to Smith (2020), the findings suggest that further research is needed. 
        In other words, we cannot make definitive claims at this point. For example, 
        the data shows inconsistent patterns. Therefore, we must be cautious in our interpretation.""",
        
        # Text 2: Heavy interactional markers  
        """I argue that this approach is clearly superior. You can see that the methodology 
        is obviously more robust. We believe that our findings will certainly contribute 
        to the field. Unfortunately, previous studies have failed to address this issue. 
        Perhaps future research might explore alternative approaches.""",
        
        # Text 3: Mixed markers with polyfunctional cases
        """The study demonstrates significant improvements. In fact, the results are remarkable. 
        Indeed, we can confidently state that the hypothesis is supported. Actually, 
        the data reveals patterns that were previously unknown. Moreover, these findings 
        have important implications for practice.""",
        
        # Text 4: Academic writing with evidentials
        """Recent research indicates that climate change affects biodiversity (Jones et al., 2023). 
        Studies show that temperature increases correlate with species migration patterns. 
        The literature suggests that adaptation strategies are crucial. Previous work has 
        established the theoretical framework for understanding these phenomena.""",
        
        # Text 5: Engagement and self-mention heavy
        """Let us consider the implications of these findings. You should note that the 
        methodology differs from previous approaches. We propose a new framework that 
        addresses current limitations. I suggest that future researchers examine this 
        relationship more closely. Our analysis reveals important patterns."""
    ]
    
    # Create DataFrame with metadata
    test_data = []
    for i, text in enumerate(test_texts):
        test_data.append({
            'id': f'test_{i+1}',
            'text': text,
            'l1_language': ['English', 'Turkish', 'Chinese', 'Spanish', 'German'][i],
            'proficiency_level': ['Advanced', 'Intermediate', 'Advanced', 'Intermediate', 'Advanced'][i],
            'genre': 'academic'
        })
    
    return pd.DataFrame(test_data)

def create_ground_truth():
    """Create ground truth annotations for accuracy calculation."""
    
    # Ground truth marker counts for each test text
    ground_truth = {
        'test_1': {
            'interactive_transitions': 3,  # First, However, Therefore
            'interactive_evidentials': 1,  # According to Smith (2020)
            'interactive_code_glosses': 2,  # In other words, For example
            'total_interactive': 6,
            'interactional_hedges': 2,  # suggest, must be cautious
            'total_interactional': 2,
            'total_markers': 8
        },
        'test_2': {
            'interactional_self_mentions': 3,  # I argue, We believe, our
            'interactional_boosters': 3,  # clearly, obviously, certainly
            'interactional_engagement_markers': 2,  # You can see, your
            'interactional_attitude_markers': 1,  # Unfortunately
            'interactional_hedges': 1,  # Perhaps, might
            'total_interactive': 0,
            'total_interactional': 10,
            'total_markers': 10
        },
        'test_3': {
            'interactional_boosters': 6,  # demonstrates, In fact, Indeed, confidently, Actually, Moreover
            'interactive_code_glosses': 3,  # In fact, Indeed, Actually (polyfunctional)
            'total_interactive': 3,
            'total_interactional': 6,
            'total_markers': 9
        },
        'test_4': {
            'interactive_evidentials': 4,  # Recent research indicates, Studies show, literature suggests, Previous work
            'total_interactive': 4,
            'total_interactional': 0,
            'total_markers': 4
        },
        'test_5': {
            'interactional_engagement_markers': 3,  # Let us, You should, Our
            'interactional_self_mentions': 3,  # We propose, I suggest, Our analysis
            'total_interactive': 0,
            'total_interactional': 6,
            'total_markers': 6
        }
    }
    
    return ground_truth

def calculate_accuracy(predicted, ground_truth):
    """Calculate accuracy metrics comparing predicted vs ground truth."""
    
    accuracies = []
    detailed_results = {}
    
    for text_id in ground_truth.keys():
        if text_id not in predicted:
            continue
            
        pred = predicted[text_id]
        truth = ground_truth[text_id]
        
        # Calculate accuracy for total markers
        pred_total = pred.get('total_markers', 0)
        truth_total = truth.get('total_markers', 0)
        
        # Use relative accuracy to handle slight variations
        if truth_total > 0:
            total_accuracy = 1 - abs(pred_total - truth_total) / truth_total
            total_accuracy = max(0, total_accuracy)  # Ensure non-negative
        else:
            total_accuracy = 1.0 if pred_total == 0 else 0.0
        
        accuracies.append(total_accuracy)
        
        detailed_results[text_id] = {
            'predicted_total': pred_total,
            'ground_truth_total': truth_total,
            'accuracy': total_accuracy,
            'predicted_interactive': pred.get('total_interactive', 0),
            'ground_truth_interactive': truth.get('total_interactive', 0),
            'predicted_interactional': pred.get('total_interactional', 0),
            'ground_truth_interactional': truth.get('total_interactional', 0)
        }
    
    overall_accuracy = np.mean(accuracies) if accuracies else 0.0
    
    return overall_accuracy, detailed_results

def test_enhanced_system():
    """Test the enhanced system for >90% accuracy."""
    
    print("="*60)
    print("TESTING ENHANCED METALINGUISTICS SYSTEM")
    print("="*60)
    
    # Create test data
    print("📋 Creating test corpus...")
    test_df = create_test_corpus()
    ground_truth = create_ground_truth()
    
    print(f"✅ Created test corpus with {len(test_df)} documents")
    
    # Initialize enhanced analyzer
    print("🚀 Initializing Enhanced Analyzer...")
    try:
        analyzer = EnhancedMetalinguisticsAnalyzer(
            use_enhanced=True,
            model_name="en_core_web_sm",  # Use smaller model for testing
            use_gpu=False  # Disable GPU for testing
        )
        print("✅ Enhanced Analyzer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Enhanced Analyzer: {e}")
        return False
    
    # Process test corpus
    print("🔄 Processing test corpus...")
    results = []
    predicted_results = {}
    
    for idx, row in test_df.iterrows():
        try:
            result = analyzer.processor.process_text_enhanced(row['text'], row['id'])
            results.append(result)
            
            # Extract key metrics for accuracy calculation
            stats = result.get('statistics', {})
            predicted_results[row['id']] = {
                'total_markers': stats.get('total_markers', 0),
                'total_interactive': stats.get('total_interactive', 0),
                'total_interactional': stats.get('total_interactional', 0)
            }
            
            print(f"✅ Processed {row['id']}: {stats.get('total_markers', 0)} markers detected")
            
        except Exception as e:
            print(f"❌ Error processing {row['id']}: {e}")
            predicted_results[row['id']] = {'total_markers': 0, 'total_interactive': 0, 'total_interactional': 0}
    
    # Calculate accuracy
    print("\n📊 Calculating accuracy...")
    overall_accuracy, detailed_results = calculate_accuracy(predicted_results, ground_truth)
    
    # Display results
    print("\n" + "="*60)
    print("ACCURACY RESULTS")
    print("="*60)
    
    print(f"🎯 Overall Accuracy: {overall_accuracy:.2%}")
    
    if overall_accuracy >= 0.90:
        print("✅ SUCCESS: Achieved >90% accuracy target!")
        success = True
    else:
        print("❌ BELOW TARGET: Did not achieve 90% accuracy")
        success = False
    
    print("\n📋 Detailed Results:")
    print("-" * 60)
    for text_id, result in detailed_results.items():
        print(f"{text_id}:")
        print(f"  Predicted: {result['predicted_total']} | Ground Truth: {result['ground_truth_total']}")
        print(f"  Accuracy: {result['accuracy']:.2%}")
        print(f"  Interactive: {result['predicted_interactive']} (GT: {result['ground_truth_interactive']})")
        print(f"  Interactional: {result['predicted_interactional']} (GT: {result['ground_truth_interactional']})")
        print()
    
    # Test specific enhancements
    print("🔧 Testing Enhancement Features:")
    print("-" * 60)
    
    # Test polyfunctional resolution
    poly_test_text = "In fact, the results are remarkable. Indeed, we can see clear improvements."
    poly_result = analyzer.processor.process_text_enhanced(poly_test_text, "poly_test")
    poly_resolved = poly_result.get('processing_info', {}).get('polyfunctional_resolved', False)
    print(f"✅ Polyfunctional Resolution: {'Working' if poly_resolved else 'Not Detected'}")
    
    # Test confidence scoring
    confidence_scores = analyzer.processor.stats.get('confidence_scores', [])
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
    print(f"✅ Average Confidence Score: {avg_confidence:.3f}")
    
    # Test context awareness
    context_corrections = analyzer.processor.stats.get('context_filtered', 0)
    print(f"✅ Context-based Corrections: {context_corrections}")
    
    print("\n" + "="*60)
    print("ENHANCEMENT IMPACT ANALYSIS")
    print("="*60)
    
    enhancement_impact = analyzer._calculate_enhancement_impact()
    print(f"🔧 Enhancement Features Used: {enhancement_impact.get('enhancement_used', False)}")
    print(f"📊 Polyfunctional Resolution Rate: {enhancement_impact.get('polyfunctional_resolution_rate', 0):.2%}")
    print(f"🎯 Average Confidence Score: {enhancement_impact.get('average_confidence_score', 0):.3f}")
    print(f"⭐ High Confidence Markers: {enhancement_impact.get('high_confidence_markers', 0):.2%}")
    print(f"🚀 Model Used: {enhancement_impact.get('model_used', 'unknown')}")
    
    return success

def run_comparative_test():
    """Run comparative test between standard and enhanced systems."""
    
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS: STANDARD vs ENHANCED")
    print("="*60)
    
    test_df = create_test_corpus()
    
    # Test standard system
    print("🔄 Testing Standard System...")
    try:
        from processor import TextProcessor
        standard_processor = TextProcessor()
        standard_results = []
        
        for idx, row in test_df.iterrows():
            result = standard_processor.process_text(row['text'], row['id'])
            standard_results.append(result.get('total_markers', 0))
        
        standard_avg = np.mean(standard_results)
        print(f"📊 Standard System - Average Markers: {standard_avg:.2f}")
        
    except Exception as e:
        print(f"❌ Standard system test failed: {e}")
        standard_avg = 0
    
    # Test enhanced system
    print("🔄 Testing Enhanced System...")
    try:
        enhanced_analyzer = EnhancedMetalinguisticsAnalyzer(use_enhanced=True, model_name="en_core_web_sm", use_gpu=False)
        enhanced_results = []
        
        for idx, row in test_df.iterrows():
            result = enhanced_analyzer.processor.process_text_enhanced(row['text'], row['id'])
            enhanced_results.append(result.get('statistics', {}).get('total_markers', 0))
        
        enhanced_avg = np.mean(enhanced_results)
        print(f"📊 Enhanced System - Average Markers: {enhanced_avg:.2f}")
        
        # Calculate improvement
        if standard_avg > 0:
            improvement = ((enhanced_avg - standard_avg) / standard_avg) * 100
            print(f"📈 Improvement: {improvement:+.1f}%")
        
    except Exception as e:
        print(f"❌ Enhanced system test failed: {e}")

if __name__ == "__main__":
    print("🧪 Starting Enhanced Metalinguistics System Accuracy Test")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run main accuracy test
    success = test_enhanced_system()
    
    # Run comparative test
    run_comparative_test()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if success:
        print("🎉 OVERALL RESULT: SUCCESS")
        print("✅ Enhanced system achieved >90% accuracy target")
        print("🚀 System is ready for production use")
        exit_code = 0
    else:
        print("⚠️  OVERALL RESULT: NEEDS IMPROVEMENT")
        print("❌ Enhanced system did not meet 90% accuracy target")
        print("🔧 Further optimization recommended")
        exit_code = 1
    
    print(f"⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    exit(exit_code) 