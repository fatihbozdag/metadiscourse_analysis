"""
Basic Usage Example for Metalinguistics Library

This example demonstrates how to perform basic metadiscourse analysis
using the reorganized library structure.
"""

import sys
import os

# Add the src directory to the path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from metalinguistics import EnhancedMetadiscourseAnalyzer

def main():
    """Demonstrate basic metadiscourse analysis"""
    
    # Sample academic text
    text = """
    This study aims to demonstrate the effectiveness of the proposed methodology. 
    However, further research is needed to validate these findings. 
    In conclusion, the results clearly show significant improvement in performance.
    """
    
    print("=== Metalinguistics Basic Usage Example ===\n")
    print(f"Analyzing text: {text.strip()}\n")
    
    try:
        # Initialize the analyzer
        print("1. Initializing Enhanced Metadiscourse Analyzer...")
        analyzer = EnhancedMetadiscourseAnalyzer()
        
        # Perform analysis
        print("2. Performing metadiscourse analysis...")
        results = analyzer.analyze_text(text, use_ml=True, confidence_threshold=0.6)
        
        # Display results
        print("3. Analysis Results:")
        print(f"   Total markers found: {len(results['markers'])}")
        print(f"   Analysis method: {results['analysis_method']}")
        
        if results['markers']:
            print("\n   Detected markers:")
            for i, marker in enumerate(results['markers'], 1):
                print(f"   {i}. '{marker.text}' ({marker.category})")
                print(f"      - Confidence: {marker.confidence:.3f}")
                print(f"      - ML Prediction: {marker.ml_prediction}")
                print(f"      - Reason: {marker.validation_reason}")
                print()
        
        # Display summary
        summary = results['summary']
        print("4. Summary Statistics:")
        print(f"   - Total markers: {summary['total_markers']}")
        print(f"   - Average confidence: {summary['avg_confidence']:.3f}")
        print(f"   - ML predictions: {summary.get('ml_predictions', 0)}")
        
        if summary.get('categories'):
            print(f"   - Categories found: {list(summary['categories'].keys())}")
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("This might be due to missing model files or dependencies.")
        print("Please ensure the trained model is available in models/production/")

if __name__ == "__main__":
    main()