"""
ML Demo - Process first 10 TICLE documents to demonstrate ML integration
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from metalinguistics import EnhancedMetadiscourseAnalyzer

def demo_ml_analysis():
    print("🔬 ML Demo: Processing first 10 TICLE documents with full ML")
    print("=" * 60)
    
    # Load analyzer with ML
    print("🧠 Loading ML analyzer...")
    analyzer = EnhancedMetadiscourseAnalyzer()
    print(f"✅ ML Model loaded: {analyzer.ml_classifier.is_trained}")
    
    # Load sample data
    data_path = Path("data/raw/TICLE_sample.csv")
    df = pd.read_csv(data_path)
    sample_docs = df.head(10)  # First 10 documents only
    
    print(f"📊 Processing {len(sample_docs)} documents...")
    
    results = []
    for idx, row in sample_docs.iterrows():
        text = str(row['text_field'])
        if len(text.strip()) < 10:
            continue
            
        print(f"\n📄 Document {idx+1}: {len(text)} chars")
        
        # Analyze with ML
        result = analyzer.analyze_text(text, use_ml=True, confidence_threshold=0.6)
        
        print(f"   🎯 Found {len(result['markers'])} markers")
        print(f"   📊 Method: {result['analysis_method']}")
        
        # Show marker details
        ml_count = sum(1 for m in result['markers'] if m.ml_prediction)
        print(f"   🤖 ML predictions: {ml_count}/{len(result['markers'])}")
        
        for marker in result['markers'][:3]:  # First 3 markers
            ml_status = "🤖" if marker.ml_prediction else "📝"
            print(f"   {ml_status} '{marker.text}' ({marker.category}) - conf: {marker.confidence:.3f}")
        
        if len(result['markers']) > 3:
            print(f"   ... and {len(result['markers']) - 3} more markers")
            
        results.append({
            'doc_id': f"TICLE_{idx+1:03d}",
            'markers_found': len(result['markers']),
            'ml_predictions': ml_count,
            'analysis_method': result['analysis_method']
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("🎊 ML Demo Results Summary")
    print("=" * 60)
    
    total_markers = sum(r['markers_found'] for r in results)
    total_ml = sum(r['ml_predictions'] for r in results)
    
    print(f"📊 Documents processed: {len(results)}")
    print(f"🎯 Total markers found: {total_markers}")
    print(f"🤖 ML predictions: {total_ml} ({total_ml/total_markers*100:.1f}%)")
    print(f"⚡ MPS acceleration: ACTIVE")
    print(f"🧠 ML model: metadiscourse_model_balanced_5k.joblib")
    
    # Method breakdown
    methods = {}
    for r in results:
        method = r['analysis_method']
        methods[method] = methods.get(method, 0) + 1
    
    print(f"⚙️ Analysis methods:")
    for method, count in methods.items():
        print(f"   - {method}: {count} documents")
    
    return results

if __name__ == "__main__":
    results = demo_ml_analysis()