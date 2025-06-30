#!/usr/bin/env python3
"""
Metadiscourse Validation Starter Script
Using TED-MDB Human Annotations for System Validation
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json

class TEDMDBParser:
    """Parse TED-MDB PDTB-style annotations"""
    
    def __init__(self, ted_mdb_path: str):
        self.ted_mdb_path = Path(ted_mdb_path)
        self.annotations = {}
        
    def parse_annotation_file(self, file_path: str) -> List[Dict]:
        """Parse a single TED-MDB annotation file"""
        annotations = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('|')
                if len(parts) >= 10:
                    annotation = {
                        'relation_type': parts[0],  # Explicit, Implicit, etc.
                        'connective_span': parts[1],
                        'sense': parts[8],  # Discourse relation sense
                        'arg1_span': parts[14],
                        'arg2_span': parts[20],
                    }
                    annotations.append(annotation)
        
        return annotations
    
    def load_all_annotations(self) -> Dict:
        """Load all English annotations from TED-MDB"""
        english_path = self.ted_mdb_path / "English" / "ann" / "01"
        
        annotations = {}
        for file_path in english_path.glob("*.txt"):
            talk_id = file_path.stem
            annotations[talk_id] = self.parse_annotation_file(file_path)
            
        return annotations

class MetadiscourseValidator:
    """Validate our metadiscourse system against human annotations"""
    
    def __init__(self, ted_mdb_annotations: Dict, our_system_output: Dict):
        self.human_annotations = ted_mdb_annotations
        self.system_output = our_system_output
        
        # Map PDTB senses to our metadiscourse categories
        self.sense_mapping = {
            'Expansion.Conjunction': 'Interactive_Transitions',
            'Expansion.Level-of-detail': 'Interactive_Code_Glosses',
            'Expansion.Instantiation': 'Interactive_Code_Glosses',
            'Temporal.Synchronous': 'Interactive_Transitions',
            'Temporal.Asynchronous': 'Interactive_Transitions',
            'Contingency.Cause': 'Interactive_Transitions',
            'Contingency.Purpose': 'Interactive_Transitions',
            'Comparison.Concession': 'Interactive_Transitions',
            'Comparison.Contrast': 'Interactive_Transitions',
        }
    
    def map_pdtb_to_metadiscourse(self, pdtb_sense: str) -> str:
        """Map PDTB discourse sense to our metadiscourse category"""
        for pdtb_pattern, md_category in self.sense_mapping.items():
            if pdtb_sense.startswith(pdtb_pattern):
                return md_category
        return 'Unknown'
    
    def calculate_overlap_metrics(self, talk_id: str) -> Dict:
        """Calculate precision/recall for a single talk"""
        if talk_id not in self.human_annotations or talk_id not in self.system_output:
            return None
            
        human_spans = set()
        for annotation in self.human_annotations[talk_id]:
            # Extract character spans from human annotations
            if annotation['connective_span']:
                spans = self.parse_span(annotation['connective_span'])
                for span in spans:
                    human_spans.add(span)
        
        system_spans = set()
        if talk_id in self.system_output:
            for marker in self.system_output[talk_id]:
                span = (marker['start'], marker['end'])
                system_spans.add(span)
        
        # Calculate metrics
        true_positives = len(human_spans.intersection(system_spans))
        false_positives = len(system_spans - human_spans)
        false_negatives = len(human_spans - system_spans)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def parse_span(self, span_str: str) -> List[Tuple[int, int]]:
        """Parse character span string into (start, end) tuples"""
        spans = []
        if not span_str:
            return spans
            
        for span_part in span_str.split(';'):
            if '..' in span_part:
                start, end = span_part.split('..')
                spans.append((int(start), int(end)))
        
        return spans
    
    def evaluate_all_talks(self) -> Dict:
        """Evaluate system performance across all talks"""
        results = {}
        all_metrics = []
        
        for talk_id in self.human_annotations.keys():
            metrics = self.calculate_overlap_metrics(talk_id)
            if metrics:
                results[talk_id] = metrics
                all_metrics.append(metrics)
        
        # Calculate aggregate metrics
        if all_metrics:
            total_tp = sum(m['true_positives'] for m in all_metrics)
            total_fp = sum(m['false_positives'] for m in all_metrics)
            total_fn = sum(m['false_negatives'] for m in all_metrics)
            
            overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
            
            results['overall'] = {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'total_talks': len(all_metrics)
            }
        
        return results

def main():
    """Main validation workflow"""
    print("🔍 Starting Metadiscourse Validation with TED-MDB")
    
    # 1. Load TED-MDB annotations
    print("📚 Loading TED-MDB human annotations...")
    ted_mdb_path = "Ted-MDB-Annotations"
    
    if not os.path.exists(ted_mdb_path):
        print("❌ TED-MDB dataset not found. Please run:")
        print("   git clone https://github.com/MurathanKurfali/Ted-MDB-Annotations.git")
        return
    
    parser = TEDMDBParser(ted_mdb_path)
    human_annotations = parser.load_all_annotations()
    print(f"✅ Loaded annotations for {len(human_annotations)} talks")
    
    # 2. Load our system output (placeholder - replace with actual data)
    print("🤖 Loading our system output...")
    # TODO: Replace this with actual system output
    our_system_output = {}
    
    # 3. Run validation
    print("⚖️  Running validation comparison...")
    validator = MetadiscourseValidator(human_annotations, our_system_output)
    results = validator.evaluate_all_talks()
    
    # 4. Display results
    print("\n📊 VALIDATION RESULTS")
    print("=" * 50)
    
    if 'overall' in results:
        overall = results['overall']
        print(f"Overall Performance:")
        print(f"  Precision: {overall['precision']:.3f}")
        print(f"  Recall:    {overall['recall']:.3f}")
        print(f"  F1 Score:  {overall['f1']:.3f}")
        print(f"  Talks:     {overall['total_talks']}")
    else:
        print("No validation results - missing system output data")
    
    # 5. Save results
    output_file = "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {output_file}")
    
    print("\n🎯 Next Steps:")
    print("1. Integrate your actual system output")
    print("2. Run validation on TICLE sample data")
    print("3. Analyze error patterns for system improvement")
    print("4. Contact metaTED authors for additional validation data")

if __name__ == "__main__":
    main() 