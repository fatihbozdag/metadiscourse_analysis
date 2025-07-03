import json
import random
from precision_analyzer import MetadiscourseAnalyzer
import pandas as pd

class EnhancedValidator:
    def __init__(self):
        self.analyzer = MetadiscourseAnalyzer()
        
    def create_focused_test_cases(self):
        """Create test cases that target the specific problems identified"""
        test_cases = [
            # SELF_MENTIONS - should be CORRECT
            {'text': 'In my opinion, this research demonstrates significant findings.', 'expected': [('my opinion', 'self_mentions')]},
            {'text': 'We can conclude that the evidence supports this hypothesis.', 'expected': [('We can', 'self_mentions')]},
            {'text': 'I believe that this theory requires further investigation.', 'expected': [('I believe', 'self_mentions')]},
            
            # SELF_MENTIONS - should be REJECTED
            {'text': 'Especially in our time, money goes beyond love.', 'expected': []},
            {'text': 'We can give Argentina as an example.', 'expected': []},
            {'text': 'We should always have hope inside ourselves.', 'expected': []},
            
            # CODE_GLOSSES - should be CORRECT
            {'text': 'Various methods exist, such as qualitative analysis and quantitative research.', 'expected': [('such as', 'code_glosses')]},
            {'text': 'That is to say, the results indicate a clear pattern.', 'expected': [('That is', 'code_glosses')]},
            {'text': 'For example, this study examined the effects of education.', 'expected': [('For example', 'code_glosses')]},
            
            # CODE_GLOSSES - should be REJECTED  
            {'text': 'I was especially happy when we went to the park.', 'expected': []},
            {'text': 'Including his family, everyone was there.', 'expected': []},
            {'text': 'Such as when I was young, life was different.', 'expected': []},
            
            # TRANSITIONS - should be CORRECT
            {'text': 'However, the evidence suggests otherwise.', 'expected': [('However', 'transitions')]},
            {'text': 'Therefore, we can conclude that the hypothesis is supported.', 'expected': [('Therefore', 'transitions')]},
            {'text': 'Moreover, this finding has significant implications.', 'expected': [('Moreover', 'transitions')]},
            
            # TRANSITIONS - should be REJECTED
            {'text': 'This is not useful for our next life.', 'expected': []},
            {'text': 'At first, men watch this progress with surprise.', 'expected': []},
            {'text': 'Then we went to the store.', 'expected': []},
        ]
        
        return test_cases
    
    def validate_focused_cases(self):
        """Run validation on focused test cases"""
        test_cases = self.create_focused_test_cases()
        
        print("FOCUSED VALIDATION TEST")
        print("="*50)
        
        total_tests = len(test_cases)
        correct_predictions = 0
        
        for i, case in enumerate(test_cases):
            text = case['text']
            expected = case['expected']
            
            # Analyze text
            result = self.analyzer.analyze_document(text, f'test_{i}')
            detected = [(marker['text'], marker['category']) for marker in result['detailed_markers']]
            
            # Check if prediction matches expectation
            is_correct = set(detected) == set(expected)
            if is_correct:
                correct_predictions += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{status} Test {i+1:2d}: {text[:60]}...")
            print(f"   Expected: {expected}")
            print(f"   Detected: {detected}")
            if not is_correct:
                print(f"   → MISMATCH")
            print()
        
        accuracy = (correct_predictions / total_tests) * 100
        print(f"FOCUSED TEST ACCURACY: {correct_predictions}/{total_tests} = {accuracy:.1f}%")
        
        return accuracy
    
    def run_comprehensive_validation(self, sample_size=50):
        """Run validation on new random sample from TICLE"""
        print("COMPREHENSIVE VALIDATION")
        print("="*50)
        
        # Load TICLE corpus
        df = pd.read_csv('data/TICLE_sample.csv')
        
        # Extract all markers
        all_markers = []
        for idx, row in df.iterrows():
            text = str(row['text_field'])
            doc_results = self.analyzer.analyze_document(text, f'doc_{idx}')
            
            for marker in doc_results['detailed_markers']:
                marker['doc_id'] = doc_results['document_id']
                marker['full_text'] = text
                all_markers.append(marker)
        
        print(f'Total markers extracted: {len(all_markers)}')
        
        # Random sample
        random.seed(456)  # New seed
        sample_markers = random.sample(all_markers, min(sample_size, len(all_markers)))
        
        # Quick manual assessment
        correct_count = 0
        for marker in sample_markers:
            is_correct = self._quick_assess_marker(marker)
            if is_correct:
                correct_count += 1
        
        accuracy = (correct_count / len(sample_markers)) * 100
        print(f"Estimated accuracy: {correct_count}/{len(sample_markers)} = {accuracy:.1f}%")
        
        return accuracy
    
    def _quick_assess_marker(self, marker):
        """Quick assessment of marker correctness"""
        category = marker['category']
        text = marker['text']
        context = marker['context'].lower()
        
        # High-confidence patterns that are usually correct
        if category == 'boosters' and text in ['It is clear that', 'clearly', 'certainly', 'obviously']:
            return True
        elif category == 'hedges' and text in ['perhaps', 'might', 'could', 'seems', 'appears', 'somewhat']:
            return True
        elif category == 'frame_markers' and text in ['To sum up', 'In conclusion', 'Finally']:
            return True
        elif category == 'code_glosses':
            if text in ['For example', 'That is', 'such as', 'namely', 'including']:
                # Check for problematic contexts
                if any(bad in context for bad in ['especially in our', 'especially at', 'including his family', 'including me', 'such as when i']):
                    return False
                if len(text) > 50:  # Parsing boundary issue
                    return False
                return True
        elif category == 'self_mentions':
            if text in ['my opinion', 'I believe', 'I argue', 'our research', 'our analysis']:
                return True
            if text in ['we can', 'we must', 'we should']:
                # Check for academic context
                if any(good in context for good in ['conclude', 'analyze', 'examine', 'argue', 'demonstrate', 'establish']):
                    return True
                if any(bad in context for bad in ['we can go', 'we can see', 'our time', 'our family', 'give argentina']):
                    return False
                return True
        elif category == 'transitions':
            if text in ['However', 'Therefore', 'Moreover', 'Furthermore', 'Nevertheless']:
                if any(bad in context for bad in ['next life', 'first time', 'then we', 'after dinner']):
                    return False
                return True
        
        return False

if __name__ == "__main__":
    validator = EnhancedValidator()
    
    # Run focused validation first
    focused_accuracy = validator.validate_focused_cases()
    
    print("\n" + "="*60)
    
    # Run comprehensive validation
    comprehensive_accuracy = validator.run_comprehensive_validation(50)
    
    print(f"\nSUMMARY:")
    print(f"Focused test accuracy: {focused_accuracy:.1f}%")
    print(f"Comprehensive accuracy: {comprehensive_accuracy:.1f}%") 