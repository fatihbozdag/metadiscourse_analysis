#!/usr/bin/env python3
"""
Pattern Testing Framework for Metadiscourse Analyzer
Automated validation against manually verified test cases
"""

import json
import re
from typing import Dict, List, Tuple
from precision_analyzer import MetadiscourseAnalyzer

class PatternTestFramework:
    """Test framework for validating metadiscourse pattern accuracy"""
    
    def __init__(self):
        self.analyzer = MetadiscourseAnalyzer()
        self.test_cases = self._load_test_cases()
        
    def _load_test_cases(self) -> Dict:
        """Load test cases with known true/false positives"""
        return {
            'boosters': {
                'true_positives': [
                    "It is clear that this research demonstrates significant findings.",
                    "This certainly proves the hypothesis is correct.",
                    "Obviously, this argument supports our position.",
                    "The evidence clearly indicates strong correlation.",
                    "There is no doubt that this method is effective."
                ],
                'false_positives': [
                    "I had a very good day yesterday.",
                    "She is very beautiful and smart.",
                    "The weather is really nice today.",
                    "Most people like chocolate very much.",
                    "All students are extremely busy with exams.",
                    "It's obviously true that he likes her.",
                    "Clearly, we need to go shopping."
                ]
            },
            'hedges': {
                'true_positives': [
                    "The results seem to indicate a correlation.",
                    "This might suggest alternative interpretations.",
                    "It appears that further research is needed.",
                    "Perhaps this finding warrants investigation.",
                    "The data may imply different conclusions.",
                    "I believe that this theory holds merit.",
                    "To some extent, this supports our hypothesis."
                ],
                'false_positives': [
                    "I think about my family every day.",
                    "We should talk about this problem.",
                    "Maybe tomorrow we can go shopping.",
                    "She might come to the party tonight.",
                    "I would like to visit Paris someday.",
                    "May God bless you and your family.",
                    "It seems very difficult to understand."
                ]
            },
            'self_mentions': {
                'true_positives': [
                    "I argue that this approach is effective.",
                    "We believe that further study is needed.",
                    "I propose that alternative methods be considered.", 
                    "We suggest that these findings indicate correlation.",
                    "Our research demonstrates significant results.",
                    "My analysis reveals important patterns.",
                    "I conclude that this theory is valid."
                ],
                'false_positives': [
                    "I went to school yesterday.",
                    "We can go to the movies tonight.",
                    "My family lives in Turkey.",
                    "I think people should be kind.",
                    "We believe life is beautiful.",
                    "Our society needs more education.",
                    "I can speak three languages fluently."
                ]
            },
            'engagement_markers': {
                'true_positives': [
                    "Note that this method requires careful consideration.",
                    "Consider how this affects the overall results.",
                    "You should note that these findings are preliminary.",
                    "It is important to observe that correlation exists.",
                    "Let us examine this phenomenon more closely.",
                    "You might ask why this pattern emerges."
                ],
                'false_positives': [
                    "You are very smart and talented.",
                    "Consider that you might be wrong about this.",
                    "If you think about it carefully enough.",
                    "You should go to bed early tonight.",
                    "What do you want to eat for dinner?",
                    "Imagine you are walking in the park."
                ]
            },
            'evidentials': {
                'true_positives': [
                    "According to Smith (2020), this method is effective.",
                    "Research shows that correlation exists between variables.",
                    "As demonstrated by previous studies, this theory holds.",
                    "Evidence indicates strong support for this hypothesis.",
                    "According to the data, significant patterns emerge."
                ],
                'false_positives': [
                    "According to my mother, I should study harder.",
                    "According to the weather forecast, it will rain."
                ]
            },
            'transitions': {
                'true_positives': [
                    "However, this finding contradicts previous research.",
                    "Furthermore, additional evidence supports this claim.",
                    "In conclusion, the results are significant.",
                    "On the other hand, alternative explanations exist.",
                    "Therefore, we recommend further investigation.",
                    "In contrast, previous studies found different results."
                ],
                'false_positives': [
                    "However much I try, I can't understand.",
                    "Then we went to the shopping mall.",
                    "After we had dinner, we watched a movie.",
                    "While we were walking, it started raining."
                ]
            },
            'frame_markers': {
                'true_positives': [
                    "First, we examined the methodology.",
                    "In conclusion, the results support our hypothesis.",
                    "The purpose of this study is to investigate.",
                    "This paper aims to analyze the relationship.",
                    "Finally, we recommend further research.",
                    "To summarize, three main findings emerged."
                ],
                'false_positives': [
                    "First time I saw her, she was beautiful.",
                    "Finally we arrived at our destination.",
                    "Second hand cars are usually cheaper."
                ]
            },
            'code_glosses': {
                'true_positives': [
                    "Several factors were considered, namely cost and efficiency.",
                    "This includes important elements such as methodology.",
                    "Multiple approaches exist, for example qualitative methods.",
                    "Various techniques apply, i.e., statistical analysis.",
                    "That is, the correlation between variables is strong.",
                    "The results show significance, specifically in two areas."
                ],
                'false_positives': [
                    "For example, my brother likes football very much.",
                    "Such as when we go shopping together."
                ]
            }
        }
    
    def test_category_patterns(self, category: str) -> Dict:
        """Test patterns for a specific category"""
        if category not in self.test_cases:
            return {'error': f'No test cases for category: {category}'}
        
        test_data = self.test_cases[category]
        results = {
            'category': category,
            'true_positive_accuracy': 0.0,
            'false_positive_avoidance': 0.0,
            'detailed_results': {
                'true_positives': {'correct': 0, 'missed': 0, 'examples': []},
                'false_positives': {'correctly_avoided': 0, 'incorrectly_detected': 0, 'examples': []}
            }
        }
        
        # Test true positives (should be detected)
        tp_correct = 0
        for text in test_data['true_positives']:
            markers = self.analyzer._extract_markers(text, category)
            if markers:
                tp_correct += 1
                results['detailed_results']['true_positives']['examples'].append({
                    'text': text,
                    'detected': True,
                    'markers': [m['text'] for m in markers]
                })
            else:
                results['detailed_results']['true_positives']['examples'].append({
                    'text': text,
                    'detected': False,
                    'markers': []
                })
        
        # Test false positives (should NOT be detected)
        fp_avoided = 0
        for text in test_data['false_positives']:
            markers = self.analyzer._extract_markers(text, category)
            if not markers:
                fp_avoided += 1
                results['detailed_results']['false_positives']['examples'].append({
                    'text': text,
                    'incorrectly_detected': False,
                    'markers': []
                })
            else:
                results['detailed_results']['false_positives']['examples'].append({
                    'text': text,
                    'incorrectly_detected': True,
                    'markers': [m['text'] for m in markers]
                })
        
        # Calculate accuracy metrics
        total_tp = len(test_data['true_positives'])
        total_fp = len(test_data['false_positives'])
        
        results['true_positive_accuracy'] = (tp_correct / total_tp * 100) if total_tp > 0 else 0
        results['false_positive_avoidance'] = (fp_avoided / total_fp * 100) if total_fp > 0 else 0
        results['detailed_results']['true_positives']['correct'] = tp_correct
        results['detailed_results']['true_positives']['missed'] = total_tp - tp_correct
        results['detailed_results']['false_positives']['correctly_avoided'] = fp_avoided
        results['detailed_results']['false_positives']['incorrectly_detected'] = total_fp - fp_avoided
        
        return results
    
    def run_comprehensive_test(self) -> Dict:
        """Run tests on all categories"""
        all_results = {
            'timestamp': json.dumps(None, default=str),
            'overall_summary': {},
            'category_results': {}
        }
        
        total_tp_correct = 0
        total_tp_count = 0
        total_fp_avoided = 0
        total_fp_count = 0
        
        print("🧪 RUNNING COMPREHENSIVE PATTERN TESTS")
        print("=" * 50)
        
        for category in self.test_cases.keys():
            print(f"\n📝 Testing {category.upper()} patterns...")
            results = self.test_category_patterns(category)
            all_results['category_results'][category] = results
            
            # Accumulate totals
            tp_details = results['detailed_results']['true_positives']
            fp_details = results['detailed_results']['false_positives']
            
            total_tp_correct += tp_details['correct']
            total_tp_count += tp_details['correct'] + tp_details['missed']
            total_fp_avoided += fp_details['correctly_avoided']
            total_fp_count += fp_details['correctly_avoided'] + fp_details['incorrectly_detected']
            
            # Print category summary
            print(f"   ✅ True Positive Detection: {results['true_positive_accuracy']:.1f}%")
            print(f"   🚫 False Positive Avoidance: {results['false_positive_avoidance']:.1f}%")
            
            # Show problematic cases
            if tp_details['missed'] > 0:
                print(f"   ⚠️  Missed {tp_details['missed']} true positives")
            if fp_details['incorrectly_detected'] > 0:
                print(f"   ❌ {fp_details['incorrectly_detected']} false positives detected")
        
        # Calculate overall metrics
        overall_tp_accuracy = (total_tp_correct / total_tp_count * 100) if total_tp_count > 0 else 0
        overall_fp_avoidance = (total_fp_avoided / total_fp_count * 100) if total_fp_count > 0 else 0
        combined_score = (overall_tp_accuracy + overall_fp_avoidance) / 2
        
        all_results['overall_summary'] = {
            'true_positive_accuracy': overall_tp_accuracy,
            'false_positive_avoidance': overall_fp_avoidance,
            'combined_score': combined_score,
            'total_tests': total_tp_count + total_fp_count,
            'grade': self._calculate_grade(combined_score)
        }
        
        print(f"\n🎯 OVERALL RESULTS")
        print("=" * 50)
        print(f"True Positive Detection: {overall_tp_accuracy:.1f}%")
        print(f"False Positive Avoidance: {overall_fp_avoidance:.1f}%")
        print(f"Combined Score: {combined_score:.1f}%")
        print(f"Grade: {all_results['overall_summary']['grade']}")
        
        return all_results
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade based on combined score"""
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Good)" 
        elif score >= 70:
            return "C (Satisfactory)"
        elif score >= 60:
            return "D (Needs Improvement)"
        else:
            return "F (Poor)"
    
    def generate_test_report(self, results: Dict, output_file: str = None) -> str:
        """Generate detailed test report"""
        if not output_file:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"pattern_test_report_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return output_file
    
    def identify_problem_patterns(self, results: Dict) -> List[str]:
        """Identify patterns that need improvement"""
        problems = []
        
        for category, cat_results in results['category_results'].items():
            if cat_results['true_positive_accuracy'] < 70:
                problems.append(f"{category}: Low true positive detection ({cat_results['true_positive_accuracy']:.1f}%)")
            if cat_results['false_positive_avoidance'] < 80:
                problems.append(f"{category}: High false positive rate ({100-cat_results['false_positive_avoidance']:.1f}%)")
        
        return problems

def main():
    """Run pattern testing framework"""
    framework = PatternTestFramework()
    results = framework.run_comprehensive_test()
    
    # Save detailed report
    report_file = framework.generate_test_report(results)
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Identify problem areas
    problems = framework.identify_problem_patterns(results)
    if problems:
        print(f"\n⚠️  AREAS NEEDING ATTENTION:")
        for problem in problems:
            print(f"   • {problem}")
    else:
        print(f"\n✅ All patterns performing well!")

if __name__ == "__main__":
    main() 