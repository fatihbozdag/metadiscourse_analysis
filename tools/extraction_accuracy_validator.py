#!/usr/bin/env python3
"""
Extraction Accuracy Validator for Learner Corpus
Focuses on validating the quality and correctness of detected metadiscourse markers
"""

import json
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import random
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractionAccuracyValidator:
    """
    Validates the accuracy of metadiscourse extractions from learner corpus
    Focus: Are our detections actually correct metadiscourse markers?
    """
    
    def __init__(self):
        """Initialize validator with quality assessment criteria"""
        
        # Known high-quality metadiscourse patterns for validation
        self.gold_standard_patterns = {
            'hedges': {
                'definite': ['might', 'may', 'could', 'would', 'seem', 'appear', 'suggest', 'tend', 'likely', 'probably', 'possibly', 'perhaps', 'maybe'],
                'contextual': ['somewhat', 'rather', 'quite', 'fairly', 'relatively', 'approximately']
            },
            'boosters': {
                'definite': ['certainly', 'definitely', 'clearly', 'obviously', 'undoubtedly', 'indeed', 'surely', 'absolutely'],
                'contextual': ['very', 'extremely', 'highly', 'strongly', 'significantly']
            },
            'self_mentions': {
                'definite': ['i argue', 'i believe', 'i claim', 'i suggest', 'we argue', 'we believe', 'our study', 'our research'],
                'avoid': ['i went', 'i saw', 'i like', 'we went', 'we were']
            },
            'transitions': {
                'definite': ['however', 'therefore', 'furthermore', 'moreover', 'consequently', 'nevertheless'],
                'contextual': ['first', 'second', 'finally', 'in conclusion']
            },
            'engagement_markers': {
                'definite': ['consider that', 'note that', 'observe that', 'you can see', 'one can see'],
                'avoid': ['you are', 'you have', 'you went']
            },
            'evidentials': {
                'definite': ['according to', 'research shows', 'studies indicate', 'evidence suggests'],
                'contextual': ['as shown by', 'as noted by']
            },
            'code_glosses': {
                'definite': ['that is', 'namely', 'in other words', 'for example', 'for instance', 'such as'],
                'contextual': ['i.e.', 'e.g.']
            },
            'frame_markers': {
                'definite': ['first', 'second', 'finally', 'in conclusion', 'to summarize'],
                'contextual': ['the first section', 'the next part', 'this paper']
            }
        }
        
        # Quality indicators for context validation
        self.academic_context_indicators = [
            r'\b(?:research|study|analysis|paper|article|thesis|dissertation|investigation|examination|exploration)\b',
            r'\b(?:argue|claim|suggest|propose|conclude|demonstrate|show|indicate|reveal|find)\b',
            r'\b(?:data|evidence|findings|results|conclusion|hypothesis|theory|methodology)\b',
            r'\b(?:academic|scholarly|scientific|empirical|theoretical|analytical)\b'
        ]
        
        # Red flags for non-academic content
        self.non_academic_indicators = [
            r'\b(?:yesterday|today|tomorrow|weekend|vacation|holiday|birthday|party)\b',
            r'\b(?:family|friends|mother|father|sister|brother|girlfriend|boyfriend)\b',
            r'\b(?:movie|film|music|song|game|sport|football|shopping|cooking)\b',
            r'\b(?:went|came|saw|met|ate|drank|bought|sold|played|watched)\b'
        ]
    
    def load_latest_results(self) -> Dict:
        """Load the most recent analysis results"""
        import os
        
        results_files = [
            "final_optimized_results.json",
            "enhanced_analysis_results.json",
            "calibrated_analysis_results.json",
            "analysis_results.json"
        ]
        
        if os.path.exists("results/"):
            results_dir_files = [f"results/{f}" for f in os.listdir("results/") 
                               if f.endswith('.json')]
            results_files.extend(results_dir_files)
        
        # Find the most recent file
        latest_file = None
        latest_time = 0
        
        for file_path in results_files:
            if os.path.exists(file_path):
                mod_time = os.path.getmtime(file_path)
                if mod_time > latest_time:
                    latest_time = mod_time
                    latest_file = file_path
        
        if not latest_file:
            raise FileNotFoundError("No analysis results files found")
        
        logger.info(f"Loading results from: {latest_file}")
        
        # Sample large files for validation
        file_size = os.path.getsize(latest_file)
        if file_size > 50 * 1024 * 1024:  # > 50MB
            logger.warning(f"Large file detected. Sampling for accuracy validation...")
            return self._sample_for_validation(latest_file)
        else:
            with open(latest_file, 'r') as f:
                return json.load(f)
    
    def _sample_for_validation(self, file_path: str, sample_size: int = 100) -> Dict:
        """Sample documents for detailed accuracy validation"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if 'document_results' in data and len(data['document_results']) > sample_size:
            # Take stratified sample
            docs = data['document_results']
            sampled_docs = random.sample(docs, min(sample_size, len(docs)))
            data['document_results'] = sampled_docs
            data['validation_sample'] = True
            logger.info(f"Sampled {len(sampled_docs)} documents for validation")
        
        return data
    
    def validate_extraction_quality(self, results: Dict) -> Dict:
        """Main validation: Check if our extractions are actually correct metadiscourse"""
        
        if 'document_results' not in results:
            return {'error': 'No document results available for validation'}
        
        validation_results = {
            'total_documents': len(results['document_results']),
            'total_markers_analyzed': 0,
            'category_accuracy': {},
            'context_analysis': {},
            'quality_issues': [],
            'high_confidence_extractions': [],
            'questionable_extractions': []
        }
        
        category_stats = defaultdict(lambda: {
            'total_count': 0,
            'high_quality': 0,
            'questionable': 0,
            'context_appropriate': 0,
            'examples': {'good': [], 'questionable': []}
        })
        
        # Analyze a sample of documents in detail
        sample_docs = random.sample(results['document_results'], 
                                   min(20, len(results['document_results'])))
        
        for doc in sample_docs:
            if 'detailed_markers' not in doc:
                continue
                
            doc_text = doc.get('text', '').lower()
            
            # Check document context (academic vs narrative)
            doc_academic_score = self._assess_academic_context(doc_text)
            
            for marker in doc['detailed_markers']:
                category = marker.get('category', 'unknown')
                text = marker.get('text', '')
                context = marker.get('context', '')
                confidence = marker.get('confidence', 0)
                
                validation_results['total_markers_analyzed'] += 1
                category_stats[category]['total_count'] += 1
                
                # Validate this specific extraction
                quality_score = self._validate_single_extraction(
                    category, text, context, doc_text
                )
                
                if quality_score >= 0.8:
                    category_stats[category]['high_quality'] += 1
                    if confidence >= 0.7:
                        validation_results['high_confidence_extractions'].append({
                            'category': category,
                            'text': text,
                            'context': context[:100],
                            'confidence': confidence,
                            'quality_score': quality_score
                        })
                elif quality_score <= 0.4:
                    category_stats[category]['questionable'] += 1
                    validation_results['questionable_extractions'].append({
                        'category': category,
                        'text': text,
                        'context': context[:100],
                        'confidence': confidence,
                        'quality_score': quality_score,
                        'issue': self._identify_quality_issue(category, text, context)
                    })
                
                # Context appropriateness
                if doc_academic_score >= 0.6:
                    category_stats[category]['context_appropriate'] += 1
                
                # Collect examples
                if quality_score >= 0.8 and len(category_stats[category]['examples']['good']) < 5:
                    category_stats[category]['examples']['good'].append({
                        'text': text,
                        'context': context[:80],
                        'score': quality_score
                    })
                elif quality_score <= 0.4 and len(category_stats[category]['examples']['questionable']) < 5:
                    category_stats[category]['examples']['questionable'].append({
                        'text': text,
                        'context': context[:80],
                        'score': quality_score
                    })
        
        # Calculate accuracy percentages
        for category, stats in category_stats.items():
            if stats['total_count'] > 0:
                validation_results['category_accuracy'][category] = {
                    'total_analyzed': stats['total_count'],
                    'high_quality_pct': (stats['high_quality'] / stats['total_count']) * 100,
                    'questionable_pct': (stats['questionable'] / stats['total_count']) * 100,
                    'context_appropriate_pct': (stats['context_appropriate'] / stats['total_count']) * 100,
                    'examples': stats['examples']
                }
        
        # Overall accuracy assessment
        validation_results['overall_assessment'] = self._generate_overall_assessment(validation_results)
        
        return validation_results
    
    def _validate_single_extraction(self, category: str, text: str, context: str, doc_text: str) -> float:
        """Validate a single metadiscourse extraction"""
        score = 0.5  # Base score
        
        text_lower = text.lower().strip()
        context_lower = context.lower()
        
        if category not in self.gold_standard_patterns:
            return score
        
        patterns = self.gold_standard_patterns[category]
        
        # Check against definite positive patterns
        if 'definite' in patterns:
            for pattern in patterns['definite']:
                if pattern in text_lower:
                    score += 0.3
                    break
        
        # Check contextual patterns
        if 'contextual' in patterns:
            for pattern in patterns['contextual']:
                if pattern in text_lower and self._check_academic_context(context_lower):
                    score += 0.2
                    break
        
        # Penalty for avoid patterns
        if 'avoid' in patterns:
            for avoid_pattern in patterns['avoid']:
                if avoid_pattern in context_lower:
                    score -= 0.4
                    break
        
        # Academic context bonus
        if self._check_academic_context(context_lower):
            score += 0.1
        
        # Non-academic context penalty
        if self._check_non_academic_context(context_lower):
            score -= 0.3
        
        # Category-specific validation
        score += self._category_specific_validation(category, text_lower, context_lower)
        
        return max(0, min(1, score))
    
    def _category_specific_validation(self, category: str, text: str, context: str) -> float:
        """Category-specific validation logic"""
        bonus = 0
        
        if category == 'hedges':
            # Good: academic hedging
            if any(word in context for word in ['argue', 'suggest', 'indicate', 'seem', 'appear']):
                bonus += 0.1
            # Bad: conversational hedging
            if any(word in context for word in ['maybe tomorrow', 'might go', 'could be fun']):
                bonus -= 0.2
                
        elif category == 'self_mentions':
            # Good: academic self-reference
            if any(phrase in context for phrase in ['i argue', 'we propose', 'our study', 'my research']):
                bonus += 0.2
            # Bad: personal narrative
            if any(phrase in context for phrase in ['i went', 'we were', 'my family', 'our vacation']):
                bonus -= 0.3
                
        elif category == 'transitions':
            # Good: logical connection
            if re.search(r'\b(?:however|therefore|furthermore)\b.*\b(?:this|these|that|those)\b', context):
                bonus += 0.1
            # Good: sentence-initial position
            if re.match(r'^\s*(?:however|therefore|furthermore|moreover)', context):
                bonus += 0.1
                
        elif category == 'engagement_markers':
            # Good: reader-directed academic language
            if any(phrase in context for phrase in ['consider that', 'note that', 'observe', 'you can see that']):
                bonus += 0.2
            # Bad: personal address
            if any(phrase in context for phrase in ['you are', 'you have', 'you should']):
                bonus -= 0.2
        
        return bonus
    
    def _assess_academic_context(self, text: str) -> float:
        """Assess if document has academic context"""
        academic_matches = sum(1 for pattern in self.academic_context_indicators 
                             if re.search(pattern, text, re.IGNORECASE))
        non_academic_matches = sum(1 for pattern in self.non_academic_indicators 
                                 if re.search(pattern, text, re.IGNORECASE))
        
        total_indicators = academic_matches + non_academic_matches
        if total_indicators == 0:
            return 0.5
        
        return academic_matches / total_indicators
    
    def _check_academic_context(self, context: str) -> bool:
        """Check if context suggests academic discourse"""
        return any(re.search(pattern, context, re.IGNORECASE) 
                  for pattern in self.academic_context_indicators)
    
    def _check_non_academic_context(self, context: str) -> bool:
        """Check if context suggests non-academic discourse"""
        return any(re.search(pattern, context, re.IGNORECASE) 
                  for pattern in self.non_academic_indicators)
    
    def _identify_quality_issue(self, category: str, text: str, context: str) -> str:
        """Identify specific quality issues with extraction"""
        context_lower = context.lower()
        
        if category == 'self_mentions':
            if any(phrase in context_lower for phrase in ['i went', 'we were', 'my family']):
                return "Personal narrative, not academic self-reference"
        
        elif category == 'hedges':
            if any(phrase in context_lower for phrase in ['maybe tomorrow', 'might go']):
                return "Conversational hedging, not academic uncertainty"
        
        elif category == 'engagement_markers':
            if any(phrase in context_lower for phrase in ['you are', 'you have']):
                return "Personal address, not academic reader engagement"
        
        elif category == 'transitions':
            if any(phrase in context_lower for phrase in ['then we went', 'after we']):
                return "Narrative sequence, not logical transition"
        
        if self._check_non_academic_context(context_lower):
            return "Non-academic context"
        
        return "Low quality pattern match"
    
    def _generate_overall_assessment(self, validation_results: Dict) -> Dict:
        """Generate overall quality assessment"""
        category_accuracies = validation_results['category_accuracy']
        
        if not category_accuracies:
            return {'grade': 'F', 'message': 'No data to assess'}
        
        # Calculate weighted average quality
        total_markers = sum(cat['total_analyzed'] for cat in category_accuracies.values())
        weighted_quality = sum(
            cat['high_quality_pct'] * cat['total_analyzed'] 
            for cat in category_accuracies.values()
        ) / total_markers if total_markers > 0 else 0
        
        # Grade the overall quality
        if weighted_quality >= 85:
            grade = "A (Excellent)"
            message = "High-quality extractions with minimal false positives"
        elif weighted_quality >= 75:
            grade = "B (Good)"
            message = "Good extraction quality with some room for improvement"
        elif weighted_quality >= 65:
            grade = "C (Acceptable)"
            message = "Acceptable quality but needs refinement"
        elif weighted_quality >= 50:
            grade = "D (Poor)"
            message = "Poor quality with many false positives"
        else:
            grade = "F (Failing)"
            message = "Unacceptable quality, major issues with extraction"
        
        # Identify top issues
        questionable_count = len(validation_results['questionable_extractions'])
        total_analyzed = validation_results['total_markers_analyzed']
        false_positive_rate = (questionable_count / total_analyzed * 100) if total_analyzed > 0 else 0
        
        recommendations = []
        
        # Category-specific recommendations
        for category, data in category_accuracies.items():
            if data['questionable_pct'] > 30:
                recommendations.append(f"Review {category} patterns - {data['questionable_pct']:.1f}% questionable")
        
        if false_positive_rate > 25:
            recommendations.append(f"High false positive rate ({false_positive_rate:.1f}%) - tighten patterns")
        
        return {
            'grade': grade,
            'message': message,
            'overall_quality_pct': weighted_quality,
            'false_positive_rate': false_positive_rate,
            'recommendations': recommendations,
            'strength_categories': [cat for cat, data in category_accuracies.items() 
                                  if data['high_quality_pct'] >= 80],
            'problem_categories': [cat for cat, data in category_accuracies.items() 
                                 if data['questionable_pct'] >= 30]
        }
    
    def generate_accuracy_report(self) -> Dict:
        """Generate comprehensive extraction accuracy report"""
        logger.info("Starting extraction accuracy validation...")
        
        try:
            results = self.load_latest_results()
            validation = self.validate_extraction_quality(results)
            
            report = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'validation_type': 'Extraction Accuracy (Learner Corpus)',
                'focus': 'Quality of detected metadiscourse markers',
                'sample_size': validation['total_documents'],
                'markers_analyzed': validation['total_markers_analyzed'],
                'results': validation
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error during extraction validation: {e}")
            return {
                'timestamp': pd.Timestamp.now().isoformat(),
                'error': str(e),
                'status': 'Failed'
            }

def main():
    """Run extraction accuracy validation"""
    validator = ExtractionAccuracyValidator()
    report = validator.generate_accuracy_report()
    
    # Save report
    output_file = f"extraction_accuracy_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print results
    print("\n" + "="*70)
    print("🔍 METADISCOURSE EXTRACTION ACCURACY VALIDATION")
    print("="*70)
    
    if 'error' in report:
        print(f"❌ ERROR: {report['error']}")
        return
    
    results = report['results']
    assessment = results['overall_assessment']
    
    print(f"📊 Overall Quality: {assessment['grade']}")
    print(f"💬 Assessment: {assessment['message']}")
    print(f"📈 Quality Score: {assessment['overall_quality_pct']:.1f}%")
    print(f"🚨 False Positive Rate: {assessment['false_positive_rate']:.1f}%")
    
    print(f"\n📋 ANALYSIS SUMMARY")
    print(f"   Documents Sampled: {report['sample_size']}")
    print(f"   Markers Analyzed: {report['markers_analyzed']}")
    
    # Category breakdown
    print(f"\n🏷️ CATEGORY QUALITY BREAKDOWN")
    for category, data in results['category_accuracy'].items():
        print(f"   {category.title()}: {data['high_quality_pct']:.1f}% high quality "
              f"({data['total_analyzed']} analyzed)")
    
    # Strengths and issues
    if assessment['strength_categories']:
        print(f"\n✅ STRONG CATEGORIES: {', '.join(assessment['strength_categories'])}")
    
    if assessment['problem_categories']:
        print(f"\n⚠️ NEEDS ATTENTION: {', '.join(assessment['problem_categories'])}")
    
    # Recommendations
    if assessment['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in assessment['recommendations']:
            print(f"   • {rec}")
    
    # Examples
    if results['high_confidence_extractions']:
        print(f"\n🌟 EXCELLENT EXTRACTIONS (Sample):")
        for example in results['high_confidence_extractions'][:3]:
            print(f"   • {example['category']}: '{example['text']}' "
                  f"(confidence: {example['confidence']:.2f})")
    
    if results['questionable_extractions']:
        print(f"\n❓ QUESTIONABLE EXTRACTIONS (Sample):")
        for example in results['questionable_extractions'][:3]:
            print(f"   • {example['category']}: '{example['text']}' "
                  f"({example['issue']})")
    
    print(f"\n📄 Full report saved to: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main() 