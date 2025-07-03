"""
Metalinguistics: Advanced Metadiscourse Analysis Library

A comprehensive NLP library for detecting and analyzing metadiscourse markers
in academic texts using transformer models and machine learning.
"""

__version__ = "0.1.0"
__author__ = "Fatih Bozdag"

from .analyzers import EnhancedMetadiscourseAnalyzer
from .features import SpacyFeatureExtractor
from .ml import MetadiscourseClassifier

__all__ = [
    "EnhancedMetadiscourseAnalyzer",
    "SpacyFeatureExtractor", 
    "MetadiscourseClassifier"
]