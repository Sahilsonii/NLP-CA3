"""
SMS Spam Detection - Source Package

This package contains modules for:
- Data preprocessing
- Feature extraction
- Model training
- Evaluation and visualization
"""

from .data_preprocessing import preprocess_text, preprocess_pipeline
from .feature_extraction import TFIDFExtractor, SequenceExtractor
from .models import SpamDetector
from .evaluation import calculate_metrics, plot_confusion_matrix, compare_models

__version__ = "1.0.0"
__author__ = "NLP Case Study"
