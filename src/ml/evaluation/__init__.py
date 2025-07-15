"""
ML Evaluation Package

This package contains machine learning evaluation components for the AI Automation Bot,
including metrics calculation, model comparison, and performance analysis.
"""

from .metrics import MetricsCalculator
from .comparator import ModelComparator
from .analyzer import PerformanceAnalyzer
from .visualizer import ResultsVisualizer

__all__ = [
    'MetricsCalculator',
    'ModelComparator', 
    'PerformanceAnalyzer',
    'ResultsVisualizer'
] 