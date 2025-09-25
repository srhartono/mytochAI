"""
Genomics Correlation Framework

A comprehensive framework for analyzing correlations between SNPs and DNA methylation
using classical machine learning models (LDA, HMM) and modern AI models (GPT, BERT).
"""

__version__ = "1.0.0"
__author__ = "Genomics Research Team"
__email__ = "research@genomics.org"

from .data import DataLoader, SNPProcessor, MethylationProcessor
from .classic_models import ClassicModels
from .ai_models import AIModels
from .analysis import CorrelationAnalyzer
from .visualization import GenomicsPlotter
from .pipeline import GenomicsPipeline

__all__ = [
    "DataLoader",
    "SNPProcessor", 
    "MethylationProcessor",
    "ClassicModels",
    "AIModels",
    "CorrelationAnalyzer",
    "GenomicsPlotter",
    "GenomicsPipeline"
]