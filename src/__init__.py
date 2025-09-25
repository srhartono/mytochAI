"""
mytochAI
"""

__version__ = "0.0.1"
__author__ = "srhartono"
__email__ = "srhartono@ucdavis.edu"

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