"""
SoundPose: A transformer-based framework for quantitative diagnosis of voice/sound anomalies.

SoundPose is a cutting-edge framework for detecting, quantifying, and diagnosing anomalies
in voice and sound patterns using transformer-based architecture and generative AI techniques.
"""

__version__ = "0.1.0"

from soundpose.analyzer import SoundPoseAnalyzer
from soundpose.baseline import BaselineBuilder
from soundpose.monitor import SoundPoseMonitor
from soundpose.results import AnalysisResults

__all__ = [
    "SoundPoseAnalyzer",
    "BaselineBuilder",
    "SoundPoseMonitor",
    "AnalysisResults",
]
