"""
Core functionality for ML processing tasks.
"""

# Expose key classes and functions from submodules
from .models import YOLODetector, AWSRekognitionDetector, AzureVisionDetector, create_detector
from .tracking import Detection, TrackedObject, DeepSORTTracker, create_tracker
from .video import VideoProcessor, ImageAnalysisProcessor
from .cost_calculator import CostCalculator, create_cost_calculator
