"""
Core functionality for ML processing tasks.
"""

# Expose key classes and functions from submodules
from .cloud import CloudStorageProvider, AWSS3Storage, AzureBlobStorage, create_storage_provider
from .models import ObjectDetector, YOLODetector, AWSRekognitionDetector, AzureVisionDetector, create_detector
from .tracking import Detection, TrackedObject, BaseTracker, IoUTracker, DeepSORTTracker, create_tracker
from .video import VideoProcessor, ImageAnalysisProcessor, VideoTrackingProcessor
from .metrics import MetricsCollector, CostCalculator, create_cost_calculator
