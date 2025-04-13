"""
Main entry point for the video processing application.
Utilizes core components to implement the video processing pipeline.
"""

import os
import sys
import time
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import uuid
from datetime import datetime

# Load environment variables from root directory
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Import core components
from core import (
    VideoProcessor, ImageAnalysisProcessor, VideoTrackingProcessor,
    create_detector, create_tracker, create_storage_provider, create_cost_calculator
)
from core.db_utils import Database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global dictionary to store detectors by provider
_detectors = {}

def get_detector_for_provider(provider: str):
    """
    Get a detector for a specific provider from the global _detectors dictionary.
    
    Args:
        provider: Provider name (e.g., 'local', 'aws', 'azure')
        
    Returns:
        Detector instance or None if not found
    """
    # Convert provider name to standard form for lookup
    if provider.lower() in ['local', 'yolo']:
        lookup_key = 'local'
    elif provider.lower() in ['aws', 'rekognition']:
        lookup_key = 'aws'
    elif provider.lower() in ['azure', 'vision']:
        lookup_key = 'azure'
    else:
        lookup_key = provider.lower()
    
    # Return from global dict if available
    if lookup_key in _detectors:
        return _detectors[lookup_key]
    
    # Try to create a new detector if not found
    try:
        detector = create_detector(provider=lookup_key)
        _detectors[lookup_key] = detector
        return detector
    except Exception as e:
        logger.error(f"Failed to create detector for provider {provider}: {e}")
        return None

def setup_directories():
    """Set up the required directories."""
    directories = [
        "./data",
        "./data/object_detection",
        "./data/object_detection/images",
        "./data/object_detection/tracks",
        "./data/results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("Created required directories")

class VideoPipeline:
    def __init__(self, output_dir: str = "./data/object_detection/images"):
        """Initialize video processing pipeline.
        
        Args:
            output_dir: Directory to save extracted images
        """
        # Set up directories
        setup_directories()
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db = Database()
        self.db.create_tables()
        
        # Initialize processors
        self.image_processor = ImageAnalysisProcessor(output_dir=str(self.output_dir))
        self.tracking_processor = VideoTrackingProcessor(output_dir="./data/object_detection/tracks")
        
        # Initialize cost calculator
        self.cost_calculator = create_cost_calculator(config_path="cost_config.ini")
        
        # Initialize detectors
        self._initialize_detectors()
        
        # Initialize trackers
        self._initialize_trackers()
        
        # Initialize storage providers
        self._initialize_storage()
        
        logger.info("Video pipeline initialized successfully")
    
    def _initialize_detectors(self):
        """Initialize detectors for each provider."""
        global _detectors
        
        # YOLO detector for local processing
        yolo_detector = create_detector(
            provider="yolo",
            model_path="yolov8n.pt",
            confidence_threshold=0.25
        )
        self.image_processor.register_detector("local", yolo_detector)
        _detectors["local"] = yolo_detector  # Add to global dict
        
        # AWS Rekognition detector
        aws_detector = create_detector(provider="aws")
        self.image_processor.register_detector("aws", aws_detector)
        _detectors["aws"] = aws_detector  # Add to global dict
        
        # Azure Vision detector
        azure_detector = create_detector(provider="azure")
        self.image_processor.register_detector("azure", azure_detector)
        _detectors["azure"] = azure_detector  # Add to global dict
        
        logger.info("Detectors initialized")
    
    def _initialize_trackers(self):
        """Initialize trackers for each provider."""
        # Local DeepSORT tracker
        local_tracker = create_tracker(provider="deepsort")
        self.tracking_processor.register_tracker("local", local_tracker)
        
        # AWS Fargate tracker (uses DeepSORT under the hood)
        aws_tracker = create_tracker(provider="aws")
        self.tracking_processor.register_tracker("aws", aws_tracker)
        
        # Azure Container App tracker (uses DeepSORT under the hood)
        azure_tracker = create_tracker(provider="azure")
        self.tracking_processor.register_tracker("azure", azure_tracker)
        
        logger.info("Trackers initialized")
    
    def _initialize_storage(self):
        """Initialize cloud storage providers."""
        # AWS S3 storage
        try:
            aws_storage = create_storage_provider(provider="aws")
            self.tracking_processor.register_storage_provider("aws", aws_storage)
            logger.info("AWS S3 storage provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize AWS S3 storage: {str(e)}")
        
        # Azure Blob storage
        try:
            # Get Azure connection details from environment variables
#            azure_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
#            azure_container_name = os.getenv("AZURE_CONTAINER_NAME")
#            azure_account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
#            azure_account_key = os.getenv("AZURE_STORAGE_KEY")
            # Create storage provider with explicit parameters
            azure_storage = create_storage_provider(
                provider="azure",
#                connection_string=azure_connection_string,
#                container_name=azure_container_name,
#                account_name=azure_account_name,
#                account_key=azure_account_key
            )
            self.tracking_processor.register_storage_provider("azure", azure_storage)
            logger.info("Azure Blob storage provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Azure Blob storage: {str(e)}")
        
        logger.info("Storage providers initialization completed")
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process video through the pipeline.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing video: {video_path}")
        video_path = Path(video_path)
        
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return {'error': 'Video file not found'}
        
        # Generate a unique ID for this processing job
        job_id = str(uuid.uuid4())
        
        try:
            # Process Task A: Image Analysis with Object Detection
            task_a_results = self._process_task_a(video_path)
            
            # Process Task B: Video Processing with Object Tracking
            task_b_results = self._process_task_b(video_path)
            
            # Combine results
            results = {
                'job_id': job_id,
                'video_path': str(video_path),
                'task_a': task_a_results,
                'task_b': task_b_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save results to file
            result_file = Path("./data/results") / f"results_{video_path.stem}.json"
            with open(result_file, "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Video processing completed successfully. Results saved to {result_file}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            return {'error': str(e), 'job_id': job_id}
    
    def _process_task_a(self, video_path: Path) -> Dict[str, Any]:
        """Process Task A: Image Analysis with Object Detection.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with Task A results
        """
        logger.info("Starting Task A: Image Analysis with Object Detection")
        
        # Process video with image analysis processor
        results = self.image_processor.process_video(
            video_path=str(video_path),
            fps_reduction_factor=5,  # Sample 1 out of every 5 frames
            providers=["local", "aws", "azure"]
        )
        
        # Store metrics in database
        self._store_task_a_metrics(results)
        
        logger.info("Task A processing completed")
        return results
    
    def _process_task_b(self, video_path: Path) -> Dict[str, Any]:
        """Process Task B: Video Processing with Object Tracking.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with Task B results
        """
        logger.info("Starting Task B: Video Processing with Object Tracking")
        
        # Process video with tracking processor
        results = self.tracking_processor.process_video(
            video_path=str(video_path),
            providers=["local", "aws", "azure"]
        )
        
        # Store metrics in database
        self._store_task_b_metrics(results)
        
        logger.info("Task B processing completed")
        return results
    
    def _store_task_a_metrics(self, results: Dict[str, Any]):
        """Store Task A metrics in the database.
        
        Args:
            results: Task A results dictionary
        """
        # For each provider, store metrics
        for provider, provider_results in results.get('providers', {}).items():
            metrics = provider_results.get('metrics', {})
            
            # Calculate cost
            if provider == 'aws':
                cost = self.cost_calculator.calculate_aws_cost(metrics)
            elif provider == 'azure':
                cost = self.cost_calculator.calculate_azure_cost(metrics)
            else:
                cost = self.cost_calculator.calculate_local_cost(metrics)
            
            # Save to database
            self.db.save_metrics({
                'image_id': results.get('video_info', {}).get('video_name', ''),
                'source': provider,
                'latency': metrics.get('avg_latency', 0),
                'cost_image_processing': cost.get('total_cost', 0)
            })
    
    def _store_task_b_metrics(self, results: Dict[str, Any]):
        """Store Task B metrics in the database.
        
        Args:
            results: Task B results dictionary
        """
        # For each provider, store metrics
        for provider, provider_results in results.get('providers', {}).items():
            processing_time = provider_results.get('processing_time', 0)
            
            # Store tracking results
            self.db.save_tracking_results({
                'video_id': results.get('video_name', ''),
                'source': provider,
                'processing_time': processing_time,
                'people_tracked': provider_results.get('summary', {}).get('people_tracked', 0),
                'vehicles_tracked': provider_results.get('summary', {}).get('vehicles_tracked', 0),
                'tracking_data': provider_results
            })
            
            # Save metrics
            self.db.save_metrics({
                'image_id': results.get('video_name', ''),
                'source': provider,
                'total_processing_time': processing_time,
                'cost_video_processing': provider_results.get('cost', {}).get('total_cost', 0)
            })

def main():
    """Main entry point for video processing pipeline."""
    parser = argparse.ArgumentParser(description='Video Processing Pipeline')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='./data/object_detection/images', help='Output directory for frames')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VideoPipeline(output_dir=args.output)
    
    # Process video
    pipeline.process_video(args.video)

if __name__ == '__main__':
    main() 