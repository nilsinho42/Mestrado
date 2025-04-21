"""
Main entry point for the video processing application.
Utilizes core components to implement the video processing pipeline.
"""

import os
import argparse
import logging
from pathlib import Path
import glob
from core.tracking import create_tracker, Detection
from dataclasses import asdict
import time
# Load environment variables from root directory
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Import core components
from core import (
    ImageAnalysisProcessor, create_detector, create_tracker, create_cost_calculator
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
    logger.info(f"Getting detector for provider: {provider}")
    # Convert provider name to standard form for lookup
    lookup_key = provider.lower()

    logger.info(f"Lookup key for provider {provider}: {lookup_key}")
    logger.info(f"Available detectors: {list(_detectors.keys())}")
    
    # Return from global dict if available
    if lookup_key in _detectors:
        logger.info(f"Found detector for {lookup_key} in global registry")
        return _detectors[lookup_key]

def setup_directories():
    """Set up the required directories."""
    directories = [
        "./data",
        "./data/object_detection",
        "./data/object_detection/images",
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
        try:
            self.db = Database()
            self.db.create_tables()
        except Exception as db_error:
            logger.warning(f"Failed to initialize database: {db_error}. Continuing without database.")
            self.db = Database(disable_db=True)
        
        # Initialize processors
        self.image_processor = ImageAnalysisProcessor(output_dir=str(self.output_dir))
        
        # Initialize cost calculator
        self.cost_calculator = create_cost_calculator(config_path="cost_config.ini")
        
        # Initialize detectors
        self._initialize_detectors()
        
        # Initialize trackers
        # self._initialize_trackers()
        
        logger.info("Video pipeline initialized successfully")
    
    def _initialize_detectors(self):
        """Initialize detectors for each provider with simplified approach."""
        global _detectors
        
        # Local detector based on YOLOv11n
        try:
            logger.info("Initializing local detector for image processing...")
            local_detector = create_detector(
                provider="local",
                model_path="yolo11n.pt",
                confidence_threshold=0.20  # Slightly lower threshold to detect more objects
            )
            self.image_processor.register_detector("local", local_detector)
            _detectors["local"] = local_detector  # Add to global dict
            logger.info("Local detector initialized for image processing")
        except Exception as e:
            error_msg = f"Failed to initialize local detector: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # AWS Rekognition detector
        aws_detector = create_detector(provider="aws")
        self.image_processor.register_detector("aws", aws_detector)
        _detectors["aws"] = aws_detector  # Add to global dict
        
        # Azure Vision detector
        logger.info("Initializing Azure Vision detector...")
        azure_detector = create_detector(provider="azure")
        self.image_processor.register_detector("azure", azure_detector)
        _detectors["azure"] = azure_detector  # Add to global dict
        logger.info("Azure Vision detector initialized")
        
        logger.info("All detectors initialized successfully")
    
    # def _initialize_trackers(self):
    #     """Initialize trackers for each provider."""
    #     # DeepSORT tracker with dedicated tracking detector
    #     try:
    #         local_tracker = create_tracker(
    #             max_age=30,
    #             n_init=3
    #         )
    #         self.tracking_processor.register_tracker("local", local_tracker)
    #         logger.info("DeepSORT tracker initialized for local processing")
    #     except Exception as e:
    #         error_msg = f"Failed to initialize DeepSORT tracker: {e}"
    #         logger.error(error_msg)
    #         raise RuntimeError(error_msg)
        
    #     # AWS and Azure trackers use the same DeepSORT implementation
    #     # rather than relying on cloud services
    #     aws_tracker = local_tracker  # Reuse the same tracker
    #     self.tracking_processor.register_tracker("aws", aws_tracker)
        
    #     azure_tracker = local_tracker  # Reuse the same tracker
    #     self.tracking_processor.register_tracker("azure", azure_tracker)
        
    #     logger.info("Trackers initialized successfully")
    
    
    def process_video(self, video_path, job_id, providers=['local', 'aws', 'azure']):
        """
        Process a video using detection methods and return results.
        
        Args:
            video_path: Path to video file
            providers: List of providers to use ['local', 'aws', 'azure']
            
        Returns:
            Dictionary with job ID, video path, and results from both tasks
        """
        # Create output directory for this job
        output_dir = os.path.join(self.output_dir, job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Processing video {video_path} with providers: {providers}")
        
        # Temporarily set the output directory for this job
        original_output_dir = self.image_processor.output_dir
        self.image_processor.output_dir = Path(output_dir)

        try:
            # Extract frames from video
            video_info = self.image_processor.extract_frames(
                video_path=video_path)
            
            # First look directly in the output directory
            frame_paths = glob.glob(os.path.join(output_dir, "*.jpg"))
            frame_paths.sort()  # Ensure order
            if not frame_paths:
                logger.warning(f"No frames found for video {video_path}")
                return None
                
        finally:
            # Restore the original output directory
            self.image_processor.output_dir = original_output_dir
        
        # Process frames with each requested provider
        all_results = {}
        latency_metrics = {'video_id': job_id, 'frames': len(frame_paths)}
        processing_time_metrics = {'video_id': job_id, 'frames': len(frame_paths)} 
        fps_metrics = {'video_id': job_id, 'frames': len(frame_paths)}
        count_vehicles = {'video_id': job_id, 'frames': len(frame_paths)}
        count_people = {'video_id': job_id, 'frames': len(frame_paths)}

        for provider in providers:
            tracker = create_tracker()
            logger.info(f"Processing frames with {provider} provider")
            processing_time = time.time()

            # Process all frames with this detector
            frame_results = []
            
            # Process each frame    
            for frame_path in frame_paths:
                # Extract frame number from path for tracking
                frame_number = int(os.path.basename(frame_path).split('_')[-1].split('.')[0])
                
                # Process image using provider's detector
                # try:
                logger.info(f"Processing image {frame_path} with provider {provider}")
                # Load image directly to verify it exists and can be loaded
                
                detections, latency = self.image_processor.process_image(
                    image_path=frame_path,
                    provider=provider
                )
                
                if detections:
                    detections = [Detection(**d, frame_number=frame_number) for d in detections]
                    
                # Add frame info to results
                result = {
                    'frame_path': frame_path,
                    'frame_number': frame_number,
                    'detections': [asdict(d) for d in detections],
                    'latency': latency
                }
                
                frame_results.append(result)
                # except Exception as e:
                    # logger.error(f"Error processing frame {frame_path} with {provider}: {e}")

                # Using detections, run tracking
                tracker.process_frame(detections)

                
            # Append results for this provider
            all_results[provider] = {
                'frame_results': frame_results,
                'tracked_objects': tracker.get_results()  # Will be filled by tracking step
            }
            print(tracker.get_results())

            # Calculate average latency for this provider
            avg_latency = sum(f['latency'] for f in frame_results) / len(frame_results)
            processing_time = time.time() - processing_time 

            if provider == 'azure':
                latency_metrics['latency_azure_ms'] = round(avg_latency*1000, 0)
                processing_time_metrics['pt_azure_sec'] = round(processing_time, 0)
                fps_metrics['fps_azure'] = round(len(frame_results)/processing_time, 0)
            elif provider == 'aws':
                latency_metrics['latency_aws_ms'] = round(avg_latency*1000, 0)
                processing_time_metrics['pt_aws_sec'] = round(processing_time, 0)
                fps_metrics['fps_aws'] = round(len(frame_results)/processing_time, 0)
            elif provider == 'gcp':
                latency_metrics['latency_gcp_ms'] = round(avg_latency*1000, 0)
                processing_time_metrics['pt_gcp_sec'] = round(processing_time, 0)
                fps_metrics['fps_gcp'] = round(len(frame_results)/processing_time, 0)
            elif provider == 'local':
                latency_metrics['latency_edge_ms'] = round(avg_latency*1000, 0)
                processing_time_metrics['pt_edge_sec'] = round(processing_time, 0)
                fps_metrics['fps_edge'] = round(len(frame_results)/processing_time, 0)
        # Load results into postgres database
        self.db.load_data(table_name="fps_metrics", data=all_results)

        return all_results
    
    # def test_local_detector(self, image_path=None):
    #     """
    #     Test the local detector on a sample image.
        
    #     Args:
    #         image_path: Path to test image, or None to generate a test image
            
    #     Returns:
    #         Detection results
    #     """
    #     # Initialize detectors if needed
    #     self._initialize_detectors()
        
    #     # Use provided image or create a blank test image with a square in it
    #     if image_path and os.path.exists(image_path):
    #         image = cv2.imread(image_path)
    #         logger.info(f"Using test image: {image_path}")
    #     else:
    #         logger.info("Creating test image with rectangles (simulating objects)")
    #         # Create a blank image
    #         image = np.zeros((720, 1280, 3), dtype=np.uint8)
            
    #         # Draw rectangles of different colors (simulating objects)
    #         # Green rectangle (potential person)
    #         cv2.rectangle(image, (300, 200), (400, 500), (0, 255, 0), -1)
    #         # Red rectangle (potential car)
    #         cv2.rectangle(image, (700, 300), (900, 400), (0, 0, 255), -1)
    #         # Blue rectangle
    #         cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)
            
    #         # Save the test image
    #         test_img_path = "test_image.jpg"
    #         cv2.imwrite(test_img_path, image)
    #         logger.info(f"Test image saved to: {test_img_path}")
    #         image_path = test_img_path
        
    #     # Get the local detector
    #     local_detector = _detectors.get("local")
    #     if not local_detector:
    #         logger.info("Local detector not found. Initializing...")
    #         local_detector = create_detector(
    #             provider="local",
    #             model_path="yolo11n.pt",
    #             confidence_threshold=0.10  # Very low threshold for testing
    #         )
        
    #     logger.info(f"Running detection with confidence threshold: {local_detector.confidence_threshold}")
        
    #     # Process image with local detector
    #     detections = local_detector.process_image(image)
        
    #     # Print detection results
    #     logger.info(f"Found {len(detections)} detections:")
    #     for i, detection in enumerate(detections):
    #         logger.info(f"  Detection {i+1}: {detection['detection_type']} with confidence {detection['confidence']:.2f}")
        
    #     # Save a visualization of the detections
    #     result_image = image.copy()
    #     for detection in detections:
    #         bbox = detection["bbox"]
    #         x1, y1, x2, y2 = [int(coord) for coord in bbox]
    #         label = f"{detection['detection_type']}: {detection['confidence']:.2f}"
            
    #         # Draw bounding box
    #         cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    #         # Draw label
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(result_image, label, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)
        
    #     # Save the result
    #     result_path = "test_result.jpg"
    #     cv2.imwrite(result_path, result_image)
    #     logger.info(f"Saved detection visualization to: {result_path}")
        
    #     return detections

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