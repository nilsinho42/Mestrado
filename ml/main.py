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
import requests
import base64

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

def convert_detection(detection: Detection, provider: str) -> dict:
    """Convert our Detection object to the format expected by tracker_app"""

    if provider == 'azure':
        logger.info(f"Detection class: {detection.class_id} {detection.class_name}")
        box_data = detection.bbox._data  # Extract the dict from ImageBoundingBox
        result = {
            'box': [box_data['x'], box_data['y'], box_data['w'], box_data['h']],
            'confidence': detection.confidence,
            'class_id': detection.class_id if detection.class_id is not None else 0,
            'class_name': detection.class_name
        }
    else:  # aws, gcp, or edge
        result = {
            'box': detection.bbox,  # Already in [x1, y1, x2, y2] format
            'confidence': detection.confidence,
            'class_id': detection.class_id if detection.class_id is not None else 0,
            'class_name': detection.class_name
        }
    return result

def process_response(response: requests.Response, tracking_data: dict) -> dict:
    """Process the response from the tracker_app."""
    resp_data = response.json()

    tracks = resp_data.get('tracks', [])
    for track in tracks:
        track_id = track.get('track_id')
        class_name = track.get('class_name', '').lower()
        
        # Check if this is a vehicle
        if 'car' in class_name or 'truck' in class_name or 'bus' in class_name or 'vehicle' in class_name:
            if track_id not in tracking_data['tracked_ids']['vehicle']:
                tracking_data['tracked_ids']['vehicle'].add(track_id)
                tracking_data['vehicle_count'] += 1
        elif 'person' in class_name or 'pedestrian' in class_name:
            if track_id not in tracking_data['tracked_ids']['person']:
                tracking_data['tracked_ids']['person'].add(track_id)
                tracking_data['person_count'] += 1
    
    return tracking_data

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
        
        # Initialize processors
        self.image_processor = ImageAnalysisProcessor(output_dir=str(self.output_dir))
        
        # Initialize cost calculator
        self.cost_calculator = create_cost_calculator(config_path="cost_config.ini")
        
        # Set up tracker endpoints
        self.azure_deepsort_tracker = os.getenv("AZURE_DEEPSORT_ENDPOINT")
        self.aws_deepsort_tracker = os.getenv("AWS_FARGATE_ENDPOINT")
        self.gcp_deepsort_tracker = os.getenv("GCP_DEEPSORT_TRACKER")   
        self.edge_deepsort_tracker = os.getenv("EDGE_DEEPSORT_ENDPOINT")
        
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
        
        # Google Cloud Vision detector
        logger.info("Initializing Google Cloud Vision detector...")
        gcp_detector = create_detector(provider="gcp")
        self.image_processor.register_detector("gcp", gcp_detector)
        _detectors["gcp"] = gcp_detector  # Add to global dict
        logger.info("Google Cloud Vision detector initialized")
        
        # Edge detector (Raspberry Pi)
        try:
            logger.info("Initializing Edge detector for Raspberry Pi processing...")
            edge_detector = create_detector(
                provider="edge",
                edge_endpoint=self.edge_deepsort_tracker,
                confidence_threshold=0.20,
                verify_connection=False  # More robust, won't fail if Raspberry Pi is not available at startup
            )
            self.image_processor.register_detector("edge", edge_detector)
            _detectors["edge"] = edge_detector  # Add to global dict
            logger.info("Edge detector initialized for Raspberry Pi processing")
        except Exception as e:
            error_msg = f"Failed to initialize edge detector: {e}"
            logger.error(error_msg)
            logger.warning("Edge processing will not be available")
        
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
    


    def process_video(self, video_path, job_id, providers=['edge', 'aws', 'azure', 'gcp'], expected_vehicles=0, expected_people=0):
        """
        Process a video using detection methods and return results.
        
        Args:
            video_path: Path to video file
            job_id: Unique identifier for this processing job
            providers: List of providers to use ['local', 'aws', 'azure']
            expected_vehicles: Expected number of vehicles in the video
            expected_people: Expected number of people in the video
            
        Returns:
            Dictionary with job ID, video path, and results from both tasks
        """
        # Create output directory for this job
        output_dir = os.path.join(self.output_dir, job_id)
        os.makedirs(output_dir, exist_ok=True)
        self.db = Database()
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
        count_vehicles = {'video_id': job_id, 'frames': len(frame_paths), 'cv_expected': expected_vehicles}
        count_people = {'video_id': job_id, 'frames': len(frame_paths), 'cp_expected': expected_people}
        precision_recall = {'video_id': job_id, 'frames': len(frame_paths)}
        cost_metrics = {'video_id': job_id, 'frames': len(frame_paths)}

        for provider in providers:
            tracker = create_tracker()
            logger.info(f"Processing frames with {provider} provider")
            processing_time = time.time()

            # Process all frames with this detector
            frame_results = []
            
            tracking_data = {
                'vehicle_count': 0,  # Unique vehicles seen
                'person_count': 0,   # Unique persons seen
                'tracked_ids': {
                    'vehicle': set(),  # Set of unique vehicle IDs
                    'person': set()    # Set of unique person IDs
                }
            }

            # Process each frame    
            for frame_path in frame_paths:
                # Extract frame number from path for tracking
                frame_number = int(os.path.basename(frame_path).split('_')[-1].split('.')[0])
                
                # Process image using provider's detector
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
                    'latency': latency
                }
                
                frame_results.append(result)

                # Using detections, run tracking
                if provider == 'local':
                    frame_result = tracker.process_frame(detections)
                    # Store the response to help with debugging
                    logger.info(f"Local tracking results for frame {frame_number}: {len(frame_result.get('tracks', []))} tracks")
                elif provider == 'aws' and detections:
                    # Read the image and convert to base64
                    with open(frame_path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    # Create payload matching ImageData format
                    payload = {
                        "image": encoded_image,
                        "frame_idx": frame_number,
                        "detections": [convert_detection(det, provider) for det in detections],
                        "video_id": job_id
                    }
                    
                    # Send request to correct endpoint
                    response = requests.post(f"{self.aws_deepsort_tracker}/api/track", json=payload)
                    tracking_data = process_response(response, tracking_data)
                    logger.info(response.json())
                elif provider == 'azure' and detections:
                    # Read the image and convert to base64
                    with open(frame_path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    # Create payload matching ImageData format
                    payload = {
                        "image": encoded_image,
                        "frame_idx": frame_number,
                        "detections": [convert_detection(det, provider) for det in detections],
                        "video_id": job_id
                    }
                    
                    # Send request to correct endpoint
                    logger.info(f"Sending request to {self.azure_deepsort_tracker}/api/track")
                    response = requests.post(f"{self.azure_deepsort_tracker}/api/track", json=payload)
                    tracking_data = process_response(response, tracking_data)
                    logger.info(response.json())
                elif provider == 'gcp' and detections:
                    # Read the image and convert to base64
                    with open(frame_path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    # Create payload matching ImageData format
                    payload = {
                        "image": encoded_image,
                        "frame_idx": frame_number,
                        "detections": [convert_detection(det, provider) for det in detections],
                        "video_id": job_id
                    }
                    
                    # Send request to correct endpoint
                    logger.info(f"Sending request to {self.gcp_deepsort_tracker}/api/track")
                    response = requests.post(f"{self.gcp_deepsort_tracker}/api/track", json=payload)
                    tracking_data = process_response(response, tracking_data)
                    logger.info(response.json())
                elif provider == 'edge' and detections:
                    # Read the image and convert to base64
                    with open(frame_path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    # Create payload matching ImageData format
                    payload = {
                        "image": encoded_image,
                        "frame_idx": frame_number,
                        "detections": [convert_detection(det, provider) for det in detections],
                        "video_id": job_id
                    }
                    
                    # Send request to correct endpoint
                    logger.info(f"Edge processing: Found {len(detections)} detections in frame {frame_number}")
                    try:
                        logger.info(f"Sending request to {self.edge_deepsort_tracker}/api/track")
                        response = requests.post(f"{self.edge_deepsort_tracker}/api/track", json=payload, timeout=10)
                        tracking_data = process_response(response, tracking_data)
                        logger.info(f"Edge tracking successful: {response.json()}")
                    except requests.exceptions.Timeout:
                        logger.error(f"Edge tracking request timed out for frame {frame_number}")
                    except requests.exceptions.ConnectionError as e:
                        logger.error(f"Edge tracking connection error for frame {frame_number}: {str(e)}")
                    except Exception as e:
                        logger.error(f"Edge tracking error for frame {frame_number}: {str(e)}")
                elif provider == 'edge' and not detections:
                    logger.info(f"Edge processing: No detections found in frame {frame_number}, skipping tracking")
                
            if provider != 'local':
                vehicle_count = tracking_data['vehicle_count']
                person_count = tracking_data['person_count']
            else:
                tracking_results = tracker.get_results()
                vehicle_count = tracking_results.get('counts', {}).get('vehicle_count', 0)
                person_count = tracking_results.get('counts', {}).get('person_count', 0)

            print(f"Provider {provider}: Detected {vehicle_count} vehicles and {person_count} people")
            # Calculate average latency for this provider
            avg_latency = sum(f['latency'] for f in frame_results) / len(frame_results)
            processing_time = time.time() - processing_time 

            # Calculate accuracy if expected counts are provided
            if vehicle_count > expected_vehicles:
                true_positives = expected_vehicles
                false_positives = vehicle_count - expected_vehicles
                false_negatives = 0
            else:
                true_positives = vehicle_count
                false_positives = 0
                false_negatives = expected_vehicles - vehicle_count
                
            if expected_vehicles > 0:
                vehicle_precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
                vehicle_recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
            else:
                vehicle_precision = 0
                vehicle_recall = 0
            
            if person_count > expected_people:
                true_positives = expected_people
                false_positives = person_count - expected_people
                false_negatives = 0
            else:
                true_positives = person_count
                false_positives = 0
                false_negatives = expected_people - person_count

            if expected_people > 0:
                person_precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
                person_recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
            else:
                person_precision = 0
                person_recall = 0

            # Append results for this provider
            all_results[provider] = {
                'frame_results': frame_results,
                'tracked_objects': vehicle_count + person_count,
                'vehicle_count': vehicle_count,
                'person_count': person_count
            }

            if provider == 'azure':
                latency_metrics['latency_azure_ms'] = round(avg_latency*1000, 0)                
                processing_time_metrics['pt_azure_sec'] = round(processing_time, 0)
                fps_metrics['fps_azure'] = round(len(frame_results)/processing_time, 1)
                count_vehicles['cv_azure'] = vehicle_count
                count_people['cp_azure'] = person_count
                precision_recall['precision_azure'] = round((vehicle_precision + person_precision)/2, 2)*100
                precision_recall['recall_azure'] = round((vehicle_recall + person_recall)/2, 2)*100
                # cost_metrics['cost_azure'] = round(self.cost_calculator.calculate_cost(provider), 2)
            elif provider == 'aws':
                latency_metrics['latency_aws_ms'] = round(avg_latency*1000, 0)
                processing_time_metrics['pt_aws_sec'] = round(processing_time, 0)
                fps_metrics['fps_aws'] = round(len(frame_results)/processing_time, 1)
                count_vehicles['cv_aws'] = vehicle_count
                count_people['cp_aws'] = person_count
                precision_recall['precision_aws'] = round((vehicle_precision + person_precision)/2, 2)*100
                precision_recall['recall_aws'] = round((vehicle_recall + person_recall)/2, 2)*100
                # cost_metrics['cost_aws'] = round(self.cost_calculator.calculate_cost(provider), 2)
            elif provider == 'gcp':
                latency_metrics['latency_gcp_ms'] = round(avg_latency*1000, 0)
                processing_time_metrics['pt_gcp_sec'] = round(processing_time, 0)
                fps_metrics['fps_gcp'] = round(len(frame_results)/processing_time, 1)
                count_vehicles['cv_gcp'] = vehicle_count
                count_people['cp_gcp'] = person_count
                precision_recall['precision_gcp'] = round((vehicle_precision + person_precision)/2, 2)*100
                precision_recall['recall_gcp'] = round((vehicle_recall + person_recall)/2, 2)*100
                # cost_metrics['cost_gcp'] = round(self.cost_calculator.calculate_cost(provider), 2)
            elif provider == 'edge':
                latency_metrics['latency_edge_ms'] = round(avg_latency*1000, 0)
                processing_time_metrics['pt_edge_sec'] = round(processing_time, 0)
                fps_metrics['fps_edge'] = round(len(frame_results)/processing_time, 1)
                count_vehicles['cv_edge'] = vehicle_count
                count_people['cp_edge'] = person_count
                precision_recall['precision_edge'] = round((vehicle_precision + person_precision)/2, 2)*100
                precision_recall['recall_edge'] = round((vehicle_recall + person_recall)/2, 2)*100
                # cost_metrics['cost_edge'] = round(self.cost_calculator.calculate_cost(provider), 2)
                    
        # Add count data to the final results
        all_results['metrics'] = {
            'latency': latency_metrics,
            'processing_time': processing_time_metrics,
            'fps': fps_metrics,
            'count_vehicles': count_vehicles,
            'count_people': count_people
        }
        
        # Load results into postgres database
        self.db.load_data(table_name="latency_metrics", data=latency_metrics)
        self.db.load_data(table_name="processing_time_metrics", data=processing_time_metrics)
        self.db.load_data(table_name="fps_metrics", data=fps_metrics)
        self.db.load_data(table_name="count_vehicles", data=count_vehicles)
        self.db.load_data(table_name="count_people", data=count_people)
        self.db.load_data(table_name="precision_recall", data=precision_recall)
        self.db.load_data(table_name="cost_metrics", data=cost_metrics)

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
    parser.add_argument('--expected_vehicles', type=int, default=0, help='Expected number of vehicles in the video')
    parser.add_argument('--expected_people', type=int, default=0, help='Expected number of people in the video')
    parser.add_argument('--provider', type=str, default='local', choices=['local', 'aws', 'azure', 'gcp', 'edge'],
                        help='Provider to use for detection and tracking')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VideoPipeline(output_dir=args.output)
    
    # Process video
    job_id = f"job_{int(time.time())}"  # Generate a unique job ID
    results = pipeline.process_video(
        args.video, 
        job_id=job_id,
        providers=[args.provider],
        expected_vehicles=args.expected_vehicles,
        expected_people=args.expected_people
    )
    
    return results

if __name__ == '__main__':
    main() 