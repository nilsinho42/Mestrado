"""
Main entry point for the video processing application.
Utilizes core components to implement the video processing pipeline.
"""

import os
import argparse
import logging
from pathlib import Path
import glob
from dataclasses import dataclass
import time
import requests
import base64
import datetime
from typing import Dict, Any, List, Optional
import threading
import queue
import cv2

# Load environment variables from root directory
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Import core components
from core import (
    ImageAnalysisProcessor, create_detector, create_cost_calculator
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

# Simple Detection dataclass to replace the one from core.tracking
@dataclass
class Detection:
    frame_number: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] or ImageBoundingBox object
    class_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

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
    
    # Return from global dict if available
    if lookup_key in _detectors:
        return _detectors[lookup_key]


def convert_detection(detection: Detection, provider: str) -> dict:
    """Convert our Detection object to the format expected by tracker_app"""
    result = {
        'confidence': detection.confidence,
        'class_id': detection.class_id if detection.class_id is not None else 0,
        'class_name': standardize_class_name(detection.class_name)
    }

    # Each provider's detector returns bounding boxes in different formats:
    # AWS: [x1, y1, x2, y2] from detector - already in the correct format
    # Azure: ImageBoundingBox with _data containing {'x', 'y', 'w', 'h'}
    # GCP: [x1, y1, x2, y2] from detector - already in the correct format
    # Edge: [x1, y1, x2, y2] from detector - already in the correct format
    
    # tracker_app.py expects [x1, y1, x2, y2] format for bbox in Detection objects
    # but allows incoming 'box' in [x, y, w, h] format and converts it internally
    # We'll standardize to [x, y, w, h] format for all providers when sending to tracker
    
    if provider == 'azure':
        box_data = detection.bbox._data 
        x1 = box_data['x']
        y1 = box_data['y']
        x2 = x1 + box_data['w']
        y2 = y1 + box_data['h']
    else:  # aws, gcp, or edge
        # Already in [x1, y1, x2, y2] format
        x1, y1, x2, y2 = detection.bbox

    result.update({'box': [x1, y1, x2, y2]})

    return result

def standardize_class_name(class_name: str) -> str:
    """
    Standardize detection class names to consistent categories.
    Returns 'vehicle', 'person', or None if it doesn't match either category.
    
    Args:
        class_name: The original class name from the detector
        
    Returns:
        Standardized class name ('vehicle', 'person') or None if no match
    """
    if not class_name:
        return None
        
    class_name_lower = class_name.lower()
    
    # Vehicle classes
    vehicle_classes = [
        'car', 'vehicle', 'automobile', 'truck', 'van', 'bus', 'motorcycle', 'bicycle', 
        'transportation', 'taxi', 'ambulance', 'police car', 'suv', 'motorbike'
    ]
    
    # Person classes
    person_classes = [
        'person', 'human', 'people', 'pedestrian', 'man', 'woman', 'child', 'baby'
    ]
    
    # Check for vehicle match
    if any(vc in class_name_lower for vc in vehicle_classes):
        return 'vehicle'
        
    # Check for person match
    if any(pc in class_name_lower for pc in person_classes):
        return 'person'
        
    # No match to our categories
    return None

def process_response(response: requests.Response, tracking_data: dict) -> dict:
    """Process the response from the tracker_app."""
    # Check if response was successful and has content
    if not response or response.status_code != 200 or not response.content:
        logger.info(f"Empty or failed response received: {response.status_code if hasattr(response, 'status_code') else 'No response'}")
        return tracking_data
    
    try:
        resp_data = response.json()
    except (requests.exceptions.JSONDecodeError, ValueError) as e:
        logger.info(f"Failed to decode JSON response: {e}")
        return tracking_data

    # Validate the response structure
    if not isinstance(resp_data, dict) or 'tracks' not in resp_data:
        logger.info(f"Invalid response format: {resp_data}")
        return tracking_data
        
    if not isinstance(resp_data['tracks'], list):
        logger.info(f"Invalid tracks format: {resp_data['tracks']}")
        return tracking_data


    tracks = resp_data.get('tracks', [])
    for track in tracks:
        track_id = track.get('track_id')
        class_name = track.get('class_name', '')
        
        # Get the box dimensions
        box = track.get('box')
        
        # # Calculate box area (width * height)
        # # For [x1, y1, x2, y2] format
        # box_width = box[2] - box[0]
        # box_height = box[3] - box[1]
            
        # # Skip invalid dimensions
        # if box_width <= 0 or box_height <= 0:
        #     logger.warning(f"Invalid box dimensions: width={box_width}, height={box_height}")
        #     continue
                
        # box_size = box_width * box_height
            
        # # Skip small boxes
        # if box_size < min_box_size_threshold:
        #     continue
        
        # The class_name should already be standardized, so we can do direct comparison
        if class_name == 'vehicle':
            if track_id not in tracking_data['tracked_ids']['vehicle']:
                tracking_data['tracked_ids']['vehicle'].add(track_id)
                tracking_data['vehicle_count'] += 1
                # Store latest detection data with class name
                tracking_data['latest_detection'] = {
                    'box': [box[0], box[1], box[2], box[3]],
                    'class_name': 'vehicle'
                }
        elif class_name == 'person':
            if track_id not in tracking_data['tracked_ids']['person']:
                tracking_data['tracked_ids']['person'].add(track_id)
                tracking_data['person_count'] += 1
                # Store latest detection data with class name
                tracking_data['latest_detection'] = {
                    'box': [box[0], box[1], box[2], box[3]],
                    'class_name': 'person'
                }
    
    return tracking_data

def send_request_with_hard_timeout(url, payload, timeout=15, max_wait=30):
    """
    Send a request with a hard timeout using a separate thread.
    This ensures we never wait longer than max_wait seconds, even for SSL errors.
    
    Args:
        url: The API endpoint URL
        payload: The request payload (will be sent as JSON)
        timeout: The requests timeout parameter
        max_wait: Maximum time to wait for the thread to complete
        
    Returns:
        Response object or None if timeout/error occurred
    """
    result_queue = queue.Queue()
    
    def _worker():
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            result_queue.put(response)
        except Exception as e:
            result_queue.put(e)
    
    # Start worker thread
    thread = threading.Thread(target=_worker)
    thread.daemon = True  # Daemon threads will be killed when main thread exits
    thread.start()
    
    try:
        # Wait for result with timeout
        result = result_queue.get(timeout=max_wait)
        if isinstance(result, Exception):
            # Re-raise the exception
            raise result
        return result
    except queue.Empty:
        # Hard timeout reached
        return None
    finally:
        # The thread will be terminated when the function returns since it's a daemon
        pass

class VideoPipeline:
    def __init__(self, output_dir: str = "./data/object_detection/images"):
        """Initialize video processing pipeline.
        
        Args:
            output_dir: Directory to save extracted images
        """
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
        self.cost_calculator = create_cost_calculator(config_path="cost_config_2.ini")
        
        # Set up tracker endpoints
        self.azure_deepsort_tracker = os.getenv("AZURE_DEEPSORT_ENDPOINT")
        self.aws_deepsort_tracker = os.getenv("AWS_FARGATE_ENDPOINT")
        self.gcp_deepsort_tracker = os.getenv("GCP_DEEPSORT_TRACKER")   
        self.edge_deepsort_tracker = os.getenv("EDGE_DEEPSORT_ENDPOINT")
        
        # Initialize detectors
        self._initialize_detectors()
        
        logger.info("Video pipeline initialized successfully")
    
    def _initialize_detectors(self):
        """Initialize detectors for each provider with simplified approach."""
        global _detectors
        
        # AWS Rekognition detector
        aws_detector = create_detector(provider="aws", confidence_threshold=0.5)
        self.image_processor.register_detector("aws", aws_detector)
        _detectors["aws"] = aws_detector  # Add to global dict
        logger.info("AWS detector initialized")
        
        # Azure Vision detector
        azure_detector = create_detector(provider="azure", confidence_threshold=0.5)
        self.image_processor.register_detector("azure", azure_detector)
        _detectors["azure"] = azure_detector  # Add to global dict
        logger.info("Azure Vision detector initialized")
        
        # Google Cloud Vision detector
        gcp_detector = create_detector(provider="gcp", confidence_threshold=0.5)
        self.image_processor.register_detector("gcp", gcp_detector)
        _detectors["gcp"] = gcp_detector  # Add to global dict
        logger.info("Google Cloud Vision detector initialized")
        
        edge_detector = create_detector(
            provider="edge",
            edge_endpoint=self.edge_deepsort_tracker,
            confidence_threshold=0.5,  # Increased from 0.20 to 0.5
            verify_connection=False  # More robust, won't fail if Raspberry Pi is not available at startup
        )
        self.image_processor.register_detector("edge", edge_detector)
        _detectors["edge"] = edge_detector  # Add to global dict
        logger.info("Edge detector initialized for Raspberry Pi processing")
        
    def _filter_out_small_detections(self, detections, frame=None):
        """Filter out detections that are too small."""
        # Set a minimum box size threshold (as a fraction of image dimensions)
        # Boxes smaller than this percentage of the frame will be ignored
        min_box_size_threshold = 0.01  # 1% of frame size
        
        # If we have frame, use its dimensions
        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_width * frame_height
        
        filtered_detections = []    
        for det in detections:
            # Check if we have 'bbox' or 'box' key
            box_key = 'box' if 'box' in det else 'bbox'

            class_name = det['class_name']
            if class_name == 'vehicle':
                min_box_size_threshold = 0.07
            elif class_name == 'person':
                min_box_size_threshold = 0.03
                
            box_width = det[box_key][2] - det[box_key][0]
            box_height = det[box_key][3] - det[box_key][1]
            if box_width <= 0 or box_height <= 0:
                continue
                
            box_size = box_width * box_height
            relative_size = box_size / frame_area  # Normalize by frame size
            
            if relative_size < min_box_size_threshold:
                continue
                
            filtered_detections.append(det)

        return filtered_detections
    
    def process_video(self, video_path, job_id, providers=['edge'], expected_vehicles=0, expected_people=0): #'edge', 'aws', 'azure', 'gcp'
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
            video_info = self.image_processor.extract_frames(video_path=video_path)
            
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
                },
                'latest_detection': None  # Will store the most recent detection with box and class_name
            }

            # Process each frame    
            cost_start_time = datetime.datetime.now(datetime.timezone.utc)
            for frame_path in frame_paths:
                # Extract frame number from path for tracking
                frame_number = int(os.path.basename(frame_path).split('_')[-1].split('.')[0])
                
                # Process image using provider's detector
                # Load image directly to verify it exists and can be loaded
                detections, latency, frame = self.image_processor.process_image(
                    image_path=frame_path,
                    provider=provider
                )
                if detections:
                    detections = [convert_detection(Detection(**det, frame_number=frame_number), provider) for det in detections]
                    detections = self._filter_out_small_detections(detections, frame)
                else:
                    continue

                if not detections:
                    continue

                # Add frame info to results
                result = {
                    'frame_path': frame_path,
                    'frame_number': frame_number,
                    'latency': latency
                }
                
                frame_results.append(result)

                tracking_time = time.time()

                with open(frame_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                image_id = frame_path.split("_")[-1].split(".")[0]
                logger.info(f"[{provider}][{image_id}] Tracking {len(detections)} detections.")

                if provider == 'aws':
                    # Create payload matching ImageData format
                    payload = {
                        "image": encoded_image,
                        "frame_idx": frame_number,
                        "detections": detections,
                        "video_id": job_id,
                        "provider": provider
                    }
                    
                    retry = 0
                    response = None
                    while retry < 3:
                        try:
                            # Use our thread-based hard timeout to prevent hanging
                            response = send_request_with_hard_timeout(
                                f"{self.aws_deepsort_tracker}/api/track", 
                                payload, 
                                timeout=15,
                                max_wait=20  # Never wait more than 20 seconds total
                            )
                            if response is not None and response.status_code == 200:
                                break
                            if response is None:
                                raise TimeoutError("Hard timeout reached waiting for API response")
                        except Exception as e:
                            logger.warning(f"[{provider}][{image_id}] API call failed (attempt {retry+1}/3): {e}")
                            retry += 1
                            time.sleep(0.5)

                elif provider == 'azure':
                    # Create payload matching ImageData format
                    payload = {
                        "image": encoded_image,
                        "frame_idx": frame_number,
                        "detections": detections,
                        "video_id": job_id,
                        "provider": provider
                    }
                    
                    retry = 0
                    response = None
                    while retry < 3:
                        try:
                            # Use our thread-based hard timeout to prevent hanging
                            response = send_request_with_hard_timeout(
                                f"{self.azure_deepsort_tracker}/api/track", 
                                payload, 
                                timeout=15,
                                max_wait=20  # Never wait more than 20 seconds total
                            )
                            if response is not None and response.status_code == 200:
                                break
                            if response is None:
                                raise TimeoutError("Hard timeout reached waiting for API response")
                        except Exception as e:
                            logger.warning(f"[{provider}][{image_id}] API call failed (attempt {retry+1}/3): {e}")
                            retry += 1
                            time.sleep(0.5)

                elif provider == 'gcp':
                    # Create payload matching ImageData format
                    payload = {
                        "image": encoded_image,
                        "frame_idx": frame_number,
                        "detections": detections,
                        "video_id": job_id,
                        "provider": provider
                    }
                    
                    retry = 0
                    response = None
                    while retry < 3:
                        try:
                            # Use our thread-based hard timeout to prevent hanging
                            response = send_request_with_hard_timeout(
                                f"{self.gcp_deepsort_tracker}/api/track", 
                                payload, 
                                timeout=15,
                                max_wait=20  # Never wait more than 20 seconds total
                            )
                            if response is not None and response.status_code == 200:
                                break
                            if response is None:
                                raise TimeoutError("Hard timeout reached waiting for API response")
                        except Exception as e:
                            logger.warning(f"[{provider}][{image_id}] API call failed (attempt {retry+1}/3): {e}")
                            retry += 1
                            time.sleep(0.5)

                elif provider == 'edge':
                    # Create payload matching ImageData format
                    payload = {
                        "image": encoded_image,
                        "frame_idx": frame_number,
                        "detections": detections,
                        "video_id": job_id,
                        "provider": provider
                    }
                    
                    retry = 0
                    response = None
                    while retry < 3:
                        try:
                            # Use our thread-based hard timeout to prevent hanging
                            response = send_request_with_hard_timeout(
                                f"{self.edge_deepsort_tracker}/api/track", 
                                payload, 
                                timeout=10,
                                max_wait=15  # Never wait more than 15 seconds total
                            )
                            if response is not None and response.status_code == 200:
                                break
                            if response is None:
                                raise TimeoutError("Hard timeout reached waiting for API response")
                        except Exception as e:
                            logger.warning(f"[{provider}][{image_id}] API call failed (attempt {retry+1}/3): {e}")
                            retry += 1
                            time.sleep(0.5)
                
                # Skip processing if all API attempts failed
                if response is None or response.status_code != 200:
                    logger.warning(f"[{provider}][{image_id}] All API call attempts failed or returned error, continuing with empty response")
                    # Create an empty success response to allow processing to continue
                    response = type('obj', (object,), {
                        'status_code': 500,
                        'content': b'',
                        'json': lambda: {'tracks': []},
                    })
                    
                # Process the response
                previous_tracking_count = tracking_data['vehicle_count'] + tracking_data['person_count']
                tracking_data = process_response(response, tracking_data)
                new_tracking_count = tracking_data['vehicle_count'] + tracking_data['person_count']
                if new_tracking_count > previous_tracking_count and provider == 'edge' and tracking_data['latest_detection']:
                    latest_detection = tracking_data['latest_detection']
                    box = latest_detection['box']
                    # add detections to the frame
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                    class_name = latest_detection['class_name']
                    # save the frame to [output_dir]/tracked/frame_[provider]_[frame_number].jpg
                    recangle_str = f"x1_{int(box[0])}__y1_{int(box[1])}__x2_{int(box[2])}__y2_{int(box[3])}"
                    frame_filename = f"{provider}_{frame_number:04d}_{recangle_str}.jpg"
                    tracked_dir = Path(output_dir) / "tracked"
                    tracked_dir.mkdir(exist_ok=True)
                    frame_path = os.path.join(tracked_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)

                logger.info(f"[{provider}][{image_id}] Vehicle count: {tracking_data['vehicle_count']}. Person count: {tracking_data['person_count']}.")

            # Get the tracking counts directly from tracking_data
            vehicle_count = tracking_data['vehicle_count']
            person_count = tracking_data['person_count']

            logger.info(f"[{provider}]: Detected {vehicle_count} vehicles and {person_count} people")

            # Calculate average latency for this provider
            avg_latency = sum(f['latency'] for f in frame_results) / len(frame_results)
            processing_time = time.time() - processing_time 
            tracking_time = time.time() - tracking_time
            cost_end_time = datetime.datetime.now(datetime.timezone.utc)

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


            if provider == 'azure':
                latency_metrics['latency_azure_ms'] = round(avg_latency*1000, 0)                
                processing_time_metrics['pt_azure_sec'] = round(processing_time, 0)
                fps_metrics['fps_azure'] = round(len(frame_results)/processing_time, 1)
                count_vehicles['cv_azure'] = vehicle_count
                count_people['cp_azure'] = person_count
                precision_recall['precision_azure'] = round((vehicle_precision + person_precision)/2, 2)*100
                precision_recall['recall_azure'] = round((vehicle_recall + person_recall)/2, 2)*100
                
                from core.cost_calculator import get_azure_metrics
                azure_metrics = get_azure_metrics(cost_start_time, cost_end_time)
                # Calculate cost using metrics
                cost_metrics['cost_azure'] = round(self.cost_calculator.calculate_cost(
                    provider='azure',
                    frame_count=len(frame_paths),
                    cloud_metrics=azure_metrics
                ), 2)
                
            elif provider == 'aws':
                latency_metrics['latency_aws_ms'] = round(avg_latency*1000, 0)
                processing_time_metrics['pt_aws_sec'] = round(processing_time, 0)
                fps_metrics['fps_aws'] = round(len(frame_results)/processing_time, 1)
                count_vehicles['cv_aws'] = vehicle_count
                count_people['cp_aws'] = person_count
                precision_recall['precision_aws'] = round((vehicle_precision + person_precision)/2, 2)*100
                precision_recall['recall_aws'] = round((vehicle_recall + person_recall)/2, 2)*100
                
                from core.cost_calculator import get_aws_metrics
                    
                aws_metrics = get_aws_metrics(start_time=cost_start_time, end_time=cost_end_time)
                # This should not be calculating azure costs with aws metrics
                cost_metrics['cost_aws'] = round(self.cost_calculator.calculate_cost(
                    provider='aws',
                    frame_count=len(frame_paths),
                    cloud_metrics=aws_metrics
                ), 2)
                
            elif provider == 'gcp':
                latency_metrics['latency_gcp_ms'] = round(avg_latency*1000, 0)
                processing_time_metrics['pt_gcp_sec'] = round(processing_time, 0)
                fps_metrics['fps_gcp'] = round(len(frame_results)/processing_time, 1)
                count_vehicles['cv_gcp'] = vehicle_count
                count_people['cp_gcp'] = person_count
                precision_recall['precision_gcp'] = round((vehicle_precision + person_precision)/2, 2)*100
                precision_recall['recall_gcp'] = round((vehicle_recall + person_recall)/2, 2)*100
                
                # Get GCP metrics for cost calculation
                from core.cost_calculator import get_gcp_metrics
                # Use environment variables for GCP resources
                project_id = os.getenv("GCP_PROJECT_ID", "video-processor")
                region = os.getenv("GCP_REGION", "us-central1")
                service_name = os.getenv("GCP_SERVICE_NAME", "gcp-deepsort-tracker")
                
                # Verify that required environment variables are set
                # if not project_id: missing_vars.append("GCP_PROJECT_ID")
                # if not region: missing_vars.append("GCP_REGION")
                # if not service_name: missing_vars.append("GCP_SERVICE_NAME")
                    
                gcp_metrics = get_gcp_metrics(project_id, region, service_name, cost_start_time, cost_end_time)
                    
                # Calculate cost using metrics
                cost_metrics['cost_gcp'] = round(self.cost_calculator.calculate_cost(
                    provider='gcp',
                    frame_count=len(frame_paths),
                    cloud_metrics=gcp_metrics
                ), 2)
                
            elif provider == 'edge':
                latency_metrics['latency_edge_ms'] = round(avg_latency*1000, 0)
                processing_time_metrics['pt_edge_sec'] = round(processing_time, 0)
                fps_metrics['fps_edge'] = round(len(frame_results)/processing_time, 1)
                count_vehicles['cv_edge'] = vehicle_count
                count_people['cp_edge'] = person_count
                precision_recall['precision_edge'] = round((vehicle_precision + person_precision)/2, 2)*100
                precision_recall['recall_edge'] = round((vehicle_recall + person_recall)/2, 2)*100
                
                # For edge, use processing time directly
                cost_metrics['cost_edge'] = round(self.cost_calculator.calculate_cost(
                    provider='edge',
                    processing_time=processing_time
                ), 2)
                    
        
        # Load results into postgres database
        self.db.load_data(table_name="latency_metrics", data=latency_metrics)
        self.db.load_data(table_name="processing_time_metrics", data=processing_time_metrics)
        self.db.load_data(table_name="fps_metrics", data=fps_metrics)
        self.db.load_data(table_name="count_vehicles", data=count_vehicles)
        self.db.load_data(table_name="count_people", data=count_people)
        self.db.load_data(table_name="precision_recall", data=precision_recall)
        self.db.load_data(table_name="cost_metrics", data=cost_metrics)

        return all_results

def main():
    """Main entry point for video processing pipeline."""
    parser = argparse.ArgumentParser(description='Video Processing Pipeline')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='./data/object_detection/images', help='Output directory for frames')
    parser.add_argument('--expected_vehicles', type=int, default=0, help='Expected number of vehicles in the video')
    parser.add_argument('--expected_people', type=int, default=0, help='Expected number of people in the video')
    parser.add_argument('--provider', type=str, default='edge', choices=['aws', 'azure', 'gcp', 'edge'],
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