"""
Model loading and inference functionality.
Provides base classes and utilities for working with ML models.
"""

import numpy as np
import logging
import os
import cv2
from typing import List, Dict, Any
from azure.ai.vision.imageanalysis.models import VisualFeatures
import requests
import base64
import time
logging.getLogger("azure").setLevel(logging.WARNING)

# Import from our core package
logger = logging.getLogger(__name__)

# class ObjectDetector(ABC):
#     """Abstract base class for object detection models."""
    
#     def __init__(self, name: str = "base_detector"):
#         """
#         Initialize the detector.
        
#         Args:
#             name: Detector name/identifier
#         """
#         self.name = name
#         logger.info(f"Initialized {self.name} detector")
    
#     @abstractmethod
#     def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
#         """
#         Detect objects in an image.
        
#         Args:
#             image: Image as numpy array
            
#         Returns:
#             List of detection dictionaries
#         """
#         pass
    
#     def process_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
#         """
#         Process an image and return detections.
        
#         Args:
#             image: Image as numpy array
#             image_path: Optional path to the image file
            
#         Returns:
#             List of detection dictionaries
#         """
#         # Run detection
#         detections = self.detect(image)
        
#         # Log results
#         logger.debug(f"Processed image with {self.name}: " f"{len(detections)} detections.")
        
#         return detections
    
    # TODO REVIEW
    # def process_video(self, video_path: str, return_frames: bool = False) -> Dict[str, Any]:
    #     """
    #     Process a video file frame by frame.
        
    #     Args:
    #         video_path: Path to the video file
    #         return_frames: Whether to return processed frames
            
    #     Returns:
    #         Dictionary with detection results
    #     """
    #     cap = cv2.VideoCapture(video_path)
    #     if not cap.isOpened():
    #         raise ValueError(f"Could not open video file: {video_path}")
        
    #     # Get video properties
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    #     # Initialize results
    #     detection_results = []
    #     processed_frames = [] if return_frames else None
        
    #     # Process frames
    #     frame_number = 0
    #     total_latency = 0
        
    #     try:
    #         while cap.isOpened():
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
                
    #             # Process frame
    #             frame_start_time = time.time()
    #             frame_detections = self.detect(frame)
    #             frame_latency = time.time() - frame_start_time
    #             total_latency += frame_latency
                
    #             # Store results
    #             for detection in frame_detections:
    #                 detection_results.append({
    #                     "frame_number": frame_number,
    #                     **detection
    #                 })
                
    #             # Store processed frame if requested
    #             if return_frames:
    #                 processed_frames.append(frame)
                
    #             frame_number += 1
                
    #             # Log progress periodically
    #             if frame_number % 10 == 0:
    #                 logger.debug(f"Processed {frame_number}/{frame_count} frames")
        
    #     finally:
    #         cap.release()
        
    #     # Calculate summary metrics
    #     avg_latency = total_latency / frame_number if frame_number > 0 else 0
        
    #     # Return results
    #     results = {
    #         "video_path": video_path,
    #         "detector": self.name,
    #         "total_frames": frame_number,
    #         "avg_latency": avg_latency,
    #         "total_detections": len(detection_results),
    #         "detections": detection_results,
    #         "video_info": {
    #             "fps": fps,
    #             "frame_count": frame_count,
    #             "width": width,
    #             "height": height
    #         }
    #     }
        
    #     if return_frames:
    #         results["frames"] = processed_frames
        
    #     return results


class YOLODetector():
    """YOLO object detector implementation using Ultralytics API."""
    
    def __init__(self, model_path: str = "yolo11n.pt", confidence_threshold: float = 0.25, name: str = "yolo"):
        """
        Initialize YOLO detector with simplified approach.
        
        Args:
            model_path: Path to the YOLO model file (.pt)
            confidence_threshold: Minimum confidence for detections
            name: Detector name
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.people_classes = ['person', 'human', 'people', 'pedestrian', 'man', 'woman', 'child', 'baby']
        self.vehicle_classes = ['car', 'vehicle', 'automobile', 'truck', 'van', 'bus', 'motorcycle', 'transportation', 'taxi', 'ambulance', 'police car']
        
        # Load model directly using Ultralytics API
        from ultralytics import YOLO
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Loaded YOLO model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image using YOLO model.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        try:
            # Log image information
            logger.debug(f"Processing image with shape {image.shape} using YOLO model")
            
            # Run inference with confidence threshold
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            detections = []
            
            # Process the results directly
            if len(results) > 0:
                result = results[0]  # Get the first result
                # Get frame dimensions for metadata
                img_height, img_width = image.shape[:2]
                
                # Check if any boxes were detected
                if len(result.boxes) > 0:
                    logger.debug(f"YOLO found {len(result.boxes)} objects in image")
                else:
                    logger.debug("YOLO found no objects in image")
                
                # Process each detection
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls]

                    if class_name not in self.people_classes and class_name not in self.vehicle_classes:
                        continue
                    
                    logger.debug(f"Detected {class_name} with confidence {conf:.2f}")
                    
                    # Create detection with standardized format
                    detection = {
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "metadata": {
                            "class_id": cls,
                            "frame_size": [img_width, img_height]
                        }
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []


class AWSRekognitionDetector():
    """AWS Rekognition-based object detector."""
    
    def __init__(self, rekognition_client=None, confidence_threshold: float = 0.5, 
                 name: str = "aws_rekognition"):
        """
        Initialize AWS Rekognition detector.
        
        Args:
            rekognition_client: Boto3 Rekognition client
            confidence_threshold: Minimum confidence for detections
            name: Detector name
        """
        self.rekognition_client = rekognition_client
        self.confidence_threshold = confidence_threshold
        self.people_classes = ['Person', 'Human', 'People', 'Pedestrian', 'Man', 'Woman', 'Child', 'Baby']
        self.vehicle_classes = ['Car', 'Vehicle', 'Automobile', 'Truck', 'Van', 'Bus', 'Motorcycle', 'Transportation', 'Taxi', 'Ambulance', 'Police Car']
        
        # Will be lazily initialized if needed
        if self.rekognition_client is None:
            try:
                import boto3
                self.rekognition_client = boto3.client('rekognition')
                logger.info("Initialized AWS Rekognition client")
            except Exception as e:
                logger.error(f"Failed to initialize AWS Rekognition client: {str(e)}")
                raise RuntimeError(f"Failed to initialize AWS Rekognition client: {str(e)}")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image using AWS Rekognition.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        try:
            # Convert image to bytes
            _, img_bytes = cv2.imencode('.jpg', image)
            
            # Call Rekognition API
            response = self.rekognition_client.detect_labels(
                Image={'Bytes': img_bytes.tobytes()},
                MinConfidence=self.confidence_threshold * 100  # AWS uses percentage
            )
            
            # Extract detections
            detections = []
            for label in response['Labels']:
                # Get label info
                class_name = label['Name']
                confidence = label['Confidence'] / 100.0  # Convert to 0-1 scale
                
                # Filter for people and vehicles
                if class_name not in self.people_classes and class_name not in self.vehicle_classes:
                    continue
                
                # Get bounding boxes if available
                for instance in label.get('Instances', []):
                    if 'BoundingBox' in instance:
                        bbox = instance['BoundingBox']
                        
                        # AWS returns normalized coordinates
                        img_width, img_height = image.shape[1], image.shape[0]
                        x1 = bbox['Left'] * img_width
                        y1 = bbox['Top'] * img_height
                        width = bbox['Width'] * img_width
                        height = bbox['Height'] * img_height
                        x2 = x1 + width
                        y2 = y1 + height
                        
                        # Add detection
                        detections.append({
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "metadata": {
                                "parent_labels": [p['Name'] for p in label.get('Parents', [])],
                                "frame_size": [img_width, img_height]
                            }
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during AWS Rekognition detection: {str(e)}")
            return []


class AzureVisionDetector():
    """Azure Computer Vision 4.0-based object detector."""
    
    def __init__(self, vision_client=None, confidence_threshold: float = 0.5, 
                 name: str = "azure_vision"):
        """
        Initialize Azure Vision detector using Computer Vision 4.0 API.
        
        Args:
            vision_client: Azure Computer Vision client
            confidence_threshold: Minimum confidence for detections
            name: Detector name
        """
        self.vision_client = vision_client
        self.confidence_threshold = confidence_threshold
        self.people_classes = ['person', 'people', 'man', 'woman', 'child']
        self.vehicle_classes = ['car', 'vehicle', 'truck', 'van', 'bus', 'motorcycle', 'bicycle']


        # Will be lazily initialized if needed
        if self.vision_client is None:
            try:
                from azure.ai.vision.imageanalysis import ImageAnalysisClient
                from azure.core.credentials import AzureKeyCredential
                
                azure_endpoint = os.getenv('AZURE_ENDPOINT')
                azure_key = os.getenv('AZURE_KEY')
                
                if not azure_endpoint or not azure_key:
                    raise ValueError("Azure credentials not found. "
                                    "Set AZURE_ENDPOINT and AZURE_KEY environment variables.")
                
                self.vision_client = ImageAnalysisClient(
                    endpoint=azure_endpoint,
                    credential=AzureKeyCredential(azure_key)
                )

                logger.info("Initialized Azure Vision 4.0 client")
            except Exception as e:
                logger.error(f"Failed to initialize Azure Vision 4.0 client: {str(e)}")
                raise RuntimeError(f"Failed to initialize Azure Vision 4.0 client: {str(e)}")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image using Azure Computer Vision 4.0.
        Handles rate limiting with controlled retry logic.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        import time
        from azure.core.exceptions import HttpResponseError
        
        max_retries = 1
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # Convert image to bytes
                _, img_bytes = cv2.imencode('.jpg', image)
                
                # Create a binary data object
                import io
                image_stream = io.BytesIO(img_bytes.tobytes())
                
                # Call Azure API - request object detection only
                response = self.vision_client.analyze(
                    image_data=image_stream,
                    visual_features=[VisualFeatures.OBJECTS, VisualFeatures.PEOPLE],
                    language="en",
                    logging_enable=False
                )

                # Extract detections
                detections = []
                img_width, img_height = image.shape[1], image.shape[0]
                
                # Process object detections
                if hasattr(response, 'objects'):
                    for obj in response.objects.list:
                        # Get object info - take first tag if multiple exist
                        if obj.tags and len(obj.tags) > 0:
                            class_name = obj.tags[0].name.lower()
                            confidence = obj.tags[0].confidence
                        else:
                            continue
                        
                        # Skip low confidence detections
                        if confidence < self.confidence_threshold:
                            continue
                            
                        # Filter for people and vehicles
                        if class_name not in self.people_classes and class_name not in self.vehicle_classes:
                            continue
                        
                        # Get bounding box
                        bbox = obj.bounding_box
                        
                        # Add detection
                        detections.append({
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": bbox,
                            "metadata": {
                                "frame_size": [img_width, img_height]
                            }
                        })
                
                # Process people detections if available
                if hasattr(response, 'people'):
                    for person in response.people.list:
                        # Skip low confidence detections
                        if person.confidence < self.confidence_threshold:
                            continue
                        
                        # Get bounding box
                        bbox = person.bounding_box
                        
                        # Add detection
                        detections.append({
                            "class_name": "person",
                            "confidence": person.confidence,
                            "bbox": bbox,
                            "metadata": {
                                "frame_size": [img_width, img_height]
                            }
                        })
                
                return detections
                
            except HttpResponseError as e:
                # Check if this is a rate limit error (HTTP 429)
                if hasattr(e, 'status_code') and e.status_code == 429:
                    # Extract retry-after value from response headers (default to 60 seconds)
                    retry_after = 60
                    if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                        retry_after_header = e.response.headers.get('Retry-After')
                        if retry_after_header and retry_after_header.isdigit():
                            retry_after = int(retry_after_header)
                    
                    if retry_count < max_retries:
                        logger.info(f"Azure rate limit reached (429). Waiting {retry_after} seconds before retry.")
                        # Wait before retrying
                        time.sleep(retry_after)
                        retry_count += 1
                    else:
                        logger.info(f"Azure rate limit reached (429). Max retries exhausted. Skipping frame.")
                        return []
                else:
                    # Log other HTTP errors and return empty list
                    logger.info(f"HTTP error during Azure Vision detection: {str(e)}")
                    return []
            except Exception as e:
                # Log any other errors and return empty list
                logger.info(f"Error during Azure Vision detection: {str(e)}")
                return []


class GCPVisionDetector():
    """Google Cloud Vision API-based object detector."""
    
    def __init__(self, vision_client=None, confidence_threshold: float = 0.5, 
                 name: str = "gcp_vision"):
        """
        Initialize Google Cloud Vision detector.
        
        Args:
            vision_client: Google Cloud Vision client
            confidence_threshold: Minimum confidence for detections
            name: Detector name
        """
        self.vision_client = vision_client
        self.confidence_threshold = confidence_threshold
        self.people_classes = ['person', 'people', 'man', 'woman', 'child']
        self.vehicle_classes = ['car', 'vehicle', 'truck', 'van', 'bus', 'motorcycle', 'bicycle']

        # Will be lazily initialized if needed
        if self.vision_client is None:
            try:
                import os
                from google.cloud import vision
                
                # Check if credentials file exists
                credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                if not credentials_path or not os.path.exists(credentials_path):
                    raise ValueError("Google Cloud credentials not found. "
                                     "Set GOOGLE_APPLICATION_CREDENTIALS environment variable.")
                
                # Initialize the client
                self.vision_client = vision.ImageAnnotatorClient()
                logger.info("Initialized Google Cloud Vision client")
            except Exception as e:
                logger.error(f"Failed to initialize Google Cloud Vision client: {str(e)}")
                raise RuntimeError(f"Failed to initialize Google Cloud Vision client: {str(e)}")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image using Google Cloud Vision API.
        Handles rate limiting with controlled retry logic.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        import time
        import io
        
        max_retries = 1
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                from google.cloud import vision
                
                # Convert image to bytes
                _, img_bytes = cv2.imencode('.jpg', image)
                
                # Create Google Cloud Vision image
                vision_image = vision.Image(content=img_bytes.tobytes())
                
                # Perform object localization
                response = self.vision_client.object_localization(image=vision_image)
                
                # Extract detections
                detections = []
                img_width, img_height = image.shape[1], image.shape[0]
                
                # Check if the response has localized_object_annotations
                if response.localized_object_annotations:
                    for object_annotation in response.localized_object_annotations:
                        # Get object info
                        class_name = object_annotation.name.lower()
                        confidence = object_annotation.score
                        
                        # Skip low confidence detections
                        if confidence < self.confidence_threshold:
                            continue
                            
                        # Filter for people and vehicles
                        if class_name not in self.people_classes and class_name not in self.vehicle_classes:
                            continue
                        
                        # GCP returns normalized vertices
                        vertices = object_annotation.bounding_poly.normalized_vertices
                        
                        # Find min/max coordinates to create a box format [x1, y1, x2, y2]
                        x_coords = [vertex.x for vertex in vertices]
                        y_coords = [vertex.y for vertex in vertices]
                        
                        x1 = min(x_coords) * img_width
                        y1 = min(y_coords) * img_height
                        x2 = max(x_coords) * img_width
                        y2 = max(y_coords) * img_height
                        
                        # Add detection
                        detections.append({
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "metadata": {
                                "frame_size": [img_width, img_height]
                            }
                        })
                
                return detections
                
            except Exception as e:
                if "429" in str(e) or "RateLimit" in str(e):
                    # Default retry after 2 seconds for GCP rate limiting
                    retry_after = 2
                    
                    if retry_count < max_retries:
                        logger.info(f"GCP rate limit reached. Waiting {retry_after} seconds before retry.")
                        # Wait before retrying
                        time.sleep(retry_after)
                        retry_count += 1
                    else:
                        logger.info(f"GCP rate limit reached. Max retries exhausted. Skipping frame.")
                        return []
                else:
                    # Log other errors and return empty list
                    logger.error(f"Error during Google Cloud Vision detection: {str(e)}")
                    return []


class EdgeDetector():
    """Raspberry Pi edge server-based object detector."""
    
    def __init__(self, edge_endpoint: str = None, confidence_threshold: float = 0.5, 
                 name: str = "edge_detector", verify_connection: bool = True):
        """
        Initialize Edge detector that makes requests to a Raspberry Pi server.
        
        Args:
            edge_endpoint: URL endpoint for the edge server API
            confidence_threshold: Minimum confidence for detections
            name: Detector name
            verify_connection: Whether to verify connection to the edge server on initialization
        """
        # Get the endpoint from environment variable if not provided
        self.edge_endpoint = edge_endpoint or os.getenv("EDGE_DEEPSORT_ENDPOINT")
        
        if not self.edge_endpoint:
            raise ValueError("Edge endpoint URL not provided. Set EDGE_DEEPSORT_ENDPOINT environment variable.")
        
        self.confidence_threshold = confidence_threshold
        self.name = name
        self.api_endpoint = f"{self.edge_endpoint.rstrip('/')}/api/detect"
        self.max_retries = 3  # Add retry mechanism
        
        # Verify connectivity to edge server if requested
        if verify_connection:
            try:
                response = requests.get(f"{self.edge_endpoint.rstrip('/')}/healthcheck", timeout=3)
                if response.status_code == 200:
                    logger.info(f"Successfully connected to edge server at {self.edge_endpoint}")
                else:
                    logger.warning(f"Edge server responded with status {response.status_code}")
            except requests.exceptions.ConnectTimeout:
                logger.warning(f"Connection to edge server at {self.edge_endpoint} timed out")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection to edge server at {self.edge_endpoint} failed")
            except Exception as e:
                logger.warning(f"Failed to connect to edge server: {str(e)}")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image by making a request to the edge server.
        Implements retry mechanism for intermittent connection issues.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        # Convert image to base64
        _, img_bytes = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(img_bytes).decode('utf-8')
        
        # Create payload
        payload = {
            "image": encoded_image
        }
        
        # Implement retry logic
        for retry in range(self.max_retries):
            try:
                # Make request to edge server
                if retry > 0:
                    logger.info(f"Retry {retry}/{self.max_retries} for edge detection request")
                    
                logger.info(f"Sending detection request to edge server: {self.api_endpoint}")
                start_time = time.time()
                response = requests.post(self.api_endpoint, json=payload, timeout=10)
                latency = time.time() - start_time
                
                # Check response
                if response.status_code != 200:
                    logger.error(f"Edge server returned status {response.status_code}: {response.text}")
                    # Wait before retrying
                    time.sleep(0.5)
                    continue
                
                # Parse response
                resp_data = response.json()
                detections = resp_data.get("detections", [])
                
                # Filter by confidence threshold
                filtered_detections = [
                    d for d in detections 
                    if d.get("confidence", 0) >= self.confidence_threshold
                ]
                
                logger.info(f"Edge detection completed in {latency:.3f}s. Found {len(filtered_detections)} objects.")
                return filtered_detections
                
            except requests.exceptions.ConnectTimeout:
                logger.error(f"Connection to edge server at {self.api_endpoint} timed out (attempt {retry+1}/{self.max_retries})")
                # Wait before retrying
                time.sleep(0.5)  
            except requests.exceptions.ConnectionError:
                logger.error(f"Connection to edge server at {self.api_endpoint} failed (attempt {retry+1}/{self.max_retries})")
                # Wait before retrying
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error during edge detection: {str(e)} (attempt {retry+1}/{self.max_retries})")
                # Wait before retrying
                time.sleep(0.5)
                
        # All retries failed
        logger.error(f"All {self.max_retries} attempts to connect to edge server failed")
        return []


# Factory function to create detector based on provider
def create_detector(provider: str, **kwargs):
    """
    Create an appropriate detector based on the provider.
    
    Args:
        provider: Provider name ('local', 'aws', 'azure', 'gcp', 'edge')
        **kwargs: Additional configuration for the detector
    
    Returns:
        An ObjectDetector instance
    """
    if provider.lower() in ['local']:
        return YOLODetector(**kwargs)
    
    elif provider.lower() in ['aws']:
        return AWSRekognitionDetector(**kwargs)
        
    elif provider.lower() in ['azure']:
        return AzureVisionDetector(**kwargs)
        
    elif provider.lower() in ['gcp']:
        return GCPVisionDetector(**kwargs)
        
    elif provider.lower() in ['edge']:
        return EdgeDetector(**kwargs)