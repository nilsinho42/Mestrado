import os
import time
import numpy as np
import cv2
import boto3
import requests
import json
from typing import Dict, Any, List, Tuple, Optional, Union
from io import BytesIO
from ultralytics import YOLO
from pathlib import Path
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv
import logging

# Import base classes from video_processor
from video_processor import ObjectDetector, Detection

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLODetector(ObjectDetector):
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights
        """
        super().__init__(name="yolo")
        self.model = YOLO(model_path)
        
        # Define class mappings for YOLO (COCO dataset)
        self.class_map = {
            0: 'person',           # person
            2: 'car',              # car
            3: 'motorcycle',       # motorcycle
            5: 'bus',              # bus
            7: 'truck'             # truck
        }
        
        # Define people and vehicle classes
        self.people_classes = [0]
        self.vehicle_classes = [2, 3, 5, 7]
    
    def detect(self, image: np.ndarray) -> Tuple[List[Detection], float]:
        """Detect objects using YOLO model.
        
        Args:
            image: NumPy array of image
            
        Returns:
            Tuple of (detections, latency_ms)
        """
        # Measure detection time
        start_time = time.time()
        
        # Run inference
        results = self.model(image, verbose=False)
        
        # Calculate latency in milliseconds
        latency = (time.time() - start_time) * 1000
        
        # Convert results to Detection objects
        detections = []
        
        if len(results) > 0:
            # Get detection boxes, classes and scores from results
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Create Detection objects
            for box, class_id, confidence in zip(boxes, class_ids, confidences):
                class_id_int = int(class_id)
                
                # Only include classes we're interested in
                if class_id_int in self.class_map:
                    class_name = self.class_map[class_id_int]
                    
                    detections.append(Detection(
                        box=box.tolist(),
                        class_id=class_id_int,
                        class_name=class_name,
                        confidence=float(confidence)
                    ))
        
        return detections, latency
    
    def count_objects(self, detections: List[Detection]) -> Dict[str, int]:
        """Count people and vehicles from detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            Dictionary with people_count and vehicles_count
        """
        people_count = sum(1 for d in detections if d.class_id in self.people_classes)
        vehicles_count = sum(1 for d in detections if d.class_id in self.vehicle_classes)
        
        return {
            'people_count': people_count,
            'vehicles_count': vehicles_count
        }

class AWSDetector(ObjectDetector):
    def __init__(self):
        """Initialize AWS Rekognition detector."""
        super().__init__(name="aws")
        
        # Initialize AWS Rekognition client
        self.rekognition = boto3.client('rekognition')
        
        # Define people and vehicle classes for AWS
        self.people_classes = ['Person', 'Human', 'People', 'Pedestrian', 
                             'Man', 'Woman', 'Child', 'Baby']
        self.vehicle_classes = ['Car', 'Vehicle', 'Automobile', 'Truck', 
                              'Van', 'Bus', 'Motorcycle', 'Transportation', 
                              'Taxi', 'Ambulance', 'Police Car']
    
    def detect(self, image: np.ndarray) -> Tuple[List[Detection], float]:
        """Detect objects using AWS Rekognition.
        
        Args:
            image: NumPy array of image
            
        Returns:
            Tuple of (detections, latency_ms)
        """
        # Convert image to bytes for AWS API
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()
        
        # Measure detection time
        start_time = time.time()
        
        # Call AWS Rekognition
        response = self.rekognition.detect_labels(
            Image={'Bytes': img_bytes},
            MaxLabels=20,
            MinConfidence=35  # Minimum confidence percent
        )
        
        # Calculate latency in milliseconds
        latency = (time.time() - start_time) * 1000
        
        # Convert results to Detection objects
        detections = []
        
        for label in response.get('Labels', []):
            class_name = label['Name']
            confidence = label['Confidence'] / 100.0  # Convert percentage to decimal
            
            # Process bounding boxes if they exist
            for instance in label.get('Instances', []):
                if 'BoundingBox' in instance:
                    box = instance['BoundingBox']
                    # AWS returns normalized coordinates
                    h, w = image.shape[:2]
                    x1 = box['Left'] * w
                    y1 = box['Top'] * h
                    x2 = x1 + (box['Width'] * w)
                    y2 = y1 + (box['Height'] * h)
                    
                    # Use a placeholder class_id since AWS doesn't provide them
                    class_id = 0 if class_name in self.people_classes else 1
                    
                    detections.append(Detection(
                        box=[x1, y1, x2, y2],
                        class_id=class_id,
                        class_name=class_name,
                        confidence=instance.get('Confidence', confidence) / 100.0
                    ))
            
            # If there are no instances but this is a detected label, 
            # we still record it for counting (without bounding box)
            if len(label.get('Instances', [])) == 0 and (class_name in self.people_classes or class_name in self.vehicle_classes):
                class_id = 0 if class_name in self.people_classes else 1
                
                # Create a detection with full image as bounding box
                h, w = image.shape[:2]
                detections.append(Detection(
                    box=[0, 0, w, h],
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence
                ))
        
        return detections, latency
    
    def count_objects(self, detections: List[Detection]) -> Dict[str, int]:
        """Count people and vehicles from detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            Dictionary with people_count and vehicles_count
        """
        # Count unique detections - AWS might return multiple detections for the same object
        people = set()
        vehicles = set()
        
        for d in detections:
            if d.class_name in self.people_classes:
                # Use bounding box coordinates as unique identifier
                people.add(tuple(d.box))
            elif d.class_name in self.vehicle_classes:
                vehicles.add(tuple(d.box))
        
        return {
            'people_count': len(people),
            'vehicles_count': len(vehicles)
        }

class AzureDetector(ObjectDetector):
    def __init__(self):
        """Initialize Azure Computer Vision detector."""
        super().__init__(name="azure")
        
        # Initialize Azure Computer Vision client
        self.client = ComputerVisionClient(
            endpoint=os.getenv('AZURE_ENDPOINT'),
            credentials=CognitiveServicesCredentials(os.getenv('AZURE_KEY'))
        )
        
        # Define people and vehicle classes for Azure
        self.people_classes = ['person', 'people', 'man', 'woman', 'child']
        self.vehicle_classes = ['car', 'vehicle', 'truck', 'van', 'bus', 'motorcycle', 'bicycle']
        
        # Track last API call for rate limiting
        self.last_call_time = 0
        self.min_call_interval = 0.5  # seconds
    
    def detect(self, image: np.ndarray) -> Tuple[List[Detection], float]:
        """Detect objects using Azure Computer Vision.
        
        Args:
            image: NumPy array of image
            
        Returns:
            Tuple of (detections, latency_ms)
        """
        # Implement rate limiting
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.min_call_interval:
            time.sleep(self.min_call_interval - time_since_last_call)
        
        # Convert image to bytes for Azure API
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = BytesIO(img_encoded.tobytes())
        
        # Measure detection time
        start_time = time.time()
        
        # Call Azure Computer Vision
        try:
            response = self.client.analyze_image_in_stream(
                image=img_bytes,
                visual_features=['Objects', 'Tags'],
                details=[]
            )
        except Exception as e:
            logger.error(f"Error calling Azure API: {e}")
            return [], (time.time() - start_time) * 1000
        
        # Update last call time
        self.last_call_time = time.time()
        
        # Calculate latency in milliseconds
        latency = (time.time() - start_time) * 1000
        
        # Convert results to Detection objects
        detections = []
        
        # Process objects with bounding boxes
        if hasattr(response, 'objects') and response.objects:
            for obj in response.objects:
                class_name = obj.object_property.lower()
                confidence = obj.confidence
                
                # Azure returns normalized coordinates
                h, w = image.shape[:2]
                rect = obj.rectangle
                x1 = rect.x
                y1 = rect.y
                x2 = x1 + rect.w
                y2 = y1 + rect.h
                
                # Use a placeholder class_id since Azure doesn't provide them
                class_id = 0 if class_name in self.people_classes else 1
                
                detections.append(Detection(
                    box=[x1, y1, x2, y2],
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence
                ))
        
        # Process tags without bounding boxes
        if hasattr(response, 'tags') and response.tags:
            for tag in response.tags:
                class_name = tag.name.lower()
                confidence = tag.confidence
                
                # Only add detection for people/vehicles if not already detected as objects
                if (class_name in self.people_classes or class_name in self.vehicle_classes) and \
                   not any(d.class_name == class_name for d in detections):
                    class_id = 0 if class_name in self.people_classes else 1
                    
                    # Create a detection with full image as bounding box
                    h, w = image.shape[:2]
                    detections.append(Detection(
                        box=[0, 0, w, h],
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence
                    ))
        
        return detections, latency
    
    def count_objects(self, detections: List[Detection]) -> Dict[str, int]:
        """Count people and vehicles from detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            Dictionary with people_count and vehicles_count
        """
        people_count = sum(1 for d in detections if d.class_name in self.people_classes)
        vehicles_count = sum(1 for d in detections if d.class_name in self.vehicle_classes)
        
        return {
            'people_count': people_count,
            'vehicles_count': vehicles_count
        } 