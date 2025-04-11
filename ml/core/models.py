"""
Model loading and inference functionality.
Provides base classes and utilities for working with ML models.
"""

import torch
import numpy as np
import time
import logging
import os
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from abc import ABC, abstractmethod

# Import from our core package
from .tracking import Detection

logger = logging.getLogger(__name__)

class ObjectDetector(ABC):
    """Abstract base class for object detection models."""
    
    def __init__(self, name: str = "base_detector"):
        """
        Initialize the detector.
        
        Args:
            name: Detector name/identifier
        """
        self.name = name
        logger.info(f"Initialized {self.name} detector")
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        pass
    
    def process_image(self, image: np.ndarray, image_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process an image and return detections.
        
        Args:
            image: Image as numpy array
            image_path: Optional path to the image file
            
        Returns:
            List of detection dictionaries
        """
        start_time = time.time()
        
        # Run detection
        detections = self.detect(image)
        
        # Calculate metrics
        latency = time.time() - start_time
        
        # Log results
        if image_path:
            logger.debug(f"Processed {image_path} with {self.name}: "
                         f"{len(detections)} detections in {latency:.3f}s")
        else:
            logger.debug(f"Processed image with {self.name}: "
                         f"{len(detections)} detections in {latency:.3f}s")
        
        return detections
    
    def process_video(self, video_path: str, return_frames: bool = False) -> Dict[str, Any]:
        """
        Process a video file frame by frame.
        
        Args:
            video_path: Path to the video file
            return_frames: Whether to return processed frames
            
        Returns:
            Dictionary with detection results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize results
        detection_results = []
        processed_frames = [] if return_frames else None
        
        # Process frames
        frame_number = 0
        total_latency = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame_start_time = time.time()
                frame_detections = self.detect(frame)
                frame_latency = time.time() - frame_start_time
                total_latency += frame_latency
                
                # Store results
                for detection in frame_detections:
                    detection_results.append({
                        "frame_number": frame_number,
                        **detection
                    })
                
                # Store processed frame if requested
                if return_frames:
                    processed_frames.append(frame)
                
                frame_number += 1
                
                # Log progress periodically
                if frame_number % 10 == 0:
                    logger.debug(f"Processed {frame_number}/{frame_count} frames")
        
        finally:
            cap.release()
        
        # Calculate summary metrics
        avg_latency = total_latency / frame_number if frame_number > 0 else 0
        
        # Return results
        results = {
            "video_path": video_path,
            "detector": self.name,
            "total_frames": frame_number,
            "avg_latency": avg_latency,
            "total_detections": len(detection_results),
            "detections": detection_results,
            "video_info": {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height
            }
        }
        
        if return_frames:
            results["frames"] = processed_frames
        
        return results


class YOLODetector(ObjectDetector):
    """YOLO object detector implementation."""
    
    def __init__(self, model_path: str = None, device: str = None, 
                 confidence_threshold: float = 0.25, name: str = "yolo"):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to the YOLO model file (.pt)
            device: Device to run on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for detections
            name: Detector name
        """
        super().__init__(name=name)
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = self._load_model()
        
        logger.info(f"Loaded YOLO model from {model_path} on {self.device}")
    
    def _load_model(self) -> Any:
        """Load the YOLO model."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load custom model
                model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                      path=self.model_path, device=self.device)
            else:
                # Load standard YOLOv5s model
                logger.warning(f"Model path {self.model_path} not found, loading default YOLOv5s")
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
                                      pretrained=True, device=self.device)
            
            # Configure model
            model.conf = self.confidence_threshold  # Set confidence threshold
            model.eval()  # Set to evaluation mode
            return model
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image using YOLO.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        try:
            # Run inference
            results = self.model(image)
            
            # Convert results to our format
            detections = []
            for det in results.xyxy[0]:  # xyxy format
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                class_id = int(cls)
                class_name = results.names[class_id]
                
                detections.append({
                    "detection_type": class_name,
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "metadata": {
                        "class_id": class_id,
                        "frame_size": [image.shape[1], image.shape[0]]
                    }
                })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during YOLO detection: {str(e)}")
            return []


class AWSRekognitionDetector(ObjectDetector):
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
        super().__init__(name=name)
        self.rekognition_client = rekognition_client
        self.confidence_threshold = confidence_threshold
        
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
                            "detection_type": class_name,
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


class AzureVisionDetector(ObjectDetector):
    """Azure Computer Vision-based object detector."""
    
    def __init__(self, vision_client=None, confidence_threshold: float = 0.5, 
                 name: str = "azure_vision"):
        """
        Initialize Azure Vision detector.
        
        Args:
            vision_client: Azure Computer Vision client
            confidence_threshold: Minimum confidence for detections
            name: Detector name
        """
        super().__init__(name=name)
        self.vision_client = vision_client
        self.confidence_threshold = confidence_threshold
        
        # Will be lazily initialized if needed
        if self.vision_client is None:
            try:
                from azure.cognitiveservices.vision.computervision import ComputerVisionClient
                from msrest.authentication import CognitiveServicesCredentials
                
                azure_endpoint = os.getenv('AZURE_ENDPOINT')
                azure_key = os.getenv('AZURE_KEY')
                
                if not azure_endpoint or not azure_key:
                    raise ValueError("Azure credentials not found. "
                                    "Set AZURE_ENDPOINT and AZURE_KEY environment variables.")
                
                self.vision_client = ComputerVisionClient(
                    endpoint=azure_endpoint,
                    credentials=CognitiveServicesCredentials(azure_key)
                )
                logger.info("Initialized Azure Vision client")
            except Exception as e:
                logger.error(f"Failed to initialize Azure Vision client: {str(e)}")
                raise RuntimeError(f"Failed to initialize Azure Vision client: {str(e)}")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image using Azure Computer Vision.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        try:
            # Convert image to bytes
            _, img_bytes = cv2.imencode('.jpg', image)
            
            # Call Azure API
            response = self.vision_client.detect_objects_in_stream(img_bytes.tobytes())
            
            # Extract detections
            detections = []
            img_width, img_height = image.shape[1], image.shape[0]
            
            for obj in response.objects:
                confidence = obj.confidence
                if confidence < self.confidence_threshold:
                    continue
                
                # Get object info
                class_name = obj.object_property
                
                # Get bounding box
                rect = obj.rectangle
                x1 = rect.x
                y1 = rect.y
                x2 = rect.x + rect.w
                y2 = rect.y + rect.h
                
                # Add detection
                detections.append({
                    "detection_type": class_name,
                    "confidence": confidence,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "metadata": {
                        "frame_size": [img_width, img_height]
                    }
                })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during Azure Vision detection: {str(e)}")
            return []


# Factory function to create detector based on provider
def create_detector(provider: str, **kwargs) -> ObjectDetector:
    """
    Create an appropriate detector based on the provider.
    
    Args:
        provider: Provider name ('local', 'aws', 'azure')
        **kwargs: Additional configuration for the detector
    
    Returns:
        An ObjectDetector instance
    """
    if provider.lower() == 'local' or provider.lower() == 'yolo':
        return YOLODetector(**kwargs)
    
    elif provider.lower() == 'aws' or provider.lower() == 'rekognition':
        return AWSRekognitionDetector(**kwargs)
    
    elif provider.lower() == 'azure' or provider.lower() == 'vision':
        return AzureVisionDetector(**kwargs)
    
    else:
        logger.warning(f"Unknown provider '{provider}'. Using default YOLO detector.")
        return YOLODetector(**kwargs)
