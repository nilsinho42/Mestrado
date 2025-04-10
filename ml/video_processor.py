import os
import cv2
import numpy as np
import time
import uuid
import boto3
import requests
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime
import shutil
import tempfile
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base Detection class
class Detection:
    def __init__(self, 
                 box: List[float], 
                 class_id: int, 
                 class_name: str, 
                 confidence: float):
        """Initialize a detection object.
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2]
            class_id: Class ID
            class_name: Class name
            confidence: Detection confidence
        """
        self.box = box
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary."""
        return {
            'box': self.box,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence
        }

# Base detector class
class ObjectDetector(ABC):
    def __init__(self, name: str):
        """Initialize detector with name."""
        self.name = name
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[List[Detection], float]:
        """Detect objects in image and return detections with latency.
        
        Args:
            image: NumPy array of image
            
        Returns:
            Tuple of (detections, latency_ms)
        """
        pass
    
    @abstractmethod
    def count_objects(self, detections: List[Detection]) -> Dict[str, int]:
        """Count people and vehicles from detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            Dictionary with people_count and vehicles_count
        """
        pass

# Base video processor class
class VideoProcessor:
    def __init__(self, 
                 output_dir: str = "./data/object_detection/images",
                 sample_rate: int = 5):
        """Initialize video processor.
        
        Args:
            output_dir: Directory to save extracted images
            sample_rate: Sample 1 out of every N frames
        """
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dictionaries for results
        self.dict_A = {}  # Task A results
        self.dict_B = {}  # Task B results
    
    def extract_frames(self, 
                       video_path: str, 
                       save_frames: bool = True) -> Tuple[List[np.ndarray], List[str], Dict[str, Any]]:
        """Extract frames from video based on sample rate.
        
        Args:
            video_path: Path to video file
            save_frames: Whether to save extracted frames to disk
            
        Returns:
            Tuple of (frames, frame_paths, video_info)
        """
        # Create a unique video ID
        video_id = str(uuid.uuid4())
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Store video information
        video_info = {
            'video_id': video_id,
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames,
            'duration': total_frames / fps if fps > 0 else 0,
            'sample_rate': self.sample_rate
        }
        
        # Lists to store frames and their paths
        frames = []
        frame_paths = []
        
        # Extract frames at the specified sample rate
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames based on the sample rate
            if frame_idx % self.sample_rate == 0:
                # Create a unique image ID
                image_id = f"{video_id}_{frame_idx}"
                
                # Save frame if required
                if save_frames:
                    # Create frame directory if it doesn't exist
                    image_path = self.output_dir / f"{image_id}.jpg"
                    cv2.imwrite(str(image_path), frame)
                    frame_paths.append(str(image_path))
                else:
                    # If not saving, use a temporary ID for reference
                    frame_paths.append(image_id)
                
                # Add frame to list
                frames.append(frame)
            
            frame_idx += 1
        
        # Release video capture
        cap.release()
        
        # Update video info with extracted frames count
        video_info['extracted_frames'] = len(frames)
        
        return frames, frame_paths, video_info
    
    def process_video(self, 
                     video_path: str,
                     detectors: List[ObjectDetector],
                     save_frames: bool = True) -> Dict[str, Dict[str, Any]]:
        """Process video through Task A and Task B.
        
        Args:
            video_path: Path to video file
            detectors: List of detector objects
            save_frames: Whether to save extracted frames
            
        Returns:
            Dictionary containing both dict_A and dict_B results
        """
        # Extract frames from video
        frames, frame_paths, video_info = self.extract_frames(video_path, save_frames)
        
        logger.info(f"Extracted {len(frames)} frames from video with ID {video_info['video_id']}")
        
        # Process Task A: Image Analysis with Object Detection
        task_a_results = self._process_task_a(frames, frame_paths, detectors)
        
        # Process Task B: Video Processing with Object Tracking
        task_b_results = self._process_task_b(video_path, video_info, detectors)
        
        # Store results in instance dictionaries
        self.dict_A = task_a_results
        self.dict_B = task_b_results
        
        return {
            'task_a': task_a_results,
            'task_b': task_b_results,
            'video_info': video_info
        }
    
    def _process_task_a(self, 
                       frames: List[np.ndarray], 
                       frame_paths: List[str], 
                       detectors: List[ObjectDetector]) -> Dict[str, Dict[str, Any]]:
        """Process Task A: Image Analysis with Object Detection.
        
        Args:
            frames: List of extracted frames
            frame_paths: List of frame paths
            detectors: List of detector objects
            
        Returns:
            Task A results dictionary
        """
        results = {}
        
        # Process each frame with each detector
        for detector in detectors:
            detector_name = detector.name
            results[detector_name] = {
                'images': [],
                'total_latency': 0,
                'avg_latency': 0,
                'total_people': 0,
                'total_vehicles': 0
            }
            
            for i, (frame, frame_path) in enumerate(zip(frames, frame_paths)):
                # Get image ID from path
                image_id = Path(frame_path).stem if isinstance(frame_path, str) and os.path.exists(frame_path) else frame_path
                
                # Run object detection
                detections, latency = detector.detect(frame)
                
                # Count objects
                counts = detector.count_objects(detections)
                
                # Store results for this image
                image_result = {
                    'image_id': image_id,
                    'latency': latency,
                    'people_count': counts['people_count'],
                    'vehicles_count': counts['vehicles_count'],
                    'detections': [d.to_dict() for d in detections]
                }
                
                # Add to detector results
                results[detector_name]['images'].append(image_result)
                results[detector_name]['total_latency'] += latency
                results[detector_name]['total_people'] += counts['people_count']
                results[detector_name]['total_vehicles'] += counts['vehicles_count']
            
            # Calculate averages
            num_images = len(frames)
            if num_images > 0:
                results[detector_name]['avg_latency'] = results[detector_name]['total_latency'] / num_images
                results[detector_name]['avg_people_per_image'] = results[detector_name]['total_people'] / num_images
                results[detector_name]['avg_vehicles_per_image'] = results[detector_name]['total_vehicles'] / num_images
        
        return results
    
    def _process_task_b(self, 
                      video_path: str, 
                      video_info: Dict[str, Any], 
                      detectors: List[ObjectDetector]) -> Dict[str, Dict[str, Any]]:
        """Process Task B: Video Processing with Object Tracking.
        
        Args:
            video_path: Path to video file
            video_info: Video information dictionary
            detectors: List of detector objects
            
        Returns:
            Task B results dictionary
        """
        # This is a placeholder - actual tracking implementation 
        # will depend on specific tracking modules for each provider
        results = {}
        
        for detector in detectors:
            detector_name = detector.name
            results[detector_name] = {
                'video_id': video_info['video_id'],
                'processing_time': 0,
                'people_tracked': 0,
                'vehicles_tracked': 0,
                'tracking_data': {}
            }
        
        return results 