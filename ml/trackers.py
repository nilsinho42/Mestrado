import os
import cv2
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from pathlib import Path
import uuid
from abc import ABC, abstractmethod
import requests
import json
import base64
import aiohttp
import asyncio

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import base classes from video_processor
from video_processor import Detection

# A simplified tracked object class
class TrackedObject:
    def __init__(self, object_id: int, class_id: int, class_name: str):
        """Initialize tracked object.
        
        Args:
            object_id: Unique ID for the tracked object
            class_id: Class ID
            class_name: Class name
        """
        self.object_id = object_id
        self.class_id = class_id
        self.class_name = class_name
        self.frames = []
        self.boxes = []
        self.confidences = []
        self.last_seen = 0
    
    def update(self, frame_idx: int, box: List[float], confidence: float):
        """Update tracked object with new detection.
        
        Args:
            frame_idx: Frame index
            box: Bounding box coordinates [x1, y1, x2, y2]
            confidence: Detection confidence
        """
        self.frames.append(frame_idx)
        self.boxes.append(box)
        self.confidences.append(confidence)
        self.last_seen = frame_idx
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tracked object to dictionary."""
        return {
            'object_id': self.object_id,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'frames': self.frames,
            'boxes': self.boxes,
            'confidences': self.confidences
        }

# Base tracker class
class ObjectTracker(ABC):
    def __init__(self, name: str):
        """Initialize tracker with name."""
        self.name = name
        self.tracks = {}  # Dictionary of object_id -> TrackedObject
        self.next_id = 1
        self.current_frame = 0
        
        # Define people and vehicle classes
        self.people_classes = []
        self.vehicle_classes = []
    
    @abstractmethod
    def update(self, detections: List[Detection], frame_idx: int) -> List[TrackedObject]:
        """Update tracker with new detections.
        
        Args:
            detections: List of Detection objects
            frame_idx: Current frame index
            
        Returns:
            List of active TrackedObject instances
        """
        pass
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = {}
        self.next_id = 1
        self.current_frame = 0
    
    def get_active_tracks(self) -> List[TrackedObject]:
        """Get list of currently active tracked objects."""
        return list(self.tracks.values())
    
    def get_counts(self) -> Dict[str, int]:
        """Count unique people and vehicles that have been tracked."""
        people_count = sum(1 for obj in self.tracks.values() if obj.class_name in self.people_classes)
        vehicle_count = sum(1 for obj in self.tracks.values() if obj.class_name in self.vehicle_classes)
        
        return {
            'people_tracked': people_count,
            'vehicles_tracked': vehicle_count,
            'total_tracked': len(self.tracks)
        }
    
    def get_results(self) -> Dict[str, Any]:
        """Get tracking results in dictionary format."""
        return {
            'tracker': self.name,
            'tracks': [track.to_dict() for track in self.tracks.values()],
            'counts': self.get_counts(),
            'total_frames': self.current_frame
        }

# Simple IoU-based tracker
class IoUTracker(ObjectTracker):
    def __init__(self, 
                iou_threshold: float = 0.3, 
                max_age: int = 5,
                name: str = "iou"):
        """Initialize IoU tracker.
        
        Args:
            iou_threshold: IoU threshold for matching
            max_age: Maximum frames to keep track without update
            name: Tracker name
        """
        super().__init__(name=name)
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        
        # Define classes based on common format
        self.people_classes = ['person', 'Person', 'people', 'People', 'pedestrian', 'Pedestrian']
        self.vehicle_classes = [
            'car', 'Car', 'vehicle', 'Vehicle', 'truck', 'Truck', 
            'bus', 'Bus', 'motorcycle', 'Motorcycle', 'bicycle', 'Bicycle'
        ]
    
    def update(self, detections: List[Detection], frame_idx: int) -> List[TrackedObject]:
        """Update tracker with new detections.
        
        Args:
            detections: List of Detection objects
            frame_idx: Current frame index
            
        Returns:
            List of active TrackedObject instances
        """
        self.current_frame = frame_idx
        
        # Convert detections to format for matching
        detection_boxes = np.array([d.box for d in detections]) if detections else np.empty((0, 4))
        
        # Get currently tracked boxes
        tracked_ids = list(self.tracks.keys())
        tracked_boxes = np.array([self.tracks[tid].boxes[-1] for tid in tracked_ids]) if tracked_ids else np.empty((0, 4))
        
        # Match detections to existing tracks
        if len(tracked_boxes) > 0 and len(detection_boxes) > 0:
            # Calculate IoU between all pairs
            iou_matrix = self._calculate_iou_matrix(tracked_boxes, detection_boxes)
            
            # Match based on IoU
            matches, unmatched_tracks, unmatched_detections = self._match_detections(
                iou_matrix, 
                len(tracked_ids), 
                len(detections)
            )
            
            # Update matched tracks
            for track_idx, det_idx in matches:
                track_id = tracked_ids[track_idx]
                detection = detections[det_idx]
                self.tracks[track_id].update(frame_idx, detection.box, detection.confidence)
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_detections:
                detection = detections[det_idx]
                new_id = self.next_id
                self.next_id += 1
                
                track = TrackedObject(
                    object_id=new_id,
                    class_id=detection.class_id,
                    class_name=detection.class_name
                )
                track.update(frame_idx, detection.box, detection.confidence)
                self.tracks[new_id] = track
            
        else:
            # If no existing tracks, create new ones for all detections
            if len(detection_boxes) > 0:
                for detection in detections:
                    new_id = self.next_id
                    self.next_id += 1
                    
                    track = TrackedObject(
                        object_id=new_id,
                        class_id=detection.class_id,
                        class_name=detection.class_name
                    )
                    track.update(frame_idx, detection.box, detection.confidence)
                    self.tracks[new_id] = track
        
        # Remove old tracks
        self._remove_old_tracks(frame_idx)
        
        return self.get_active_tracks()
    
    def _calculate_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Calculate IoU between all pairs of boxes.
        
        Args:
            boxes1: Array of boxes [N, 4]
            boxes2: Array of boxes [M, 4]
            
        Returns:
            IoU matrix of shape [N, M]
        """
        # Convert boxes to [x1, y1, x2, y2] format if needed
        boxes1 = np.array(boxes1).reshape(-1, 4)
        boxes2 = np.array(boxes2).reshape(-1, 4)
        
        # Create IoU matrix
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))
        
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_matrix[i, j] = self._calculate_iou(box1, box2)
        
        return iou_matrix
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes.
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        if union_area > 0:
            return intersection_area / union_area
        else:
            return 0.0
    
    def _match_detections(self, 
                        iou_matrix: np.ndarray, 
                        num_tracks: int, 
                        num_detections: int) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to tracks using IoU.
        
        Args:
            iou_matrix: IoU matrix of shape [num_tracks, num_detections]
            num_tracks: Number of existing tracks
            num_detections: Number of new detections
            
        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
        """
        matches = []
        unmatched_tracks = list(range(num_tracks))
        unmatched_detections = list(range(num_detections))
        
        # Sort IoU matrix by value
        flat_indices = np.argsort(-iou_matrix.flatten())
        row_indices, col_indices = np.unravel_index(flat_indices, iou_matrix.shape)
        
        # Match greedily
        for i in range(min(len(row_indices), len(col_indices))):
            row, col = row_indices[i], col_indices[i]
            
            # Skip if IoU is below threshold
            if iou_matrix[row, col] < self.iou_threshold:
                continue
            
            # Skip if track or detection is already matched
            if row in unmatched_tracks and col in unmatched_detections:
                matches.append((row, col))
                unmatched_tracks.remove(row)
                unmatched_detections.remove(col)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _remove_old_tracks(self, current_frame: int):
        """Remove tracks that haven't been updated recently.
        
        Args:
            current_frame: Current frame index
        """
        track_ids_to_remove = []
        
        for track_id, track in self.tracks.items():
            if current_frame - track.last_seen > self.max_age:
                track_ids_to_remove.append(track_id)
        
        for track_id in track_ids_to_remove:
            del self.tracks[track_id]

# Placeholder for DeepSORT tracker
# In a real implementation, this would use the DeepSORT algorithm
class DeepSORTTracker(ObjectTracker):
    def __init__(self, model_path: str = None, name: str = "deepsort"):
        """Initialize DeepSORT tracker.
        
        Args:
            model_path: Path to DeepSORT model weights
            name: Tracker name
        """
        super().__init__(name=name)
        
        # In a real implementation, this would initialize DeepSORT
        # Using IoUTracker as a placeholder
        self.tracker = IoUTracker(iou_threshold=0.4, max_age=10)
        
        # Define people and vehicle classes
        self.people_classes = self.tracker.people_classes
        self.vehicle_classes = self.tracker.vehicle_classes
        
        logger.info(f"Initialized {name} tracker (using IoUTracker as placeholder)")
    
    def update(self, detections: List[Detection], frame_idx: int) -> List[TrackedObject]:
        """Update tracker with new detections.
        
        Args:
            detections: List of Detection objects
            frame_idx: Current frame index
            
        Returns:
            List of active TrackedObject instances
        """
        self.current_frame = frame_idx
        
        # In a real implementation, this would use DeepSORT
        # Using IoUTracker as a placeholder
        tracked_objects = self.tracker.update(detections, frame_idx)
        
        # Copy tracker state
        self.tracks = self.tracker.tracks
        self.next_id = self.tracker.next_id
        
        return tracked_objects
    
    def reset(self):
        """Reset tracker state."""
        super().reset()
        self.tracker.reset()

# Azure Container App Tracker implementation
class AzureContainerAppTracker(ObjectTracker):
    def __init__(self, name: str = "azure_container_app"):
        """Initialize Azure Container App tracker.
        
        Args:
            name: Tracker name
        """
        super().__init__(name=name)
        
        # Get Azure DeepSORT endpoint from environment
        self.endpoint = os.getenv('AZURE_DEEPSORT_ENDPOINT', '').rstrip('/')
        if not self.endpoint:
            logger.warning("AZURE_DEEPSORT_ENDPOINT not set or empty. Azure Container App tracking will not work.")
        
        # Define people and vehicle classes (same as IoUTracker for consistency)
        self.people_classes = ['person', 'Person', 'people', 'People', 'pedestrian', 'Pedestrian']
        self.vehicle_classes = [
            'car', 'Car', 'vehicle', 'Vehicle', 'truck', 'Truck', 
            'bus', 'Bus', 'motorcycle', 'Motorcycle', 'bicycle', 'Bicycle'
        ]
        
        # Initialize track mapping
        self.track_mapping = {}  # Maps Azure track_id to our internal object_id
        
        # Initialize session for HTTP requests
        self._initialize_http_session()
        
        # Store video ID for tracking session
        self.video_id = str(uuid.uuid4())
        
        logger.info(f"Initialized {name} tracker with endpoint: {self.endpoint}")
        
        # Check if endpoint is reachable
        self._check_endpoint()
    
    def _initialize_http_session(self):
        """Initialize HTTP session."""
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _check_endpoint(self):
        """Check if endpoint is reachable."""
        if not self.endpoint:
            return
        
        try:
            response = self.session.get(f"{self.endpoint}/healthcheck", timeout=5)
            if response.status_code == 200:
                logger.info(f"Azure Container App endpoint is reachable: {response.json()}")
            else:
                logger.warning(f"Azure Container App endpoint returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to reach Azure Container App endpoint: {e}")
    
    def update(self, detections: List[Detection], frame_idx: int) -> List[TrackedObject]:
        """Update tracker by sending detections to Azure Container App.
        
        Args:
            detections: List of Detection objects
            frame_idx: Current frame index
            
        Returns:
            List of active TrackedObject instances
        """
        self.current_frame = frame_idx
        
        if not self.endpoint:
            logger.warning("AZURE_DEEPSORT_ENDPOINT not set. Falling back to IoU tracking.")
            # Fall back to IoU tracking if endpoint not set
            if not hasattr(self, 'fallback_tracker'):
                self.fallback_tracker = IoUTracker(name=f"{self.name}_fallback")
            return self.fallback_tracker.update(detections, frame_idx)
        
        try:
            # Convert frame for the first detection to base64
            # (this assumes all detections are from the same frame)
            frame = None
            if detections and hasattr(detections[0], '_frame'):
                frame = detections[0]._frame
            
            # If frame is not available, we can still send detections
            if frame is None:
                logger.debug("No frame available, sending only detections")
            
            # Convert detections to format expected by Container App
            detection_data = []
            for det in detections:
                detection_data.append({
                    'box': det.box,
                    'class_id': det.class_id,
                    'class_name': det.class_name,
                    'confidence': det.confidence
                })
            
            # Prepare request data
            request_data = {
                'frame_idx': frame_idx,
                'detections': detection_data,
                'video_id': self.video_id
            }
            
            # Add frame data if available
            if frame is not None:
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_base64 = base64.b64encode(img_encoded).decode('utf-8')
                request_data['image'] = img_base64
            
            # Send request to Container App
            response = self.session.post(
                f"{self.endpoint}/api/track",
                json=request_data,
                timeout=30  # Longer timeout for video processing
            )
            
            # Check response
            if response.status_code != 200:
                logger.error(f"Failed to send tracking request: {response.status_code} {response.text}")
                return []
            
            # Parse response
            result = response.json()
            
            # Update our tracking objects based on container app results
            self._update_tracks_from_response(result, frame_idx)
            
            return self.get_active_tracks()
            
        except Exception as e:
            logger.error(f"Error sending tracking request: {e}", exc_info=True)
            return []
    
    def _update_tracks_from_response(self, response: Dict[str, Any], frame_idx: int):
        """Update tracking objects based on Container App response.
        
        Args:
            response: Response from Container App
            frame_idx: Current frame index
        """
        # Extract tracks from response
        tracks = response.get('tracks', [])
        
        # Update our tracking objects
        for track in tracks:
            # Get track_id from response
            track_id = track.get('track_id')
            if track_id is None:
                continue
            
            # Map to our internal object_id if exists, or create new
            if track_id not in self.track_mapping:
                self.track_mapping[track_id] = self.next_id
                self.next_id += 1
            
            object_id = self.track_mapping[track_id]
            
            # Get or create TrackedObject
            if object_id not in self.tracks:
                # Create new TrackedObject
                self.tracks[object_id] = TrackedObject(
                    object_id=object_id,
                    class_id=track.get('class_id', 0),
                    class_name=track.get('class_name', 'unknown')
                )
            
            # Update TrackedObject with new data
            box = track.get('box', [0, 0, 0, 0])
            confidence = track.get('confidence', 0.0)
            self.tracks[object_id].update(frame_idx, box, confidence)
    
    def reset(self):
        """Reset tracker state and also reset tracking on Container App."""
        super().reset()
        self.track_mapping = {}
        self.video_id = str(uuid.uuid4())
        
        # Reset tracking on Container App
        if self.endpoint:
            try:
                response = self.session.post(f"{self.endpoint}/api/reset/{self.video_id}")
                logger.info(f"Reset tracking on Container App: {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to reset tracking on Container App: {e}")
    
    def __del__(self):
        """Clean up resources."""
        # Close session
        if hasattr(self, 'session'):
            self.session.close() 