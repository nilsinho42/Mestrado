"""
Object tracking implementations for video processing.
Supports IoU-based tracking and DeepSORT tracking for various providers.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union
import uuid
import logging

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Object detection result."""
    frame_number: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] normalized coordinates
    service: str
    class_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, detection_dict: Dict[str, Any], service: str) -> 'Detection':
        """Create a Detection from a dictionary."""
        return cls(
            frame_number=detection_dict.get("frame_number", 0),
            class_name=detection_dict.get("detection_type", "unknown"),
            confidence=detection_dict.get("confidence", 0.0),
            bbox=detection_dict.get("bbox", [0, 0, 0, 0]),
            service=service,
            class_id=detection_dict.get("metadata", {}).get("class_id"),
            metadata=detection_dict.get("metadata", {})
        )

@dataclass
class TrackedObject:
    """Object tracked across multiple frames."""
    id: str
    class_name: str
    first_seen: int  # frame number
    last_seen: int   # frame number
    confidence_history: List[float]
    bbox_history: List[List[float]]
    service: str
    frame_numbers: List[int]  # List of frame numbers where object was detected
    frames_since_last_update: int = 0
    class_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_detection(cls, detection: Detection, next_id: int) -> 'TrackedObject':
        """Create a TrackedObject from a Detection."""
        return cls(
            id=str(next_id),
            class_name=detection.class_name,
            first_seen=detection.frame_number,
            last_seen=detection.frame_number,
            confidence_history=[detection.confidence],
            bbox_history=[detection.bbox],
            service=detection.service,
            frame_numbers=[detection.frame_number],
            class_id=detection.class_id,
            metadata=detection.metadata
        )

    def update(self, detection: Detection) -> None:
        """Update the tracked object with a new detection."""
        self.confidence_history.append(detection.confidence)
        self.bbox_history.append(detection.bbox)
        self.frame_numbers.append(detection.frame_number)
        self.last_seen = detection.frame_number
        self.frames_since_last_update = 0
        # Update metadata with latest detection
        if detection.metadata:
            if not self.metadata:
                self.metadata = {}
            self.metadata.update(detection.metadata)

    def average_confidence(self) -> float:
        """Calculate average confidence over the tracking history."""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracked object to dictionary."""
        return {
            "id": self.id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "frame_count": len(self.frame_numbers),
            "frame_numbers": self.frame_numbers,
            "confidence": self.average_confidence(),
            "service": self.service,
            "last_bbox": self.bbox_history[-1] if self.bbox_history else None,
            "trajectory": self.bbox_history,
            "metadata": self.metadata
        }


class BaseTracker:
    """Base class for object trackers."""
    
    def __init__(self, name: str = "base", **kwargs):
        self.name = name
        self.tracks = []
        self.next_id = 0
        self.current_frame = 0
        logger.debug(f"Initialized {self.name} tracker")
    
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Update tracker with new detections. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement update method")
    
    def get_active_tracks(self) -> List[TrackedObject]:
        """Get currently active tracks."""
        return [track for track in self.tracks if track.frames_since_last_update == 0]
    
    def get_all_tracks(self) -> List[TrackedObject]:
        """Get all tracks."""
        return self.tracks
    
    def get_unique_objects(self) -> List[TrackedObject]:
        """Get all unique objects that have been tracked."""
        return [track for track in self.tracks if len(track.frame_numbers) > 0]
    
    def count_by_class(self) -> Dict[str, int]:
        """Count unique objects by class."""
        counts = {}
        for track in self.get_unique_objects():
            if track.class_name in counts:
                counts[track.class_name] += 1
            else:
                counts[track.class_name] = 1
        return counts
    
    def reset(self) -> None:
        """Reset the tracker."""
        self.tracks = []
        self.next_id = 0
        self.current_frame = 0


class IoUTracker(BaseTracker):
    """
    Simple IoU-based object tracker.
    Tracks objects by calculating Intersection over Union between bounding boxes
    in consecutive frames.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_frames_to_skip: int = 5, 
                 name: str = "iou_tracker", **kwargs):
        super().__init__(name=name)
        self.iou_threshold = iou_threshold
        self.max_frames_to_skip = max_frames_to_skip
        logger.debug(f"IoU tracker initialized with threshold={iou_threshold}, max_skip={max_frames_to_skip}")
    
    @staticmethod
    def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes in format [x1, y1, x2, y2]."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        # Calculate areas
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Update tracks with new detections."""
        self.current_frame += 1
        
        # If no tracks exist, create new tracks for all detections
        if not self.tracks:
            for detection in detections:
                track = TrackedObject.from_detection(detection, self.next_id)
                self.tracks.append(track)
                self.next_id += 1
            return self.get_active_tracks()
        
        # Calculate IoU between all tracks and detections
        matched_track_indices = set()
        matched_detection_indices = set()
        
        # Update existing tracks with matched detections
        for i, track in enumerate(self.tracks):
            best_iou = self.iou_threshold
            best_detection_idx = None
            
            for j, detection in enumerate(detections):
                if j in matched_detection_indices:
                    continue
                    
                if track.class_name != detection.class_name:
                    continue
                    
                iou = self.calculate_iou(track.bbox_history[-1], detection.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_detection_idx = j
            
            if best_detection_idx is not None:
                track.update(detections[best_detection_idx])
                matched_track_indices.add(i)
                matched_detection_indices.add(best_detection_idx)
            else:
                track.frames_since_last_update += 1
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                track = TrackedObject.from_detection(detection, self.next_id)
                self.tracks.append(track)
                self.next_id += 1
        
        # Remove stale tracks
        self.tracks = [
            track for track in self.tracks 
            if track.frames_since_last_update <= self.max_frames_to_skip
        ]
        
        # Return only active tracks (those updated in current frame)
        return self.get_active_tracks()


# Placeholder for DeepSORT integration - will be implemented later
class DeepSORTTracker(BaseTracker):
    """
    DeepSORT tracker integration.
    This is a placeholder and will be implemented with actual DeepSORT code.
    """
    
    def __init__(self, model_path: Optional[str] = None, max_age: int = 30, 
                 n_init: int = 3, name: str = "deepsort", **kwargs):
        super().__init__(name=name)
        self.max_age = max_age
        self.n_init = n_init
        self.model_path = model_path
        # Fallback to IoU tracker for now
        self.iou_tracker = IoUTracker(name=f"{name}_fallback")
        logger.warning(f"DeepSORT tracker is a placeholder. Using IoU tracker as fallback.")
    
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """
        Update tracker with new detections.
        Currently falls back to IoU tracker.
        """
        # This will be replaced with actual DeepSORT implementation
        return self.iou_tracker.update(detections)


# Factory function to create the appropriate tracker based on provider
def create_tracker(provider: str, **kwargs) -> BaseTracker:
    """
    Create an appropriate tracker based on the provider.
    
    Args:
        provider: Provider name ('local', 'aws', 'azure')
        **kwargs: Additional configuration for the tracker
    
    Returns:
        A tracker instance
    """
    name = kwargs.get('name', provider)
    
    if provider.lower() == 'local':
        # For local processing, use IoU tracker by default
        return IoUTracker(name=f"{name}_tracker", **kwargs)
    
    elif provider.lower() == 'aws':
        # For AWS, we'll use a placeholder for AWS Fargate + DeepSORT
        return DeepSORTTracker(name=f"{name}_tracker", **kwargs)
    
    elif provider.lower() == 'azure':
        # For Azure, we'll use a placeholder for Azure Container App + DeepSORT
        return DeepSORTTracker(name=f"{name}_tracker", **kwargs)
    
    else:
        # Default to IoU tracker
        logger.warning(f"Unknown provider '{provider}'. Using default IoU tracker.")
        return IoUTracker(name=f"{name}_tracker", **kwargs)
