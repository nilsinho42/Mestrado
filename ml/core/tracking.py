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
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a video file for object tracking.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with tracking results
        """
        import cv2
        import time
        import numpy as np
        from pathlib import Path
        
        # Reset tracker state for new video
        self.reset()
        
        # Start timing
        start_time = time.time()
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return {"error": "Could not open video file"}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize results
        all_tracks = []
        frame_results = []
        
        # Find detector in global scope or context
        # This is a workaround - ideally the detector should be passed as an argument
        detector = None
        from main import get_detector_for_provider
        try:
            detector = get_detector_for_provider(self.name.split('_')[0])
        except Exception as e:
            logger.error(f"Could not get detector for provider {self.name}: {e}")
            # Fallback to a simple per-frame processing approach without detection
            
        # Process frames
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with detector if available
                if detector:
                    detections_dict = detector.detect(frame)
                    # Convert detection dictionaries to Detection objects
                    detections = []
                    for det in detections_dict:
                        detections.append(Detection(
                            frame_number=frame_idx,
                            class_name=det.get("detection_type", det.get("class_name", "unknown")),
                            confidence=det.get("confidence", 0.0),
                            bbox=det.get("bbox", [0, 0, 0, 0]),
                            service=self.name.split('_')[0],
                            class_id=det.get("class_id"),
                            metadata=det.get("metadata", {})
                        ))
                else:
                    # Skip detection if no detector available
                    detections = []
                
                # Update tracker with new detections
                active_tracks = self.update(detections)
                
                # Store tracks for this frame
                frame_result = {
                    "frame_number": frame_idx,
                    "detections": [d.__dict__ for d in detections],
                    "active_tracks": [t.to_dict() for t in active_tracks],
                }
                frame_results.append(frame_result)
                
                # Add unique tracks to overall results
                for track in active_tracks:
                    if track.id not in [t.id for t in all_tracks]:
                        all_tracks.append(track)
                
                frame_idx += 1
                
        finally:
            cap.release()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Count objects by class
        class_counts = self.count_by_class()
        
        # Build final results
        results = {
            "video_path": video_path,
            "video_name": Path(video_path).stem,
            "processing_time": processing_time,
            "frames_processed": frame_idx,
            "unique_tracks": len(all_tracks),
            "class_counts": class_counts,
            "frame_results": frame_results,
            "tracks": [t.to_dict() for t in self.get_unique_objects()],
            "summary": {
                "people_count": class_counts.get("person", 0),
                "vehicle_count": sum([class_counts.get(c, 0) for c in ["car", "truck", "bus", "motorcycle"]]),
                "processing_time": processing_time,
                "cost": 0.0  # Local processing has no cloud cost
            }
        }
        
        return results


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
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a video file for object tracking.
        Currently falls back to IoU tracker.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with tracking results
        """
        # Use the IoU tracker's process_video method as a fallback
        return self.iou_tracker.process_video(video_path)


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
