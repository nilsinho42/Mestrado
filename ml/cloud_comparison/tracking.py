import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import uuid

@dataclass
class Detection:
    frame_number: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] normalized coordinates
    service: str

@dataclass
class TrackedObject:
    id: str
    class_name: str
    first_seen: int  # frame number
    last_seen: int   # frame number
    confidence_history: List[float]
    bbox_history: List[List[float]]
    service: str
    frame_numbers: List[int]  # List of frame numbers where object was detected
    frames_since_last_update: int = 0

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
            frame_numbers=[detection.frame_number]
        )

    def update(self, detection: Detection, current_frame: int) -> None:
        """Update the tracked object with a new detection."""
        self.confidence_history.append(detection.confidence)
        self.bbox_history.append(detection.bbox)
        self.frame_numbers.append(detection.frame_number)
        self.last_seen = detection.frame_number
        self.frames_since_last_update = 0

class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_frames_to_skip=5):
        self.tracks = []
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_frames_to_skip = max_frames_to_skip
        self.current_frame = 0

    def calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes."""
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

    def update(self, detections):
        """Update tracks with new detections."""
        self.current_frame += 1
        
        # If no tracks exist, create new tracks for all detections
        if not self.tracks:
            for detection in detections:
                track = TrackedObject.from_detection(detection, self.next_id)
                self.tracks.append(track)
                self.next_id += 1
            return self.tracks

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
                track.update(detections[best_detection_idx], self.current_frame)
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
        return [track for track in self.tracks if track.frames_since_last_update == 0]

    def get_unique_objects(self):
        """Get all unique objects that have been tracked."""
        return [track for track in self.tracks if len(track.frame_numbers) > 0] 