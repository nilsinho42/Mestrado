from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    frame_number: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    class_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Detection':
        return cls(
            frame_number=d.get("frame_number", 0),
            class_name=d.get("class_name", "unknown"),
            confidence=d.get("confidence", 0.0),
            bbox=d.get("bbox", [0, 0, 0, 0]),
            class_id=d.get("metadata", {}).get("class_id"),
            metadata=d.get("metadata", {})
        )

@dataclass
class TrackedObject:
    id: str
    class_name: str
    first_seen: int
    last_seen: int
    bboxes: List[List[float]]
    confidences: List[float]
    frames_since_last_update: int = 0
    class_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    hits: int = 1
    state: int = 1  # 1=tentative, 2=confirmed, 3=deleted (matching tracker_app)

    def update(self, det: Detection):
        self.last_seen = det.frame_number
        self.bboxes.append(det.bbox)
        self.confidences.append(det.confidence)
        self.frames_since_last_update = 0
        self.hits += 1
        if det.metadata:
            if not self.metadata:
                self.metadata = {}
            self.metadata.update(det.metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "confidence": sum(self.confidences) / len(self.confidences) if self.confidences else 0.0,
            "last_bbox": self.bboxes[-1] if self.bboxes else [0, 0, 0, 0],
            "trajectory": self.bboxes,
            "metadata": self.metadata,
            "state": self.state
        }
    
    def is_confirmed(self) -> bool:
        """Returns True if this track is confirmed."""
        return self.state == 2
    
    def is_tentative(self) -> bool:
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == 1
    
    def is_deleted(self) -> bool:
        """Returns True if this track is deleted."""
        return self.state == 3
    
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        self.frames_since_last_update += 1

class DeepSORTTracker:
    def __init__(self, max_iou_distance=0.7, max_age=30, n_init=3):
        """
        Initialize with parameters matching tracker_app.
        
        Args:
            max_iou_distance: Maximum IOU distance for considering a match
            max_age: Maximum number of frames a track can be missing before deletion
            n_init: Minimum number of hits required for a track to be confirmed
        """
        self.max_iou_distance = max_iou_distance
        self.iou_threshold = max_iou_distance  # For backward compatibility
        self.max_age = max_age
        self.n_init = n_init
        self.tracks: List[TrackedObject] = []
        self.next_id = 1  # Start with 1 to match tracker_app
        self.frame_idx = 0
        self.results = []
        self.vehicle_ids = set()  # Set of unique vehicle track IDs
        self.person_ids = set()   # Set of unique person track IDs
        self.session_id = f"local_{int(time.time())}"  # Generate a session ID similar to video_id

    @staticmethod
    def _is_vehicle(class_name: str) -> bool:
        """Check if class name represents a vehicle."""
        if not class_name:
            return False
        vehicle_types = ['car', 'truck', 'bus', 'vehicle', 'automobile', 'van', 'suv', 'motorbike', 'bicycle']
        return class_name.lower() in vehicle_types

    @staticmethod
    def _is_person(class_name: str) -> bool:
        """Check if class name represents a person."""
        if not class_name:
            return False
        person_types = ['person', 'pedestrian', 'human', 'man', 'woman', 'child', 'baby', 'girl', 'boy']
        return class_name.lower() in person_types

    @staticmethod
    def iou(b1: List[float], b2: List[float]) -> float:
        x1, y1, x2, y2 = b1
        x1_, y1_, x2_, y2_ = b2
        xi1, yi1 = max(x1, x1_), max(y1, y1_)
        xi2, yi2 = min(x2, x2_), min(y2, y2_)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        a1 = (x2 - x1) * (y2 - y1)
        a2 = (x2_ - x1_) * (y2_ - y1_)
        return inter / (a1 + a2 - inter)

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Update tracks with new detections."""
        updated = set()

        # First, increment frames_since_last_update for all tracks
        for track in self.tracks:
            track.mark_missed()

        # Associate existing tracks with new detections
        for track in self.tracks:
            if track.is_deleted():
                continue
                
            best_iou, best_det = 0, None
            for i, det in enumerate(detections):
                if i in updated:
                    continue
                iou = self.iou(track.bboxes[-1], det.bbox)
                if track.class_name == det.class_name:
                    iou += 0.1
                if iou > best_iou and iou >= self.max_iou_distance:
                    best_iou, best_det = iou, i

            if best_det is not None:
                track.update(detections[best_det])
                # Confirm the track if it has enough hits
                if track.state == 1 and track.hits >= self.n_init:
                    track.state = 2  # confirmed
                updated.add(best_det)

        # Mark tracks for deletion if they've been missing too long
        for track in self.tracks:
            if track.state == 1 and track.frames_since_last_update > 0:
                track.state = 3  # deleted
            elif track.frames_since_last_update > self.max_age:
                track.state = 3  # deleted

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in updated:
                track_id = str(self.next_id)
                new_track = TrackedObject(
                    id=track_id,
                    class_name=det.class_name,
                    first_seen=det.frame_number,
                    last_seen=det.frame_number,
                    bboxes=[det.bbox],
                    confidences=[det.confidence],
                    class_id=det.class_id,
                    metadata=det.metadata,
                    hits=1,
                    state=1  # tentative
                )
                
                # Categorize the new track based on its class name
                if self._is_vehicle(det.class_name):
                    self.vehicle_ids.add(track_id)
                elif self._is_person(det.class_name):
                    self.person_ids.add(track_id)
                
                self.tracks.append(new_track)
                self.next_id += 1

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Return active tracks (not deleted and not missing in this frame)
        return [t for t in self.tracks if t.frames_since_last_update == 0]

    def process_frame(self, detections: List[Detection]) -> Dict[str, Any]:
        """
        Process a frame of detections, updating tracking state.
        Returns both the traditional result format and tracker_app compatible format.
        """
        self.frame_idx += 1
        start_time = time.time()
        active_tracks = self.update(detections)
        
        # Store in traditional format for backward compatibility
        legacy_frame_result = {
            "frame_number": self.frame_idx,
            "active_tracks": [t.to_dict() for t in active_tracks],
            "detections": [d.__dict__ for d in detections]
        }
        self.results.append(legacy_frame_result)
        
        # Create tracker_app compatible format
        tracker_app_tracks = []
        for track in self.tracks:
            if track.is_confirmed():
                # Format the track like tracker_app response
                tracker_app_tracks.append({
                    'track_id': int(track.id),
                    'class_id': track.class_id if track.class_id is not None else 0,
                    'class_name': track.class_name,
                    'box': track.bboxes[-1] if track.bboxes else [0, 0, 0, 0],
                    'confidence': track.confidences[-1] if track.confidences else 0.0
                })
        
        processing_time = time.time() - start_time
        
        # Return tracker_app compatible format
        return {
            'video_id': self.session_id,
            'frame_idx': self.frame_idx,
            'tracks': tracker_app_tracks,
            'processing_time': processing_time,
            # Keep the legacy format data for backward compatibility
            'legacy_data': legacy_frame_result
        }

    def get_results(self) -> Dict[str, Any]:
        """
        Get tracking results in both traditional format and counts compatible with tracker_app.
        """
        # Get all the tracked objects (keeping backward compatibility)
        all_tracks = [t.to_dict() for t in self.tracks]
        
        # Count unique vehicles and people based on track IDs
        vehicle_count = len(self.vehicle_ids)
        person_count = len(self.person_ids)
        
        # Create a list of vehicle tracks and person tracks for reference
        vehicle_tracks = [t for t in all_tracks if self._is_vehicle(t['class_name'])]
        person_tracks = [t for t in all_tracks if self._is_person(t['class_name'])]
        
        # Return the comprehensive results (backward compatible format)
        return {
            "frames": self.results,
            "all_tracks": all_tracks,
            "counts": {
                "vehicle_count": vehicle_count,
                "person_count": person_count,
                "total_unique_objects": vehicle_count + person_count
            },
            "vehicle_tracks": vehicle_tracks,
            "person_tracks": person_tracks,
            # Add tracker_app compatible session info
            "session_info": {
                "video_id": self.session_id,
                "frames_processed": self.frame_idx,
                "last_update": time.time()
            }
        }


def create_tracker(**kwargs) -> DeepSORTTracker:
    """Create and return a DeepSORTTracker instance with the specified parameters."""
    return DeepSORTTracker(**kwargs)
