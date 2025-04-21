from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging

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

    def update(self, det: Detection):
        self.last_seen = det.frame_number
        self.bboxes.append(det.bbox)
        self.confidences.append(det.confidence)
        self.frames_since_last_update = 0
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
            "confidence": sum(self.confidences) / len(self.confidences),
            "last_bbox": self.bboxes[-1],
            "trajectory": self.bboxes,
            "metadata": self.metadata
        }

class DeepSORTTracker:
    def __init__(self, iou_threshold=0.3, max_age=30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: List[TrackedObject] = []
        self.next_id = 0
        self.frame_idx = 0
        self.results = []

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
        self.frame_idx += 1
        updated = set()

        for track in self.tracks:
            best_iou, best_det = 0, None
            for i, det in enumerate(detections):
                if i in updated:
                    continue
                iou = self.iou(track.bboxes[-1], det.bbox)
                if track.class_name == det.class_name:
                    iou += 0.1
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou, best_det = iou, i

            if best_det is not None:
                track.update(detections[best_det])
                updated.add(best_det)
            else:
                track.frames_since_last_update += 1

        for i, det in enumerate(detections):
            if i not in updated:
                self.tracks.append(
                    TrackedObject(
                        id=str(self.next_id),
                        class_name=det.class_name,
                        first_seen=det.frame_number,
                        last_seen=det.frame_number,
                        bboxes=[det.bbox],
                        confidences=[det.confidence],
                        class_id=det.class_id,
                        metadata=det.metadata
                    )
                )
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.frames_since_last_update <= self.max_age]
        return [t for t in self.tracks if t.frames_since_last_update == 0]

    def process_frame(self, detections: List[Detection]) -> Dict[str, Any]:
        active = self.update(detections)
        frame_result = {
            "frame_number": self.frame_idx,
            "active_tracks": [t.to_dict() for t in active],
            "detections": [d.__dict__ for d in detections]
        }
        self.results.append(frame_result)
        return frame_result

    def get_results(self) -> Dict[str, Any]:
        return {
            "frames": self.results,
            "all_tracks": [t.to_dict() for t in self.tracks]
        }


def create_tracker(**kwargs) -> DeepSORTTracker:
    return DeepSORTTracker(**kwargs)
