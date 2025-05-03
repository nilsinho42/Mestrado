import numpy as np
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("Ultralytics package not found. Please install with 'pip install ultralytics'")
    raise

class Detection:
    """
    This class represents a bounding box detection in a single image.
    """
    def __init__(self, bbox, confidence, class_id, class_name, frame_number=None, features=None):
        """
        Initialize a detection.
        
        Args:
            bbox: Bounding box in [x1, y1, x2, y2] format
            confidence: Detection confidence
            class_id: Class ID
            class_name: Class name
            frame_number: Optional frame number
            features: Optional feature vector
        """
        # Convert [x1, y1, x2, y2] to [x, y, width, height]
        self.tlwh = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], dtype=np.float64)
        self.confidence = float(confidence)
        self.class_id = class_id
        self.class_name = class_name
        self.frame_number = frame_number
        self.features = features

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio, height)`"""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

class YOLOTracker:
    """
    YOLO-based tracker implementation that follows the same interface as DeepSORTTracker.
    This allows for drop-in replacement of DeepSORT with YOLO's tracking capabilities.
    """
    def __init__(self, model_path="yolo11n.pt", max_age=70, n_init=5, conf=0.5, use_features=False, max_iou_distance=0.7):
        """
        Initialize the YOLO tracker.
        
        Args:
            model_path: Path to the YOLO model weights
            max_age: Maximum number of frames to keep a track alive without matching detections
            n_init: Number of consecutive frames a detection should appear to start a new track
            conf: Confidence threshold for detections
            use_features: Flag to use appearance features (not used in YOLO tracking)
            max_iou_distance: Maximum IoU distance for track association (not used in YOLO tracking)
        """
        # Save parameters for compatibility with DeepSORT interface
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.use_features = use_features
        self.conf = conf
        self.model_path = model_path
        
        # Load the YOLO model
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Loaded YOLO model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
            
        # Initialize counters for metrics
        self.vehicle_count = 0
        self.person_count = 0
        self.tracked_ids = {
            'vehicle': set(),
            'person': set()
        }
        
        # Track management
        self.tracks = []
        self.current_tracks = []
    
    def _is_vehicle(self, class_name):
        """Check if the class name represents a vehicle."""
        vehicle_classes = [
            'car', 'truck', 'bus', 'vehicle', 'automobile', 'van', 'suv', 'motorbike', 'bicycle', 
            'transportation', 'taxi', 'ambulance', 'police car', 'motorcycle',
            'Car', 'Truck', 'Bus', 'Vehicle', 'Automobile', 'Van', 'SUV', 'Motorbike', 'Bicycle',
            'Transportation', 'Taxi', 'Ambulance', 'Police Car', 'Motorcycle'
        ]
        return any(vehicle_class.lower() in class_name.lower() for vehicle_class in vehicle_classes)
        
    def _is_person(self, class_name):
        """Check if the class name represents a person."""
        person_classes = [
            'person', 'human', 'people', 'pedestrian', 'man', 'woman', 'child', 'baby',
            'Person', 'Human', 'People', 'Pedestrian', 'Man', 'Woman', 'Child', 'Baby'
        ]
        return any(person_class.lower() in class_name.lower() for person_class in person_classes)
    
    def _update_counts(self, track_id, class_name):
        """Update vehicle and person counts based on track class."""
        if self._is_vehicle(class_name):
            if track_id not in self.tracked_ids['vehicle']:
                self.tracked_ids['vehicle'].add(track_id)
                self.vehicle_count += 1
                
        elif self._is_person(class_name):
            if track_id not in self.tracked_ids['person']:
                self.tracked_ids['person'].add(track_id)
                self.person_count += 1
    
    def predict(self):
        """
        Stub for DeepSORT compatibility. YOLO prediction happens within the update method.
        """
        pass
    
    def update(self, detections, frame=None):
        """
        Perform tracking on an image with optional pre-computed detections.
        This method performs both detection and tracking using YOLO's built-in tracker.
        
        Args:
            detections: List of detections (dict or Detection objects) - ignored in favor of YOLO's detection
            frame: Input frame for tracking
            
        Returns:
            A list of tracks
        """
        if frame is None:
            logger.warning("No frame provided to YOLO tracker. Cannot perform tracking.")
            return self.tracks
        
        # Run the tracker
        try:
            results = self.model.track(
                source=frame,
                persist=True,  # Maintain track IDs across frames
                conf=self.conf,
                verbose=False
            )
            
            # Extract tracks from results
            self.tracks = []  # Reset tracks for this frame
            
            if results and len(results) > 0:
                result = results[0]  # Get the first result
                
                # Check if tracking information is available
                if hasattr(result, 'boxes') and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format
                    track_ids = result.boxes.id.int().cpu().numpy()  # Get track IDs
                    confs = result.boxes.conf.cpu().numpy()  # Get confidences
                    cls_ids = result.boxes.cls.int().cpu().numpy()  # Get class IDs
                    
                    # Get class names
                    class_names = [result.names[c] for c in cls_ids]
                    
                    # Create simplified track objects with only what's needed
                    for i in range(len(track_ids)):
                        track_id = int(track_ids[i])
                        box = boxes[i].tolist()  # [x1, y1, x2, y2]
                        confidence = float(confs[i])
                        class_id = int(cls_ids[i])
                        class_name = class_names[i]
                        
                        # Update counters
                        self._update_counts(track_id, class_name)
                        
                        # Create a simple Track object with only the needed attributes
                        track = type('Track', (), {
                            'track_id': track_id,
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': box  # Already converted to a list
                        })
                        
                        self.tracks.append(track)
            
            # Save current tracks for reference
            self.current_tracks = self.tracks
            
            return self.tracks
            
        except Exception as e:
            logger.error(f"Error in YOLO tracking: {e}")
            return self.current_tracks
        
    def get_results(self):
        """Get counts and other metrics."""
        return {
            "counts": {
                "vehicle_count": self.vehicle_count,
                "person_count": self.person_count
            },
            "tracked_ids": {
                "vehicle": list(self.tracked_ids['vehicle']),
                "person": list(self.tracked_ids['person'])
            }
        }

# Alias DeepSORTTracker to YOLOTracker for backward compatibility
DeepSORTTracker = YOLOTracker

# Create a tracker function that matches the API of the AWS version
def create_tracker(**kwargs):
    """Create a YOLO tracker with the specified parameters."""
    return YOLOTracker(**kwargs)

# For backward compatibility with AWS version
Tracker = YOLOTracker 