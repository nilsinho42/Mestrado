import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from deep_sort_realtime.deepsort_tracker import DeepSort

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

class DeepSortTracker:
    """
    DeepSORT tracker implementation using deep_sort_realtime package.
    """
    def __init__(self, model_path="yolo11n.pt", max_age=70, n_init=5, conf=0.5, use_features=True, max_iou_distance=0.7):
        """
        Initialize the DeepSORT tracker.
        
        Args:
            model_path: Path to the YOLO model weights for detection
            max_age: Maximum number of frames to keep a track alive without matching detections
            n_init: Number of consecutive frames a detection should appear to start a new track
            conf: Confidence threshold for detections
            use_features: Flag to use appearance features
            max_iou_distance: Maximum IoU distance for track association
        """
        # Save parameters
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.use_features = use_features
        self.conf = conf
        self.model_path = model_path
        
        # Initialize the DeepSORT trackers for vehicles and people
        self.vehicle_tracker = DeepSort(embedder_gpu=False, half=False, bgr=True, n_init=4, max_age=100, nn_budget=150, max_cosine_distance=0.7, max_iou_distance=0.7) 
        # For people tracking, use a specialized embedder and longer max_age
        self.people_tracker = DeepSort(embedder='torchreid', embedder_gpu=False, half=False, bgr=True, n_init=10, max_age=150)
            
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
    
    # def _is_vehicle(self, class_name):
    #     """Check if the class name represents a vehicle."""
    #     vehicle_classes = [
    #         'car', 'truck', 'bus', 'vehicle', 'automobile', 'van', 'suv', 'motorbike', 'bicycle', 
    #         'transportation', 'taxi', 'ambulance', 'police car', 'motorcycle',
    #         'Car', 'Truck', 'Bus', 'Vehicle', 'Automobile', 'Van', 'SUV', 'Motorbike', 'Bicycle',
    #         'Transportation', 'Taxi', 'Ambulance', 'Police Car', 'Motorcycle'
    #     ]
    #     return any(vehicle_class.lower() in class_name.lower() for vehicle_class in vehicle_classes)
        
    # def _is_person(self, class_name):
    #     """Check if the class name represents a person."""
    #     person_classes = [
    #         'person', 'human', 'people', 'pedestrian', 'man', 'woman', 'child', 'baby',
    #         'Person', 'Human', 'People', 'Pedestrian', 'Man', 'Woman', 'Child', 'Baby'
    #     ]
    #     return any(person_class.lower() in class_name.lower() for person_class in person_classes)
    
    def _update_counts(self, track_id, class_name):
        """Update vehicle and person counts based on track class."""
        if class_name.lower() == 'vehicle':
            if track_id not in self.tracked_ids['vehicle']:
                self.tracked_ids['vehicle'].add(track_id)
                self.vehicle_count += 1
        else:
            if track_id not in self.tracked_ids['person']:
                self.tracked_ids['person'].add(track_id)
                self.person_count += 1
    
    def update(self, detections, frame=None):
        """
        Perform detection and tracking on an image.
        
        Args:
            detections: List of detections (dict or Detection objects) - can be empty or None
            frame: Input frame for tracking (required)
            
        Returns:
            A list of tracks
        """
        if frame is None:
            logger.warning("No frame provided to DeepSORT tracker. Cannot perform tracking.")
            return self.tracks
        
        # Return early if no detections
        if not detections:
            return self.tracks
        
        # Process detections with DeepSORT trackers
        vehicle_detections = []
        people_detections = []
        
        # Separate detections into vehicle and people
        for det in detections:
            x1, y1, x2, y2 = det.get('box')
            box_xywh = [x1, y1, x2 - x1, y2 - y1]
            # Handle both formats: tuple from YOLO conversion or dict from external source
            # Add to appropriate tracker list
            confidence = det.get('confidence')
            class_name = det.get('class_name')

            if det.get('class_name') == 'vehicle':
                vehicle_detections.append((box_xywh, confidence, class_name))
            else:
                people_detections.append((box_xywh, confidence, class_name))
        
        all_tracks = []
        
        # Process vehicle detections
        vehicle_tracks = []
        if vehicle_detections:
            try:
                vehicle_tracks = self.vehicle_tracker.update_tracks(vehicle_detections, frame=frame)
            except Exception as e:
                logger.error(f"Error during vehicle tracker update: {e}")
        
        # Process people detections
        people_tracks = []
        if people_detections:
            try:
                people_tracks = self.people_tracker.update_tracks(people_detections, frame=frame)
            except Exception as e:
                logger.error(f"Error during people tracker update: {e}")
        
        # Combine all tracks
        all_tracks = vehicle_tracks + people_tracks
        
        # Convert DeepSort Track objects into a list of tracks
        self.tracks = []
        for track in all_tracks:
            if not track.is_confirmed():
                continue
                
            try:
                # Get bounding box in [x1, y1, x2, y2] format
                bbox = track.to_ltrb() 
                box_list = bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
                
                # Get class name from the track - this is now explicitly set above
                class_name = getattr(track, 'det_class')
                
                # Update track counters
                self._update_counts(track.track_id, class_name)
                
                # Create a track dictionary with all necessary fields
                track_dict = {
                    'track_id': int(track.track_id),
                    'box': box_list,
                    'confidence': float(getattr(track, 'det_conf', 1.0) if getattr(track, 'det_conf', 1.0) is not None else 1.0),
                    'class_name': str(class_name)
                }
                
                # Store the track
                self.tracks.append(track_dict)
            except Exception as e:
                logger.error(f"Error formatting track: {e}")
        
        # Save current tracks for reference
        self.current_tracks = self.tracks
        
        return self.tracks
    
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

# For backward compatibility
YOLOTracker = DeepSortTracker

# Create a tracker function that matches the API of the AWS version
def create_tracker(**kwargs):
    """Create a DeepSORT tracker with the specified parameters."""
    return DeepSortTracker(**kwargs)

# For backward compatibility with AWS version
Tracker = DeepSortTracker 