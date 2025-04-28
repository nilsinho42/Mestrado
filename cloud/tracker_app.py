import os
import time
import uuid
import logging
import base64
import numpy as np
import cv2
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
from dotenv import load_dotenv
import scipy.linalg
from scipy.optimize import linear_sum_assignment

# Load environment variables from root directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create temporary directory for frames
TEMP_DIR = Path("./tmp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Create FastAPI app
app = FastAPI(title="Cloud DeepSORT Tracking API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active tracking sessions
tracking_sessions = {}

def extract_color_features(image, bbox):
    """Extract color histogram features from object patch."""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure valid coordinates
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        # Invalid bbox, return empty features
        return np.zeros(48, dtype=np.float32)
    
    # Extract the patch
    patch = image[y1:y2, x1:x2]
    
    # Convert to HSV color space
    try:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    except cv2.error:
        # Handle potential errors (e.g., empty patch)
        return np.zeros(48, dtype=np.float32)
    
    # Calculate histograms for each channel
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    
    # Normalize histograms
    h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    
    # Concatenate into a feature vector
    return np.concatenate([h_hist, s_hist, v_hist])

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

class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    """
    def __init__(self):
        ndim, dt = 4, 1.
        
        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        # x' = F·x
        mean = np.dot(self._motion_mat, mean)
        # P' = F·P·F' + Q
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
            
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        
        # y = H·x
        mean = np.dot(self._update_mat, mean)
        # S = H·P·H' + R
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T)) + innovation_cov
            
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step."""
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # Compute Kalman gain: K = PH'(S)^-1
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        
        # y = z - Hx (innovation)
        innovation = measurement - projected_mean
        
        # x' = x + Ky (new state)
        new_mean = mean + np.dot(kalman_gain, innovation)
        
        # P' = P - KHP (new covariance)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
            
        return new_mean, new_covariance

class Track:
    """
    A track class for holding a single tracked object state.
    """
    def __init__(self, mean, covariance, track_id, class_id, class_name, confidence, n_init=5, max_age=70, features=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = 1  # tentative
        self._n_init = n_init
        self._max_age = max_age
        self.features = features

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y, width, height)`."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, min y, max x, max y)`."""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a Kalman filter prediction step."""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature cache."""
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0
        if self.state == 1 and self.hits >= self._n_init:
            self.state = 2  # confirmed
        if hasattr(detection, 'features') and detection.features is not None:
            self.features = detection.features

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.state == 1:
            self.state = 3  # deleted
        elif self.time_since_update > self._max_age:
            self.state = 3  # deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == 1

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == 2

    def is_deleted(self):
        """Returns True if this track is deleted."""
        return self.state == 3

class DeepSORTTracker:
    """
    DeepSORT tracker implementation.
    """
    def __init__(self, max_iou_distance=0.7, max_age=70, n_init=5, use_features=True):
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.use_features = use_features
        self.feature_weight = 0.3  # Weight for feature similarity vs IoU
        
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1
        
        # Tracking metrics
        self.vehicle_count = 0
        self.person_count = 0
        self.tracked_ids = {
            'vehicle': set(),
            'person': set()
        }

    def predict(self):
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, frame=None):
        """
        Perform measurement update and track management.
        
        Args:
            detections: List of detections (dict or Detection objects)
            frame: Optional frame for feature extraction
            
        Returns:
            Tracks after update
        """
        # Convert dict detections to Detection objects if needed
        detection_objects = []
        for det in detections:
            if isinstance(det, dict):
                detection_objects.append(Detection(
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    class_id=det['class_id'],
                    class_name=det['class_name'],
                    features=det.get('features')
                ))
            else:
                detection_objects = detections
                break
        
        # Extract features if frame is provided
        if self.use_features and frame is not None:
            for i, det in enumerate(detection_objects):
                if det.features is None:
                    det.features = extract_color_features(frame, det.to_tlbr())
        
        # Run matching cascade
        matches, unmatched_tracks, unmatched_detections = self._match(detection_objects)

        # Update track set
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detection_objects[detection_idx])
            
            # Update counts
            self._update_counts(self.tracks[track_idx])
            
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            
        for detection_idx in unmatched_detections:
            self._initiate_track(detection_objects[detection_idx])
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        return self.tracks

    def _is_vehicle(self, class_name):
        """Check if the class name represents a vehicle."""
        vehicle_keywords = ['car', 'truck', 'bus', 'vehicle', 'automobile', 'van', 'suv', 'motorbike', 'bicycle']
        return any(keyword in class_name.lower() for keyword in vehicle_keywords)
        
    def _is_person(self, class_name):
        """Check if the class name represents a person."""
        person_keywords = ['person', 'pedestrian', 'human', 'man', 'woman', 'child']
        return any(keyword in class_name.lower() for keyword in person_keywords)
    
    def _update_counts(self, track):
        """Update vehicle and person counts based on track class."""
        if not track.is_confirmed():
            return
            
        if self._is_vehicle(track.class_name):
            if track.track_id not in self.tracked_ids['vehicle']:
                self.tracked_ids['vehicle'].add(track.track_id)
                self.vehicle_count += 1
                
        elif self._is_person(track.class_name):
            if track.track_id not in self.tracked_ids['person']:
                self.tracked_ids['person'].add(track.track_id)
                self.person_count += 1

    def _match(self, detections):
        """Match tracks and detections using both IoU and features."""
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks
        matches_a, unmatched_tracks_a, unmatched_detections = \
            self._match_tracks_detections(detections, confirmed_tracks)

        # Associate remaining tracks (unconfirmed) with remaining detections
        matches_b, unmatched_tracks_b, unmatched_detections = \
            self._match_tracks_detections(detections, unconfirmed_tracks, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(unmatched_tracks_a) + list(unmatched_tracks_b)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _match_tracks_detections(self, detections, track_indices, detection_indices=None):
        """Match tracks and detections using both IoU and feature similarity."""
        if detection_indices is None:
            detection_indices = list(range(len(detections)))
        
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return [], track_indices, detection_indices
        
        # Compute IoU cost matrix
        iou_cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
        for i, track_idx in enumerate(track_indices):
            track_bbox = self.tracks[track_idx].to_tlwh()
            for j, detection_idx in enumerate(detection_indices):
                detection_bbox = detections[detection_idx].tlwh
                iou_cost_matrix[i, j] = 1.0 - self._calculate_iou(track_bbox, detection_bbox)
        
        # Compute feature distance cost matrix if features are available
        if self.use_features:
            feature_cost_matrix = np.ones((len(track_indices), len(detection_indices)))
            for i, track_idx in enumerate(track_indices):
                if hasattr(self.tracks[track_idx], 'features') and self.tracks[track_idx].features is not None:
                    track_features = self.tracks[track_idx].features
                    for j, det_idx in enumerate(detection_indices):
                        if hasattr(detections[det_idx], 'features') and detections[det_idx].features is not None:
                            detection_features = detections[det_idx].features
                            feature_cost_matrix[i, j] = self._feature_distance(track_features, detection_features)
            
            # Combine IoU and feature cost matrices
            cost_matrix = (1.0 - self.feature_weight) * iou_cost_matrix + self.feature_weight * feature_cost_matrix
        else:
            cost_matrix = iou_cost_matrix
            
        # Add class name matching bonus
        for i, track_idx in enumerate(track_indices):
            for j, det_idx in enumerate(detection_indices):
                if self.tracks[track_idx].class_name == detections[det_idx].class_name:
                    cost_matrix[i, j] *= 0.9  # 10% bonus for same class
        
        # Apply maximum distance threshold
        cost_matrix[cost_matrix > self.max_iou_distance] = float('inf')
        
        # Perform linear assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        indices = np.column_stack((row_indices, col_indices))
        
        matches, unmatched_tracks, unmatched_detections = [], [], []
        
        for col, detection_idx in enumerate(detection_indices):
            if col not in indices[:, 1]:
                unmatched_detections.append(detection_indices[col])
                
        for row, track_idx in enumerate(track_indices):
            if row not in indices[:, 0]:
                unmatched_tracks.append(track_indices[row])
                
        for row, col in indices:
            track_idx = track_indices[row]
            detection_idx = detection_indices[col]
            if cost_matrix[row, col] >= float('inf'):
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))
                
        return matches, unmatched_tracks, unmatched_detections

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes in tlwh format."""
        # Convert to format x1, y1, x2, y2
        bbox1_tlbr = np.array([bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]])
        bbox2_tlbr = np.array([bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]])
        
        # Get coordinates of intersection
        xi1 = max(bbox1_tlbr[0], bbox2_tlbr[0])
        yi1 = max(bbox1_tlbr[1], bbox2_tlbr[1])
        xi2 = min(bbox1_tlbr[2], bbox2_tlbr[2])
        yi2 = min(bbox1_tlbr[3], bbox2_tlbr[3])
        
        # Calculate area of intersection
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # Calculate areas of both bounding boxes
        bbox1_area = bbox1[2] * bbox1[3]
        bbox2_area = bbox2[2] * bbox2[3]
        
        # Calculate IoU
        union_area = bbox1_area + bbox2_area - inter_area
        if union_area <= 0:
            return 0
        return inter_area / union_area

    def _initiate_track(self, detection):
        """Initialize a new track from a detection."""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        track = Track(mean, covariance, self._next_id, detection.class_id, 
                    detection.class_name, detection.confidence, 
                    self.n_init, self.max_age, features=detection.features)
        self.tracks.append(track)
        
        # Update counts
        self._update_counts(track)
        
        self._next_id += 1

    def _feature_distance(self, track_features, detection_features):
        """Calculate cosine distance between feature vectors."""
        if track_features is None or detection_features is None:
            return 1.0  # Maximum distance
        
        # Normalize features if needed
        if np.linalg.norm(track_features) > 0:
            track_features = track_features / np.linalg.norm(track_features)
        if np.linalg.norm(detection_features) > 0:
            detection_features = detection_features / np.linalg.norm(detection_features)
            
        # Calculate cosine similarity
        similarity = np.dot(track_features, detection_features)
        # Convert to distance (1 - similarity)
        return max(0.0, 1.0 - similarity)
    
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
        
    def get_count_metrics(self):
        """Return count metrics for vehicles and people (for backward compatibility)."""
        return {
            "vehicle_count": self.vehicle_count,
            "person_count": self.person_count
        }

# For backward compatibility
Tracker = DeepSORTTracker

# Models for API requests and responses
class ImageData(BaseModel):
    image: str  # Base64 encoded image
    frame_idx: int
    detections: List[Dict[str, Any]]
    video_id: Optional[str] = None
    
class TrackingResult(BaseModel):
    video_id: Optional[str]
    frame_idx: int
    tracks: List[Dict[str, Any]]
    processing_time: float

@app.get("/")
async def root():
    return {"message": "Cloud DeepSORT Tracking API"}

@app.post("/api/track", response_model=TrackingResult)
async def track_objects(data: ImageData):
    try:
        start_time = time.time()
        
        # Create tracking session if it doesn't exist
        video_id = data.video_id or str(uuid.uuid4())
        if video_id not in tracking_sessions:
            tracking_sessions[video_id] = {
                "tracker": DeepSORTTracker(max_iou_distance=0.7, max_age=70, n_init=5, use_features=True),
                "frames_processed": 0,
                "last_update": time.time()
            }
        
        # Decode image from base64
        try:
            img_bytes = base64.b64decode(data.image)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image")
                
            # Save image temporarily for debugging (optional)
            frame_path = TEMP_DIR / f"{video_id}_{data.frame_idx}.jpg"
            cv2.imwrite(str(frame_path), img)
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        # Convert detections to the format expected by DeepSORT
        deepsort_detections = []
        for det in data.detections:
            # Get box in [x1, y1, x2, y2] format
            box = det['box']
            
            # Create detection instance
            detection = Detection(
                bbox=box,
                confidence=det['confidence'],
                class_id=det['class_id'],
                class_name=det['class_name']
            )
            
            # Extract features if using appearance model
            if tracking_sessions[video_id]["tracker"].use_features:
                features = extract_color_features(img, box)
                detection.features = features
            
            deepsort_detections.append(detection)
        
        # Update tracker with detections
        session = tracking_sessions[video_id]
        session_tracker = session["tracker"]
        
        # Predict step (propagate tracks state)
        session_tracker.predict()
        
        # Update step (associate detections with tracks)
        session_tracker.update(deepsort_detections, img)
        
        # Extract results
        tracks = []
        for track in session_tracker.tracks:
            if not track.is_confirmed():
                continue
                
            track_box = track.to_tlbr()  # [x1, y1, x2, y2] format
            
            tracks.append({
                'track_id': track.track_id,
                'class_id': track.class_id,
                'class_name': track.class_name,
                'box': track_box.tolist(),
                'confidence': track.confidence
            })
        
        session["frames_processed"] += 1
        session["last_update"] = time.time()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Clean up old sessions (sessions inactive for more than 30 minutes)
        current_time = time.time()
        for vid in list(tracking_sessions.keys()):
            if current_time - tracking_sessions[vid]["last_update"] > 1800:
                del tracking_sessions[vid]
        
        # Return tracking results
        return {
            "video_id": video_id,
            "frame_idx": data.frame_idx,
            "tracks": tracks,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error tracking objects: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset/{video_id}")
async def reset_tracking(video_id: str):
    if video_id in tracking_sessions:
        del tracking_sessions[video_id]
        return {"message": f"Tracking session {video_id} reset"}
    else:
        return {"message": f"No active tracking session for {video_id}"}

@app.get("/api/status/{video_id}")
async def get_status(video_id: str):
    if video_id in tracking_sessions:
        session = tracking_sessions[video_id]
        tracker = session["tracker"]
        results = tracker.get_results()
        
        return {
            "video_id": video_id,
            "frames_processed": session["frames_processed"],
            "last_update": session["last_update"],
            "vehicle_count": results["counts"]["vehicle_count"],
            "person_count": results["counts"]["person_count"],
            "tracked_ids": results["tracked_ids"]
        }
    else:
        raise HTTPException(status_code=404, detail=f"No active tracking session for {video_id}")

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "healthy", "active_sessions": len(tracking_sessions)}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("tracker_app:app", host="0.0.0.0", port=port, log_level="info") 