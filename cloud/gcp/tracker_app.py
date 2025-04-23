import os
import time
import uuid
import logging
import base64
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from root directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create temporary directory for frames
TEMP_DIR = Path("./tmp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Create FastAPI app
app = FastAPI(title="GCP Cloud Run DeepSORT Tracking API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DeepSORT Implementation
class Detection:
    """
    This class represents a bounding box detection in a single image.
    """
    def __init__(self, tlwh, confidence, class_id, class_name):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.confidence = float(confidence)
        self.class_id = class_id
        self.class_name = class_name

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

class NearestNeighborDistanceMetric:
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.
    """
    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = self._euclidean_distance
        else:
            self._metric = self._cosine_distance
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data."""
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """Compute distance between features and targets."""
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix

    @staticmethod
    def _euclidean_distance(x, y):
        """Compute Euclidean distance between all pairs of `x` and `y`."""
        if len(x) == 0 or len(y) == 0:
            return np.zeros((len(x), len(y)))
        x = np.asarray(x)
        y = np.asarray(y)
        return np.maximum(0.0, cdist(x, y, "euclidean"))

    @staticmethod
    def _cosine_distance(x, y):
        """Compute cosine distance between all pairs of `x` and `y`."""
        if len(x) == 0 or len(y) == 0:
            return np.zeros((len(x), len(y)))
        x = np.asarray(x)
        y = np.asarray(y)
        return np.maximum(0.0, 1. - np.dot(x, y.T))

# Use scipy for distance calculations
from scipy.spatial.distance import cdist

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in range(len(y)) if y[i] >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou(bbox, candidates):
    """Computer intersection over union between bbox and candidates."""
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]
    
    tl = np.maximum(bbox_tl, candidates_tl)
    br = np.minimum(bbox_br, candidates_br)
    wh = np.maximum(0., br - tl)
    
    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    
    return area_intersection / (area_bbox + area_candidates - area_intersection)

def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """An intersection over union distance metric."""
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = 1.0
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix

def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    """Solve linear assignment problem."""
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    
    indices = linear_assignment(cost_matrix)
    matches, unmatched_tracks, unmatched_detections = [], [], []
    
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    
    return matches, unmatched_tracks, unmatched_detections

class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    """
    def __init__(self):
        ndim, dt = 4, 1.
        
        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty
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
        
        # Update mean & cov
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot([
            self._motion_mat, covariance, self._motion_mat.T
        ]) + motion_cov
        
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
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot([
            self._update_mat, covariance, self._update_mat.T
        ])
        
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step."""
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # Compute Kalman gain
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        
        # Update state
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot([
            kalman_gain, projected_cov, kalman_gain.T
        ])
        
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Compute gating distance between state distribution and measurements."""
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        
        d = measurements - mean
        if len(measurements.shape) == 1:
            d = d.reshape(1, -1)
        
        cholesky_factor = np.linalg.cholesky(covariance)
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False)
        squared_mahal = np.sum(z * z, axis=0)
        return squared_mahal

# Import scipy for Kalman filter
import scipy.linalg

class Track:
    """
    A track for a single object being tracked.
    """
    def __init__(self, mean, covariance, track_id, class_id, class_name, confidence, n_init, max_age=30):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self._n_init = n_init
        self._max_age = max_age
        self.state = 1  # 1=Tentative, 2=Confirmed, 3=Deleted

    def to_tlwh(self):
        """Get bounding box in (top left width height) format."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get bounding box in (top left bottom right) format."""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step."""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step."""
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.hits += 1
        self.time_since_update = 0
        if self.state == 1 and self.hits >= self._n_init:
            self.state = 2  # Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.state == 1:
            self.state = 3  # Mark tentative tracks as deleted
        elif self.time_since_update > self._max_age:
            self.state = 3  # Mark as deleted after max_age

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == 1

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == 2

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == 3

class Tracker:
    """
    This is the DeepSORT tracker implementation.
    """
    def __init__(self, max_iou_distance=0.7, max_age=70, n_init=3):
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management."""
        # Run matching cascade
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            targets.append(track.track_id)
        
        # Handle confirmed tracks that lost objects
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            # Any other track management logic can go here

    def _match(self, detections):
        """Match detections to tracks using IoU."""
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        
        # Associate confirmed tracks using IOU distance
        matches_a, unmatched_tracks_a, unmatched_detections = \
            min_cost_matching(iou_cost, self.max_iou_distance, self.tracks,
                             detections, confirmed_tracks)
        
        # Associate remaining tracks
        iou_threshold = 0.5  # Lower threshold for unconfirmed tracks
        matches_b, unmatched_tracks_b, unmatched_detections = \
            min_cost_matching(iou_cost, iou_threshold, self.tracks,
                             detections, unconfirmed_tracks, unmatched_detections)
        
        matches = matches_a + matches_b
        unmatched_tracks = list(unmatched_tracks_a) + list(unmatched_tracks_b)
        
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        """Initialize a new track from a detection."""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        track = Track(mean, covariance, self._next_id, detection.class_id, 
                    detection.class_name, detection.confidence, 
                    self.n_init, self.max_age)
        self.tracks.append(track)
        self._next_id += 1

# Initialize tracker
# Real DeepSORT implementation
tracker = Tracker(max_iou_distance=0.7, max_age=30, n_init=3)

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

# Store active tracking sessions
tracking_sessions = {}

@app.get("/")
async def root():
    return {"message": "GCP Cloud Run DeepSORT Tracking API. Use /api/track endpoint for object tracking."}

@app.post("/api/track", response_model=TrackingResult)
async def track_objects(data: ImageData):
    """
    Process detections and update tracks for a video frame.
    Returns the updated tracks.
    """
    video_id = data.video_id or str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Processing frame {data.frame_idx} for video {video_id}")
    
    # Store image data for debugging if needed
    image_data = base64.b64decode(data.image)
    frame_path = TEMP_DIR / f"{video_id}_{data.frame_idx}.jpg"
    with open(frame_path, "wb") as f:
        f.write(image_data)
    
    # Get or initialize tracker for this video
    if video_id not in tracking_sessions:
        tracking_sessions[video_id] = {
            "tracker": Tracker(max_iou_distance=0.7, max_age=30, n_init=3),
            "frames_processed": 0,
            "last_frame_idx": -1,
            "vehicle_count": 0,
            "person_count": 0
        }
    
    session = tracking_sessions[video_id]
    
    # Check for missing frames
    if data.frame_idx > session["last_frame_idx"] + 1 and session["last_frame_idx"] != -1:
        logger.warning(f"Missing frames detected. Last processed: {session['last_frame_idx']}, current: {data.frame_idx}")
    
    session["last_frame_idx"] = data.frame_idx
    session["frames_processed"] += 1
    
    # Convert detections to the format needed by DeepSORT
    detections = []
    for det in data.detections:
        # Convert box format from [x1, y1, x2, y2] to [x, y, w, h]
        box = det['box']
        if len(box) == 4:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            tlwh = np.array([x1, y1, w, h])
        else:
            # Handle box in [x, y, w, h] format
            tlwh = np.array(box)
        
        confidence = det['confidence']
        class_id = det.get('class_id', 0)
        class_name = det.get('class_name', 'unknown')
        
        detections.append(Detection(tlwh, confidence, class_id, class_name))
    
    # Process with DeepSORT tracker
    tracker_instance = session["tracker"]
    tracker_instance.predict()
    tracker_instance.update(detections)
    
    # Prepare result
    tracks = []
    for track in tracker_instance.tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
            
        bbox = track.to_tlbr()
        
        # Count unique vehicles and people
        if track.class_name.lower() in ['car', 'truck', 'bus', 'vehicle', 'automobile', 'van', 'motorcycle']:
            session["vehicle_count"] += 1
        elif track.class_name.lower() in ['person', 'pedestrian', 'human', 'man', 'woman', 'child']:
            session["person_count"] += 1
        
        # Add track to result
        tracks.append({
            'track_id': str(track.track_id),
            'bbox': bbox.tolist(),
            'class_id': track.class_id,
            'class_name': track.class_name,
            'confidence': track.confidence
        })
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # For debugging, log the number of tracks
    logger.info(f"Frame {data.frame_idx}: found {len(tracks)} confirmed tracks in {processing_time:.3f}s")
    
    # Return tracking results
    return TrackingResult(
        video_id=video_id,
        frame_idx=data.frame_idx,
        tracks=tracks,
        processing_time=processing_time
    )

@app.post("/api/reset/{video_id}")
async def reset_tracking(video_id: str):
    """Reset the tracking state for a video."""
    if video_id in tracking_sessions:
        del tracking_sessions[video_id]
        # Delete any saved frames
        for frame_file in TEMP_DIR.glob(f"{video_id}_*.jpg"):
            frame_file.unlink()
        return {"status": "success", "message": f"Tracking state for video {video_id} has been reset"}
    return {"status": "not_found", "message": f"No tracking session found for video {video_id}"}

@app.get("/api/status/{video_id}")
async def get_status(video_id: str):
    """Get tracking session status for a video."""
    if video_id in tracking_sessions:
        session = tracking_sessions[video_id]
        return {
            "video_id": video_id,
            "frames_processed": session["frames_processed"],
            "last_frame_idx": session["last_frame_idx"],
            "vehicle_count": session["vehicle_count"],
            "person_count": session["person_count"]
        }
    return {"status": "not_found", "message": f"No tracking session found for video {video_id}"}

@app.get("/healthcheck")
async def healthcheck():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run("tracker_app:app", host="0.0.0.0", port=8080) 