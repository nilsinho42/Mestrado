import numpy as np
import logging
import time
from scipy.spatial.distance import cdist
import scipy.linalg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Detection:
    """
    This class represents a bounding box detection in a single image.
    """
    def __init__(self, bbox, confidence, class_id, class_name, frame_number=None):
        self.tlwh = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], dtype=np.float64)
        self.confidence = float(confidence)
        self.class_id = class_id
        self.class_name = class_name
        self.frame_number = frame_number

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

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.
        
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        
        if len(d.shape) == 1:
            d = d.reshape(1, -1)
        
        cholesky_factor = np.linalg.cholesky(covariance)
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

def linear_assignment(cost_matrix):
    """Solve linear assignment problem using scipy."""
    try:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))
    except ImportError:
        logger.error("Required scipy.optimize.linear_sum_assignment not available")
        return np.array([])

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

class Track:
    """
    A track class for holding a single tracked object state.
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
        self.state = 1  # tentative
        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, min y, max x,
        max y)`.
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0
        if self.state == 1 and self.hits >= self._n_init:
            self.state = 2  # confirmed

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
    def __init__(self, max_iou_distance=0.7, max_age=30, n_init=3):
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1
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

    def update(self, detections):
        """Perform measurement update and track management.
        
        Args:
            detections: List of detections (dict or Detection objects)
            
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
                    class_name=det['class_name']
                ))
            else:
                detection_objects = detections
                break
                
        # Predict step
        self.predict()

        # Run matching cascade
        matches, unmatched_tracks, unmatched_detections = self._match(detection_objects)

        # Update track set
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detection_objects[detection_idx])
            
            # Update counters for metrics
            track = self.tracks[track_idx]
            class_name = track.class_name.lower()
            
            # Check if this is a vehicle or person
            if 'car' in class_name or 'truck' in class_name or 'bus' in class_name or 'vehicle' in class_name:
                if track.track_id not in self.tracked_ids['vehicle']:
                    self.tracked_ids['vehicle'].add(track.track_id)
                    self.vehicle_count += 1
            elif 'person' in class_name or 'pedestrian' in class_name:
                if track.track_id not in self.tracked_ids['person']:
                    self.tracked_ids['person'].add(track.track_id)
                    self.person_count += 1
            
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            
        for detection_idx in unmatched_detections:
            self._initiate_track(detection_objects[detection_idx])
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        return self.tracks

    def _match(self, detections):
        """Match tracks and detections."""
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using IoU
        matches_a, unmatched_tracks_a, unmatched_detections = \
            min_cost_matching(
                iou_cost, self.max_iou_distance, self.tracks,
                detections, confirmed_tracks)

        # Associate remaining tracks (unconfirmed) with remaining detections
        # using IoU
        iou_track_candidates = unconfirmed_tracks
        matches_b, unmatched_tracks_b, unmatched_detections = \
            min_cost_matching(
                iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        """Initialize a new track from a detection."""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, detection.class_id,
            detection.class_name, detection.confidence, self.n_init, 
            max_age=self.max_age))
        
        # Update counters for metrics
        class_name = detection.class_name.lower()
        if 'car' in class_name or 'truck' in class_name or 'bus' in class_name or 'vehicle' in class_name:
            if self._next_id not in self.tracked_ids['vehicle']:
                self.tracked_ids['vehicle'].add(self._next_id)
                self.vehicle_count += 1
        elif 'person' in class_name or 'pedestrian' in class_name:
            if self._next_id not in self.tracked_ids['person']:
                self.tracked_ids['person'].add(self._next_id)
                self.person_count += 1
                
        self._next_id += 1
        
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