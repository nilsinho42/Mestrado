"""
Cloud YOLO Tracking API

This module provides a FastAPI application that implements object tracking using
Ultralytics YOLO. It's optimized for processing videos frame-by-frame, maintaining
object identities across frames.

Key features:
- Uses Ultralytics YOLO11n model for detection and tracking
- Supports both BoT-SORT and ByteTrack trackers
- Custom tracker configuration via YAML file
- Session-based tracking (maintains state across API calls)
- Automatic model download
- Timeouts and error handling for reliable operation

Requirements:
- Python 3.11
- ultralytics
- fastapi
- opencv-python
- numpy
- pyyaml
"""

import os
import time
import uuid
import logging
import base64
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
from dotenv import load_dotenv
import concurrent.futures
from contextlib import contextmanager
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load environment variables from root directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Cloud YOLO Tracking API", 
    version="1.0.0",
    description="API for object tracking using Ultralytics YOLO",
    docs_url="/docs",
    redoc_url="/redoc"
)

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

# Define models for API requests and responses
class ImageData(BaseModel):
    """
    Data model for tracking requests.
    """
    image: str  # Base64 encoded image
    frame_idx: int
    detections: List[Dict[str, Any]]
    video_id: Optional[str] = None
    provider: Optional[str] = None

class TrackingResult(BaseModel):
    """
    Data model for tracking results.
    """
    video_id: Optional[str]
    frame_idx: int
    tracks: List[Dict[str, Any]]
    processing_time: float

# Define a custom TimeoutError to avoid conflicts with built-in TimeoutError
class CustomTimeoutError(Exception):
    """Exception raised when a function call times out."""
    pass

# Define a cross-platform timeout function
T = TypeVar('T')

def run_with_timeout(func: Callable[..., T], timeout_sec: int, *args, **kwargs) -> T:
    """
    Run a function with a timeout.
    
    Args:
        func: The function to run
        timeout_sec: The maximum time to allow the function to run, in seconds
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function
        
    Raises:
        CustomTimeoutError: If the function takes longer than timeout_sec seconds
        Any other exception raised by the function
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            raise CustomTimeoutError(f"Operation timed out after {timeout_sec} seconds")

# Context manager for timeouts
@contextmanager
def timeout(seconds):
    """
    Context manager for running code with a timeout.
    
    Args:
        seconds: Maximum time in seconds to allow the code to run
        
    Raises:
        CustomTimeoutError: If the code takes longer than the specified timeout
    """
    class TimeoutContext:
        def __call__(self, func, *args, **kwargs):
            return run_with_timeout(func, seconds, *args, **kwargs)
            
    yield TimeoutContext()

# class YOLOTracker:
#     """
#     Wrapper class for Ultralytics YOLO tracking.
#     """
#     def __init__(self, model_path="yolo11n.pt", conf=0.5):
#         """
#         Initialize the YOLO tracker.
        
#         Args:
#             model_path: Path to the YOLO model weights
#             tracker_config: Tracker configuration file path or preset name ("botsort.yaml"/"bytetrack.yaml")
#             conf: Confidence threshold for detections
#         """
#         self.model_path = model_path
#         self.conf = conf
        
#         # Load the model
#         try:
#             self.model = YOLO(self.model_path)
#             logger.info(f"Loaded YOLO model: {self.model_path}")
#         except Exception as e:
#             logger.error(f"Failed to load YOLO model: {e}")
#             raise
            
#         # Initialize counters for metrics
#         self.vehicle_count = 0
#         self.person_count = 0
#         self.tracked_ids = {
#             'vehicle': set(),
#             'person': set()
#         }
        
#         # Most recently seen tracks
#         self.current_tracks = []
    
#     def _is_vehicle(self, class_name):
#         """Check if the class name represents a vehicle."""
#         vehicle_classes = [
#             'car', 'vehicle', 'automobile', 'truck', 'van', 'bus', 'motorcycle', 'bicycle', 
#             'transportation', 'taxi', 'ambulance', 'police car', 'suv', 'motorbike',
#             'Car', 'Vehicle', 'Automobile', 'Truck', 'Van', 'Bus', 'Motorcycle', 'Bicycle',
#             'Transportation', 'Taxi', 'Ambulance', 'Police Car', 'SUV', 'Motorbike'
#         ]
#         return any(vehicle_class.lower() in class_name.lower() for vehicle_class in vehicle_classes)
        
#     def _is_person(self, class_name):
#         """Check if the class name represents a person."""
#         person_classes = [
#             'person', 'human', 'people', 'pedestrian', 'man', 'woman', 'child', 'baby',
#             'Person', 'Human', 'People', 'Pedestrian', 'Man', 'Woman', 'Child', 'Baby'
#         ]
#         return any(person_class.lower() in class_name.lower() for person_class in person_classes)
    
#     def _update_counts(self, track_id, class_name):
#         """Update vehicle and person counts based on track class."""
#         if self._is_vehicle(class_name):
#             if track_id not in self.tracked_ids['vehicle']:
#                 self.tracked_ids['vehicle'].add(track_id)
#                 self.vehicle_count += 1
                
#         elif self._is_person(class_name):
#             if track_id not in self.tracked_ids['person']:
#                 self.tracked_ids['person'].add(track_id)
#                 self.person_count += 1
    
    
#     def track(self, image):
#         """
#         Perform tracking on an image with optional pre-computed detections.
        
#         Args:
#             image: The input image (numpy array)
            
#         Returns:
#             A list of tracks
#         """
#         # Run the tracker
#         try:
#             results = self.model.track(
#                 source=image,
#                 persist=True,  # Maintain track IDs across frames
#                 conf=self.conf,
#                 verbose=False
#             )
            
#             # Extract tracks from results
#             tracks = []
            
#             if results and len(results) > 0:
#                 result = results[0]  # Get the first result
                
#                 # Check if tracking information is available
#                 if hasattr(result, 'boxes') and hasattr(result.boxes, 'id') and result.boxes.id is not None:
#                     boxes = result.boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format
#                     track_ids = result.boxes.id.int().cpu().numpy()  # Get track IDs
#                     confs = result.boxes.conf.cpu().numpy()  # Get confidences
#                     cls_ids = result.boxes.cls.int().cpu().numpy()  # Get class IDs
                    
#                     # Get class names
#                     class_names = [result.names[c] for c in cls_ids]
                    
#                     # Create track objects
#                     for i in range(len(track_ids)):
#                         track_id = int(track_ids[i])
#                         box = boxes[i].tolist()  # Convert to list for JSON serialization
#                         confidence = float(confs[i])  # Convert to native float
#                         class_id = int(cls_ids[i])   # Convert to native int
#                         class_name = class_names[i]
                        
#                         # Update counters
#                         self._update_counts(track_id, class_name)
                        
#                         tracks.append({
#                             'track_id': track_id,
#                             'box': box,
#                             'confidence': confidence,
#                             'class_id': class_id,
#                             'class_name': class_name
#                         })
            
#             # Update current tracks
#             self.current_tracks = tracks
            
#             return tracks
            
#         except Exception as e:
#             logger.error(f"Error in YOLO tracking: {e}")
#             return []
    
#     def get_results(self):
#         """Get counts and other metrics."""
#         return {
#             "counts": {
#                 "vehicle_count": self.vehicle_count,
#                 "person_count": self.person_count
#             },
#             "tracked_ids": {
#                 "vehicle": list(self.tracked_ids['vehicle']),
#                 "person": list(self.tracked_ids['person'])
#             }
#         }

   

@app.get("/")
async def root():
    """Root endpoint, provides basic information about the API."""
    return {
        "message": "Cloud YOLO Tracking API",
        "version": "1.0.0",
        "docs": "/docs",
        "model": "yolo11n.pt",
        "tracker": "botsort.yaml",
    }


@app.post("/api/track", response_model=TrackingResult)
async def track_objects(data: ImageData):
    """
    Process tracking for a single frame.
    This endpoint has safeguards to prevent excessive processing time.
    """
    # Set a processing start time to track overall execution time
    overall_start_time = time.time()
    max_processing_time = 20  # Maximum seconds to process a single frame
    
    try:
        start_time = time.time()
        
        # Log session and request data for debugging
        video_id = data.video_id or str(uuid.uuid4())
        logger.info(f"[{video_id}] Received track request for frame {data.frame_idx} with {len(data.detections)} detections")
        
        # Check if the session exists and last update wasn't too long ago (might be abandoned)
        if video_id in tracking_sessions:
            last_update = tracking_sessions[video_id]["last_update"]
            if time.time() - last_update > 600:  # 10 minutes
                logger.warning(f"[{video_id}] Session seems abandoned (no updates in 10 minutes). Resetting.")
                del tracking_sessions[video_id]
        
        # Create tracking session if it doesn't exist
        if video_id not in tracking_sessions:
            vehicle_tracker = DeepSort(embedder_gpu=False, half=False, bgr=True, n_init=4, max_age=70) 
            people_tracker = DeepSort(embedder='torchreid', embedder_gpu=False, half=False, bgr=True, n_init=15, max_age=150)

            tracking_sessions[video_id] = {
                "vehicle_tracker": vehicle_tracker,
                "people_tracker": people_tracker,
                "frames_processed": 0,
                "last_update": time.time(),
                "last_frame_idx": None,
                "current_processing": False,
                "current_tracks": []
            }
            logger.info(f"[{video_id}] Created new YOLO tracking session with custom tracker config")
        
        # Check if we're already processing this frame or if it's a duplicate request
        session = tracking_sessions[video_id]
        if session["current_processing"]:
            logger.warning(f"[{video_id}] Another request is currently being processed. Returning empty response.")
            return {
                "video_id": video_id,
                "frame_idx": data.frame_idx,
                "tracks": [],
                "processing_time": 0.0
            }
        
        # Check if this is a duplicate or outdated frame request
        if session["last_frame_idx"] is not None:
            if data.frame_idx <= session["last_frame_idx"]:
                logger.warning(f"[{video_id}] Received outdated or duplicate frame {data.frame_idx} (last was {session['last_frame_idx']})")
                # We'll still process it, but log a warning
        
        # Mark this session as currently processing
        session["current_processing"] = True
        
        try:
            # Decode image from base64
            img_bytes = base64.b64decode(data.image)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image")
            
            # Prepare detections for vehicle and people trackers
            vehicle_detections = []
            people_detections = []
            
            # Log incoming detections and prepare them for the trackers
            for i, det in enumerate(data.detections):
                det_data = {
                    "index": i,
                    "class_name": det.get('class_name', 'unknown'),
                    "confidence": det.get('confidence', 0),
                    "box": det.get('box', [])
                }
                logger.debug(f"[{video_id}] Detection {i}: {det_data}")
                
                confidence = det.get('confidence')
                class_name = det.get('class_name', 'unknown')
                box = det.get('box', [])
                
                # Ensure box is in correct format (xywh for DeepSort)
                if data.provider != 'azure' and len(box) == 4:
                    # Convert from xyxy to xywh format
                    x1, y1, x2, y2 = box
                    box_xywh = [x1, y1, x2 - x1, y2 - y1]
                else:
                    # Azure provider already sends in correct format or box is invalid
                    box_xywh = box
                
                # Add to appropriate tracker list
                if class_name.lower() == 'vehicle':
                    vehicle_detections.append((box_xywh, confidence, class_name))
                else:
                    people_detections.append((box_xywh, confidence, class_name))
  
            track_dicts = []  # Will hold all tracks for response
            
            # Check if we're exceeding max processing time
            if time.time() - overall_start_time > max_processing_time:
                logger.warning(f"[{video_id}] Processing time limit exceeded before tracker update. Aborting.")
                raise CustomTimeoutError("Processing timeout exceeded")

            # Process vehicle detections
            vehicle_tracks = []
            if vehicle_detections:
                try:
                    vehicle_tracks = run_with_timeout(
                        session["vehicle_tracker"].update_tracks,
                        10,  # 10 second timeout
                        vehicle_detections,
                        frame=img
                    )
                except CustomTimeoutError:
                    logger.error(f"[{video_id}] Vehicle tracker update timed out")
                except Exception as e:
                    logger.error(f"[{video_id}] Error during vehicle tracker update: {e}")
                    
            # Process people detections
            people_tracks = []
            if people_detections:
                try:
                    people_tracks = run_with_timeout(
                        session["people_tracker"].update_tracks,
                        10,  # 10 second timeout
                        people_detections,
                        frame=img
                    )
                except CustomTimeoutError:
                    logger.error(f"[{video_id}] People tracker update timed out")
                except Exception as e:
                    logger.error(f"[{video_id}] Error during people tracker update: {e}")
            
            # Combine all tracks
            all_tracks = vehicle_tracks + people_tracks
            
            # Convert DeepSort Track objects into dictionaries for API response
            for track in all_tracks:
                if not track.is_confirmed():
                    continue
                    
                # Get bounding box in [x1, y1, x2, y2] format
                try:
                    bbox = track.to_ltrb() 
                    box_list = bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
                    
                    # Create a track dictionary with all necessary fields
                    track_dict = {
                        'track_id': int(track.track_id),
                        'box': box_list,
                        'confidence': float(getattr(track, 'det_conf', 1.0)),
                        'class_id': int(getattr(track, 'class_id', 0)),
                        'class_name': str(getattr(track, 'det_class', 'unknown'))
                    }
                    track_dicts.append(track_dict)
                except Exception as track_err:
                    logger.error(f"[{video_id}] Error formatting track: {track_err}")
            
            # Store formatted tracks in session
            session["current_tracks"] = track_dicts
                    
            # Update session info
            session["frames_processed"] += 1
            session["last_update"] = time.time()
            session["last_frame_idx"] = data.frame_idx
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Clean up old sessions (those inactive for more than 30 minutes)
            current_time = time.time()
            for vid in list(tracking_sessions.keys()):
                if vid != video_id and current_time - tracking_sessions[vid]["last_update"] > 1800:  # 30 minutes
                    del tracking_sessions[vid]
                    logger.info(f"Removed inactive tracking session: {vid}")
            
            # Return tracking results
            return {
                "video_id": video_id,
                "frame_idx": data.frame_idx,
                "tracks": track_dicts,  # Use track_dicts instead of tracks
                "processing_time": processing_time
            }
        
        finally:
            # Always mark the session as not processing anymore, even if an error occurred
            if video_id in tracking_sessions:
                tracking_sessions[video_id]["current_processing"] = False
    
    except Exception as e:
        logger.error(f"Error tracking objects: {e}", exc_info=True)
        # Make sure we clean up the processing flag in case of error
        if video_id in tracking_sessions:
            tracking_sessions[video_id]["current_processing"] = False
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthcheck")
async def healthcheck():
    """Health check endpoint."""
    return {"status": "healthy", "active_sessions": len(tracking_sessions)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("tracker_app:app", host="0.0.0.0", port=port, log_level="info") 