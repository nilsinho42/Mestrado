import os
import time
import uuid
import logging
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import detector and tracker modules
from detector import YOLODetector
from tracker import DeepSortTracker

# Create temporary directory for frames
TEMP_DIR = Path("./tmp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

# Initialize detector and tracker
detector = YOLODetector(model_path="yolov11n.pt", confidence_threshold=0.50)
tracker = DeepSortTracker(model_path="yolov11n.pt", max_iou_distance=0.9, max_age=70, n_init=5)

# Store active tracking sessions
tracking_sessions = {}

@app.route('/')
def home():
    return jsonify({"message": "Raspberry Pi Edge Computing Server for Object Detection and Tracking"})

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    """Endpoint for object detection only."""
    try:
        start_time = time.time()
        
        # Get base64 encoded image from request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
            
        # Decode image
        try:
            img_bytes = base64.b64decode(data['image'])
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image")
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return jsonify({"error": f"Invalid image format: {str(e)}"}), 400
        
        # Process with detector
        detections = detector.process_image(img)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return results
        return jsonify({
            "detections": detections,
            "processing_time": processing_time
        })
        
    except Exception as e:
        logger.error(f"Error detecting objects: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/track', methods=['POST'])
def track_objects():
    """Endpoint for object tracking (similar to cloud implementation)."""
    try:
        start_time = time.time()
        
        # Get data from request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "Invalid request format"}), 400
            
        # Get video_id and frame_idx
        video_id = data.get('video_id') or str(uuid.uuid4())
        frame_idx = data.get('frame_idx', 0)
        
        # Create tracking session if it doesn't exist
        if video_id not in tracking_sessions:
            tracking_sessions[video_id] = {
                "tracker": DeepSortTracker(model_path="yolov11n.pt", max_iou_distance=0.9, max_age=70, n_init=5),
                "frames_processed": 0,
                "last_update": time.time()
            }
        
        # Decode image
        try:
            img_bytes = base64.b64decode(data['image'])
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image")
                
            # Save image temporarily for debugging (optional)
            frame_path = TEMP_DIR / f"{video_id}_{frame_idx}.jpg"
            cv2.imwrite(str(frame_path), img)
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return jsonify({"error": f"Invalid image format: {str(e)}"}), 400
        
        # We ignore provided detections in YOLO mode since YOLO does its own detection
        # Track objects using the session tracker (which now uses YOLO)
        session = tracking_sessions[video_id]
        session_tracker = session["tracker"]
        
        # Track objects - with YOLO we only need to pass the frame
        tracks = session_tracker.update([], frame=img)
        
        # Extract results
        tracks_result = []
        for track in tracks:
            # Just use the track object as-is since it's simplified
            tracks_result.append({
                'track_id': track.track_id,
                'class_id': track.class_id,
                'class_name': track.class_name,
                'box': track.bbox,
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
        
        # Return tracking results (match format of cloud implementation)
        return jsonify({
            "video_id": video_id,
            "frame_idx": frame_idx,
            "tracks": tracks_result,
            "processing_time": processing_time
        })
        
    except Exception as e:
        logger.error(f"Error tracking objects: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/reset/<video_id>', methods=['POST'])
def reset_tracking(video_id):
    """Reset a tracking session."""
    if video_id in tracking_sessions:
        del tracking_sessions[video_id]
        return jsonify({"message": f"Tracking session {video_id} reset"})
    else:
        return jsonify({"message": f"No active tracking session for {video_id}"})

@app.route('/api/status/<video_id>', methods=['GET'])
def get_status(video_id):
    """Get status of a tracking session."""
    if video_id in tracking_sessions:
        session = tracking_sessions[video_id]
        return jsonify({
            "video_id": video_id,
            "frames_processed": session["frames_processed"],
            "last_update": session["last_update"]
        })
    else:
        return jsonify({"error": f"No active tracking session for {video_id}"}), 404

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy", 
        "active_sessions": len(tracking_sessions)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False) 