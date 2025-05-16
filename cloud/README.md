# Cloud YOLO Tracking API

This module provides a FastAPI application that implements object tracking using Ultralytics YOLO. It's optimized for processing videos frame-by-frame, maintaining object identities across frames.

- docker build -t nperes/deepsort-tracker:latest .
- docker push nperes/deepsort-tracker:latest
- docker tag nperes/deepsort-tracker:latest public.ecr.aws/v3x5y9w4/ntp/mestrado:latest
- docker push public.ecr.aws/v3x5y9w4/ntp/mestrado:latest

## Features

- Uses Ultralytics YOLO11n model for detection and tracking
- Supports both BoT-SORT and ByteTrack trackers
- Custom tracker configuration via YAML file
- Session-based tracking (maintains state across API calls)
- Automatic model download
- Timeouts and error handling for reliable operation

## Requirements

- ultralytics>=8.3.2 (required for YOLO11n support)
- fastapi
- uvicorn
- opencv-python
- numpy
- pyyaml
- python-dotenv
- requests

## Usage

### Running the Server

```bash
# From the project root directory
cd cloud
uvicorn tracker_app_yolo:app --host 0.0.0.0 --port 8080
```

The server will automatically download the YOLO11n model the first time it runs.

### API Endpoints

#### Root `/`

- **Method**: GET
- **Description**: Returns basic information about the API
- **Response**: JSON with API details

#### Health Check `/healthcheck`

- **Method**: GET
- **Description**: Simple health check endpoint
- **Response**: JSON with status and active session count

#### Track Objects `/api/track`

- **Method**: POST
- **Description**: Process a single frame for object tracking
- **Request Body**:
  ```json
  {
    "image": "base64_encoded_image",
    "frame_idx": 0,
    "detections": [],
    "video_id": "optional_video_id"
  }
  ```
- **Response**: JSON with tracking results
  ```json
  {
    "video_id": "video_id",
    "frame_idx": 0,
    "tracks": [
      {
        "track_id": 1,
        "box": [x1, y1, x2, y2],
        "confidence": 0.98,
        "class_id": 0,
        "class_name": "person"
      }
    ],
    "processing_time": 0.25
  }
  ```

#### Configure Tracker `/api/configure/{video_id}`

- **Method**: POST
- **Description**: Configure tracking parameters for a session
- **Request Body**:
  ```json
  {
    "model_path": "yolo11n.pt",
    "tracker_type": "botsort",
    "conf": 0.25,
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "with_reid": false
  }
  ```
- **Response**: JSON with configuration status

#### Reset Tracking `/api/reset/{video_id}`

- **Method**: POST
- **Description**: Reset a tracking session
- **Response**: JSON with reset status

#### Get Status `/api/status/{video_id}`

- **Method**: GET
- **Description**: Get the status of a tracking session
- **Response**: JSON with session details, vehicle and person counts, and active tracks

## Tracker Configuration

The tracker can be configured using a YAML file located at `cloud/tracker_config.yaml`. The default configuration is based on the BoT-SORT algorithm and includes:

```yaml
# Tracker configuration file
tracker_type: botsort  # tracker type, ['botsort', 'bytetrack']

# Threshold settings
track_high_thresh: 0.25  # threshold for the first association
track_low_thresh: 0.1  # threshold for the second association
new_track_thresh: 0.25  # threshold for init new track if the detection does not match any tracks
track_buffer: 30  # buffer to calculate the time when to remove tracks
match_thresh: 0.8  # threshold for matching tracks
fuse_score: True  # Whether to fuse confidence scores with the iou distances before matching

# BoT-SORT settings
gmc_method: sparseOptFlow  # method of global motion compensation

# ReID model related thresh
proximity_thresh: 0.5  # minimum IoU for valid match with ReID
appearance_thresh: 0.8  # minimum appearance similarity for ReID
with_reid: False  # Whether to use ReID
model: auto  # uses native features if detector is YOLO else yolo11n-cls.pt
```

## Differences from the Original DeepSORT Implementation

This tracker uses the built-in tracking capabilities of Ultralytics YOLO instead of the custom DeepSORT implementation. The main differences are:

1. **YOLO Integration**: Uses YOLO's built-in detector for better accuracy and performance
2. **Multiple Tracker Support**: Supports both BoT-SORT and ByteTrack algorithms
3. **Simplified Implementation**: Leverages Ultralytics' optimized tracking code
4. **Enhanced Configuration**: More flexible configuration options
5. **Improved Performance**: Generally faster and more accurate tracking

## Example Code

### Python Client Example

```python
import requests
import base64
import cv2
import json
import numpy as np

# API endpoint
api_url = "http://localhost:8080/api/track"

# Function to encode frame as base64
def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# Function to send a frame for tracking
def track_frame(frame, frame_idx, video_id=None):
    # Encode the frame
    encoded_frame = encode_frame(frame)
    
    # Create request payload
    payload = {
        "image": encoded_frame,
        "frame_idx": frame_idx,
        "detections": [],  # Let YOLO handle detection
        "video_id": video_id or "test_video"
    }
    
    # Send request
    response = requests.post(api_url, json=payload)
    
    # Return tracking results
    return response.json()

# Example usage with a video file
cap = cv2.VideoCapture('your_video.mp4')
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get tracking results
    results = track_frame(frame, frame_idx)
    
    # Draw tracks on the frame
    for track in results["tracks"]:
        x1, y1, x2, y2 = map(int, track["box"])
        track_id = track["track_id"]
        class_name = track["class_name"]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"{class_name} #{track_id}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
```

## License

This code is provided under the same license as the main project. 