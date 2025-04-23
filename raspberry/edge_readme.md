# Edge Computing with Raspberry Pi

This documentation covers the integration of a Raspberry Pi as an edge computing resource for object detection and tracking in the video processing pipeline.

## Overview

Instead of running YOLO detection and DeepSORT tracking locally on the main machine, this implementation offloads these tasks to a Raspberry Pi that acts as an edge computing server. This approach offers several benefits:

- Distributes computational load
- Enables parallel processing
- Can be deployed closer to data sources
- Follows the same pattern as existing cloud providers (AWS, Azure, GCP)

## Architecture

```
┌──────────────────┐           ┌───────────────────┐
│                  │           │                   │
│   Main Machine   │  ────►    │   Raspberry Pi    │
│                  │   HTTP    │  Edge Computing   │
│   (ml/main.py)   │  ◄────    │      Server       │
│                  │           │                   │
└──────────────────┘           └───────────────────┘
```

The integration follows the same pattern used for cloud providers:
1. Main application extracts frames from video
2. Frames are sent to the Raspberry Pi server via HTTP
3. Raspberry Pi performs detection and tracking
4. Results are returned to the main application
5. Main application processes and displays the results

## Setup Instructions

### 1. Raspberry Pi Setup

1. Follow the initial setup in `setup.txt` to prepare your Raspberry Pi
2. Copy the files from the `raspberry` directory to your Raspberry Pi:
   ```
   scp -r raspberry/* ntp@raspberrypi.local:~/mestrado/
   ```
3. Install the required dependencies:
   ```
   ssh ntp@raspberrypi.local
   cd ~/mestrado
   source ~/mestrado/bin/activate
   pip install -r requirements.txt
   ```
4. Make the start script executable:
   ```
   chmod +x start_edge_server.sh
   ```
5. Start the edge server:
   ```
   ./start_edge_server.sh
   ```

### 2. Main Application Configuration

1. Set the environment variable to connect to the edge server:
   ```
   export EDGE_DEEPSORT_ENDPOINT="http://raspberrypi.local:5000"
   ```
   Or use the provided script:
   ```
   source raspberry/edge_env_setup.sh
   ```

2. Run the application with the edge provider:
   ```
   python ml/main.py --video your_video.mp4 --provider edge
   ```
   Or use the helper script:
   ```
   python ml/run_with_edge.py --video your_video.mp4
   ```

## Testing the Connection

To verify that your Raspberry Pi edge server is running and accessible:

```
curl http://raspberrypi.local:5000/healthcheck
```

You should receive a response like:
```json
{"status": "healthy", "active_sessions": 0}
```

For a more comprehensive test, you can use the test script:

```
# On the Raspberry Pi:
python test_edge_server.py --image test_image.jpg --url http://localhost:5000
```

## Edge Computing Server APIs

The edge server provides the following endpoints:

- `GET /` - Basic information about the server
- `POST /api/detect` - Object detection using YOLO
- `POST /api/track` - Object tracking using DeepSORT
- `POST /api/reset/<video_id>` - Reset a tracking session
- `GET /api/status/<video_id>` - Get status of a tracking session
- `GET /healthcheck` - Check server health

## Performance Considerations

- The Raspberry Pi has limited resources compared to the main machine or cloud servers
- Consider using a Raspberry Pi 4 with at least 4GB of RAM for better performance
- The YOLOv11n model is optimized for edge devices but still requires significant resources
- Resolution scaling and frame skipping can help improve performance

## Troubleshooting

If you encounter issues:

1. Verify that the Raspberry Pi is on the same network and reachable
2. Check that the EDGE_DEEPSORT_ENDPOINT environment variable is set correctly
3. Examine the logs on the Raspberry Pi for any errors
4. Ensure the Raspberry Pi has enough memory and CPU resources available
5. Try restarting the edge server
6. Verify that all required dependencies are installed 