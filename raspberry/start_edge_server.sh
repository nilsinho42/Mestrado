#!/bin/bash

# Activate virtual environment
source ~/mestrado/bin/activate

# Set port for the server (default: 5000)
export PORT=5000

# Download YOLO model if not present
if [ ! -f "yolov11n.pt" ]; then
    echo "Downloading YOLOv11n model..."
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov11n.pt
fi

# Create tmp directory if it doesn't exist
mkdir -p tmp

# Start the edge server
echo "Starting Edge Computing Server on port $PORT..."
python edge_server.py 