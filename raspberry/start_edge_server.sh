#!/bin/bash

# Activate virtual environment
source ~/mestrado/bin/activate

# Set port for the server (default: 5000)
export PORT=5000

# Download YOLO model if not present

# Start the edge server
echo "Starting Edge Computing Server on port $PORT..."
python edge_server.py 