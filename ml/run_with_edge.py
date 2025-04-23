#!/usr/bin/env python3
"""
Helper script to run the main video processing pipeline with the edge provider.
This sets the necessary environment variables and calls main.py.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    """Parse arguments and run the main.py script with edge provider."""
    parser = argparse.ArgumentParser(description='Run video processing with Raspberry Pi Edge server')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='./data/object_detection/images', 
                      help='Output directory for frames')
    parser.add_argument('--expected_vehicles', type=int, default=0, 
                      help='Expected number of vehicles in the video')
    parser.add_argument('--expected_people', type=int, default=0, 
                      help='Expected number of people in the video')
    parser.add_argument('--edge_url', type=str, default='http://raspberrypi.local:5000',
                      help='URL of the Edge server (Raspberry Pi)')
    args = parser.parse_args()
    
    # Set environment variable for edge endpoint
    os.environ['EDGE_DEEPSORT_ENDPOINT'] = args.edge_url
    print(f"Using Edge server at: {args.edge_url}")
    
    # Build command for main.py
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "main.py"),
        "--provider", "edge",
        "--video", args.video,
        "--output", args.output,
        "--expected_vehicles", str(args.expected_vehicles),
        "--expected_people", str(args.expected_people)
    ]
    
    # Run the command
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running main.py: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main() 