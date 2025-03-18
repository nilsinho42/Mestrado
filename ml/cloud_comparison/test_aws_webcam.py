import os
import cv2
import time
import boto3
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Detection:
    name: str
    confidence: float
    bounding_box: dict  # Contains x, y, width, height

def process_frame(rekognition, frame):
    """Process a single frame with AWS Rekognition."""
    # Encode frame to JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()
    
    # Detect objects
    response = rekognition.detect_labels(
        Image={'Bytes': img_bytes}
    )
    
    # Convert results to our Detection class
    detections = []
    for label in response['Labels']:
        if 'Instances' in label:
            for instance in label['Instances']:
                if 'BoundingBox' in instance:
                    det = Detection(
                        name=label['Name'],
                        confidence=label['Confidence'],
                        bounding_box=instance['BoundingBox']
                    )
                    detections.append(det)
    
    return detections

def draw_detections(frame, detections):
    """Draw bounding boxes and labels for all detections."""
    height, width = frame.shape[:2]
    
    for det in detections:
        # Convert normalized coordinates to pixel coordinates
        x = int(det.bounding_box['Left'] * width)
        y = int(det.bounding_box['Top'] * height)
        w = int(det.bounding_box['Width'] * width)
        h = int(det.bounding_box['Height'] * height)
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add label
        label = f"{det.name} ({det.confidence:.1f}%)"
        cv2.putText(frame, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    # Create AWS Rekognition client
    rekognition = boto3.client(
        'rekognition',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize counters and rate limiting
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    last_api_call = datetime.now()
    api_calls_per_minute = 0
    minute_start = datetime.now()
    last_detections = []  # Store last successful detections
    
    print("Press 'q' to quit")
    print("Note: Processing is rate-limited to stay within free tier (5,000 images/month)")
    print("      Current limit: 1 call per second")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame")
                break
            
            current_time = datetime.now()
            
            # Reset minute counter if needed
            if (current_time - minute_start).total_seconds() >= 60:
                api_calls_per_minute = 0
                minute_start = current_time
            
            # Check if we can make another API call (1 call per second)
            can_process = (current_time - last_api_call).total_seconds() >= 1.0
            
            display_frame = frame.copy()
            
            if can_process:
                try:
                    # Process frame
                    new_detections = process_frame(rekognition, frame)
                    last_detections = new_detections  # Update last detections
                    last_api_call = current_time
                    api_calls_per_minute += 1
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
            
            # Draw the last known detections
            display_frame = draw_detections(display_frame, last_detections)
            
            # Calculate and display FPS and API calls
            fps = processed_count / (time.time() - start_time) if processed_count > 0 else 0
            cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"API calls/min: {api_calls_per_minute}", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('AWS Rekognition Test', display_frame)
            
            frame_count += 1
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        total_time = time.time() - start_time
        print(f"\nTest completed:")
        print(f"Total frames captured: {frame_count}")
        print(f"Frames processed with AWS: {processed_count}")
        print(f"Average processing FPS: {processed_count/total_time:.1f}")
        print(f"Average capture FPS: {frame_count/total_time:.1f}")

if __name__ == "__main__":
    main() 