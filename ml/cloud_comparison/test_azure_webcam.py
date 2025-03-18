import os
import cv2
import time
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
from dataclasses import dataclass
from typing import List

# Load environment variables
load_dotenv()

@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    label: str
    confidence: float

def draw_detections(frame, detections):
    """Draw bounding boxes and labels for all detections."""
    for det in detections:
        # Draw rectangle
        cv2.rectangle(frame, (det.x, det.y), (det.x + det.w, det.y + det.h), (0, 255, 0), 2)
        
        # Add label
        cv2.putText(frame, f"{det.label} ({det.confidence:.2f})", 
                   (det.x, det.y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def process_frame(client, frame):
    """Process a single frame with Azure Computer Vision."""
    # Encode frame to JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()
    
    # Create a BytesIO object from the bytes
    image_stream = BytesIO(img_bytes)
    
    # Detect objects
    result = client.detect_objects_in_stream(image_stream)
    
    # Convert results to our Detection class
    detections = []
    for obj in result.objects:
        det = Detection(
            x=obj.rectangle.x,
            y=obj.rectangle.y,
            w=obj.rectangle.w,
            h=obj.rectangle.h,
            label=obj.object_property,
            confidence=obj.confidence
        )
        detections.append(det)
    
    return detections

def main():
    # Create Azure Computer Vision client
    client = ComputerVisionClient(
        endpoint=os.getenv('AZURE_COMPUTER_VISION_ENDPOINT'),
        credentials=CognitiveServicesCredentials(os.getenv('AZURE_COMPUTER_VISION_KEY'))
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
    print("Note: Processing is rate-limited to stay within free tier (20 calls/minute)")
    
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
            
            # Check if we can make another API call (1 call per second, max 20 per minute)
            can_process = (
                (current_time - last_api_call).total_seconds() >= 1.0 and
                api_calls_per_minute < 20
            )
            
            display_frame = frame.copy()
            
            if can_process:
                try:
                    # Process frame
                    new_detections = process_frame(client, frame)
                    last_detections = new_detections  # Update last detections
                    last_api_call = current_time
                    api_calls_per_minute += 1
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
            
            # Draw the last known detections (whether from this frame or previous)
            display_frame = draw_detections(display_frame, last_detections)
            
            # Calculate and display FPS and API calls
            fps = processed_count / (time.time() - start_time) if processed_count > 0 else 0
            cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"API calls/min: {api_calls_per_minute}/20", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Azure Computer Vision Test', display_frame)
            
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
        print(f"Frames processed with Azure: {processed_count}")
        print(f"Average processing FPS: {processed_count/total_time:.1f}")
        print(f"Average capture FPS: {frame_count/total_time:.1f}")

if __name__ == "__main__":
    main()