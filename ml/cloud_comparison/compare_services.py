import os
import time
import cv2
import numpy as np
import boto3
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from io import BytesIO
import yaml
import sys
import argparse
from pathlib import Path

# Add the ml directory to Python path
ml_dir = Path(__file__).parent.parent
sys.path.append(str(ml_dir))

from models.yolo_model import YOLODetector

# Load environment variables
load_dotenv()

@dataclass
class Detection:
    name: str
    confidence: float
    bounding_box: dict

class VideoProcessor:
    def __init__(self):
        """Initialize video processor with different service clients."""
        # Load YOLO configuration
        with open('ml/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize YOLO model
        self.yolo = YOLODetector(config)
        
        # AWS Rekognition client
        self.rekognition = boto3.client(
            'rekognition',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-2')
        )
        
        # Azure Computer Vision client
        self.azure_client = ComputerVisionClient(
            endpoint=os.getenv('AZURE_COMPUTER_VISION_ENDPOINT'),
            credentials=CognitiveServicesCredentials(os.getenv('AZURE_COMPUTER_VISION_KEY'))
        )
        
        # Results storage
        self.results = {
            'yolo': {'times': [], 'detections': []},
            'aws': {'times': [], 'detections': []},
            'azure': {'times': [], 'detections': []}
        }
    
    def process_yolo(self, frame) -> tuple[float, List[Detection]]:
        """Process frame using YOLO model."""
        start_time = time.time()
        
        # Process frame with YOLO
        results, _ = self.yolo.process_frame(frame)
        
        # Convert YOLO results to our Detection format
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            # Convert to normalized coordinates (0-1)
            height, width = frame.shape[:2]
            det = Detection(
                name=results.names[cls],
                confidence=conf,
                bounding_box={
                    'Left': x1 / width,
                    'Top': y1 / height,
                    'Width': (x2 - x1) / width,
                    'Height': (y2 - y1) / height
                }
            )
            detections.append(det)
        
        end_time = time.time()
        return end_time - start_time, detections
    
    def process_aws(self, frame) -> tuple[float, List[Detection]]:
        """Process frame using AWS Rekognition."""
        start_time = time.time()
        
        # Convert frame to bytes
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        
        # Call AWS Rekognition
        response = self.rekognition.detect_labels(
            Image={'Bytes': img_bytes},
            MaxLabels=10,
            MinConfidence=70
        )
        
        # Convert response to our Detection format
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
        
        end_time = time.time()
        return end_time - start_time, detections
    
    def process_azure(self, frame) -> tuple[float, List[Detection]]:
        """Process frame using Azure Computer Vision."""
        start_time = time.time()
        
        # Convert frame to bytes
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        
        # Create a BytesIO object from the bytes
        image_stream = BytesIO(img_bytes)
        
        # Call Azure Computer Vision
        result = self.azure_client.detect_objects_in_stream(image_stream)
        
        # Convert response to our Detection format
        detections = []
        for obj in result.objects:
            det = Detection(
                name=obj.object_property,
                confidence=obj.confidence,
                bounding_box={
                    'Left': obj.rectangle.x,
                    'Top': obj.rectangle.y,
                    'Width': obj.rectangle.w,
                    'Height': obj.rectangle.h
                }
            )
            detections.append(det)
        
        end_time = time.time()
        return end_time - start_time, detections
    
    def run_comparison(self, video_path: str, num_frames: int = 50, delay: float = 1.0, confidence: float = 70.0):
        """Run comparison test on video file."""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        print(f"Starting comparison with {num_frames} frames...")
        print(f"Video file: {video_path}")
        print(f"Delay between frames: {delay} seconds")
        print(f"Confidence threshold: {confidence}")
        
        while cap.isOpened() and frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process with each service
            yolo_time, yolo_detections = self.process_yolo(frame)
            aws_time, aws_detections = self.process_aws(frame)
            azure_time, azure_detections = self.process_azure(frame)
            
            # Store results
            self.results['yolo']['times'].append(yolo_time)
            self.results['yolo']['detections'].append(yolo_detections)
            self.results['aws']['times'].append(aws_time)
            self.results['aws']['detections'].append(aws_detections)
            self.results['azure']['times'].append(azure_time)
            self.results['azure']['detections'].append(azure_detections)
            
            frame_count += 1
            print(f"Processed frame {frame_count}/{num_frames}")
            
            # Add a small delay to respect rate limits
            time.sleep(delay)
            
        cap.release()
    
    def analyze_results(self):
        """Analyze and plot comparison results."""
        # Convert timing results to DataFrame
        df = pd.DataFrame({
            'YOLO': self.results['yolo']['times'],
            'AWS': self.results['aws']['times'],
            'Azure': self.results['azure']['times']
        })
        
        # Calculate statistics
        stats = df.describe()
        print("\nProcessing Time Statistics (seconds):")
        print(stats)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        df.boxplot()
        plt.title('Processing Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.savefig('ml/cloud_comparison/results/comparison_results.png')
        plt.close()
        
        # Calculate detection statistics
        yolo_detections = [len(dets) for dets in self.results['yolo']['detections']]
        aws_detections = [len(dets) for dets in self.results['aws']['detections']]
        azure_detections = [len(dets) for dets in self.results['azure']['detections']]
        
        print("\nDetection Statistics:")
        print(f"YOLO - Average detections per frame: {np.mean(yolo_detections):.2f}")
        print(f"AWS - Average detections per frame: {np.mean(aws_detections):.2f}")
        print(f"Azure - Average detections per frame: {np.mean(azure_detections):.2f}")
        
        # Save detailed results
        results_dict = {
            'timing': {
                'yolo': self.results['yolo']['times'],
                'aws': self.results['aws']['times'],
                'azure': self.results['azure']['times']
            },
            'detections': {
                'yolo': yolo_detections,
                'aws': aws_detections,
                'azure': azure_detections
            }
        }
        
        with open('ml/cloud_comparison/results/detailed_results.json', 'w') as f:
            json.dump(results_dict, f, indent=4)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compare object detection services (YOLO, AWS, Azure)')
    parser.add_argument('--frames', type=int, default=50,
                      help='Number of frames to process (default: 50)')
    parser.add_argument('--video', type=str, default='ml/data/test_videos/test.mp4',
                      help='Path to the test video file (default: ml/data/test_videos/test.mp4)')
    parser.add_argument('--delay', type=float, default=1.0,
                      help='Delay between frames in seconds (default: 1.0)')
    parser.add_argument('--confidence', type=float, default=70.0,
                      help='Minimum confidence threshold for detections (default: 70.0)')
    parser.add_argument('--output-dir', type=str, default='ml/cloud_comparison/results',
                      help='Directory to save results (default: ml/cloud_comparison/results)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = VideoProcessor()
    
    # Run comparison with command line arguments
    print(f"Starting comparison with {args.frames} frames...")
    print(f"Video file: {args.video}")
    print(f"Delay between frames: {args.delay} seconds")
    print(f"Confidence threshold: {args.confidence}")
    
    processor.run_comparison(
        video_path=args.video,
        num_frames=args.frames,
        delay=args.delay,
        confidence=args.confidence
    )
    
    # Analyze results
    processor.analyze_results()

if __name__ == "__main__":
    main() 