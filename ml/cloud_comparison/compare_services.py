import os
import time
import cv2
import numpy as np
import boto3
from azure.ai.videoanalyzer import VideoAnalyzerClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Load environment variables
load_dotenv()

class VideoProcessor:
    def __init__(self):
        """Initialize video processor with different service clients."""
        # AWS Rekognition client
        self.rekognition = boto3.client(
            'rekognition',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        # Azure Video Analyzer client
        self.azure_credential = DefaultAzureCredential()
        self.video_analyzer = VideoAnalyzerClient(
            endpoint=os.getenv('AZURE_VIDEO_ANALYZER_ENDPOINT'),
            credential=self.azure_credential
        )
        
        # Results storage
        self.results = {
            'local': [],
            'aws': [],
            'azure': []
        }
    
    def process_local(self, frame):
        """Process frame using local YOLOv8 model."""
        # This will be implemented using your existing YOLOv8 implementation
        # For now, we'll just measure the time
        start_time = time.time()
        # TODO: Add your YOLOv8 processing here
        end_time = time.time()
        return end_time - start_time
    
    def process_aws(self, frame):
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
        
        end_time = time.time()
        return end_time - start_time, response
    
    def process_azure(self, frame):
        """Process frame using Azure Video Analyzer."""
        start_time = time.time()
        
        # Convert frame to bytes
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        
        # Call Azure Video Analyzer
        # Note: This is a simplified version. Azure Video Analyzer typically
        # requires more setup with IoT Hub and edge modules
        response = self.video_analyzer.analyze_frame(
            frame_data=img_bytes,
            features=['objectDetection']
        )
        
        end_time = time.time()
        return end_time - start_time, response
    
    def run_comparison(self, video_path, num_frames=100):
        """Run comparison test on video file."""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened() and frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process with each service
            local_time = self.process_local(frame)
            aws_time, aws_response = self.process_aws(frame)
            azure_time, azure_response = self.process_azure(frame)
            
            # Store results
            self.results['local'].append(local_time)
            self.results['aws'].append(aws_time)
            self.results['azure'].append(azure_time)
            
            frame_count += 1
            print(f"Processed frame {frame_count}/{num_frames}")
            
        cap.release()
    
    def analyze_results(self):
        """Analyze and plot comparison results."""
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Calculate statistics
        stats = df.describe()
        print("\nProcessing Time Statistics (seconds):")
        print(stats)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        df.boxplot()
        plt.title('Processing Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.savefig('comparison_results.png')
        plt.close()
        
        # Save detailed results
        with open('detailed_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)

def main():
    # Create output directory if it doesn't exist
    output_dir = Path('ml/cloud_comparison/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = VideoProcessor()
    
    # Run comparison
    video_path = "path/to/your/test/video.mp4"  # Update this path
    processor.run_comparison(video_path)
    
    # Analyze results
    processor.analyze_results()

if __name__ == "__main__":
    main() 