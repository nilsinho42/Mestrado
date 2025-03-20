import os
import time
import cv2
import numpy as np
from datetime import datetime
import boto3
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from ultralytics import YOLO
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow_tracking import DetectionTracker

class CloudServiceComparator:
    def __init__(self):
        """Initialize cloud services and MLFlow tracking."""
        # Initialize MLFlow tracker
        self.tracker = DetectionTracker()
        
        # Initialize AWS Rekognition
        self.rekognition = boto3.client('rekognition')
        
        # Initialize Azure Computer Vision
        self.azure_client = ComputerVisionClient(
            endpoint=os.getenv('AZURE_ENDPOINT'),
            credentials=CognitiveServicesCredentials(os.getenv('AZURE_KEY'))
        )
        
        # Initialize YOLO
        self.yolo_model = YOLO('yolov8n.pt')
    
    def process_image_yolo(self, image_path):
        """Process image using YOLO and track metrics."""
        # Start MLFlow run
        run_id = self.tracker.start_detection_run(
            service_type="yolo",
            model_name="yolov8n",
            input_type="image"
        )
        
        # Process image
        start_time = time.time()
        results = self.yolo_model(image_path)
        end_time = time.time()
        
        # Calculate metrics
        latency = (end_time - start_time) * 1000  # Convert to ms
        fps = 1 / (end_time - start_time)
        
        # Extract detection results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                detections.append({
                    "class": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })
        
        # Log metrics
        self.tracker.log_detection_metrics(
            run_id,
            metrics={
                "fps": fps,
                "latency_ms": latency,
                "objects_detected": len(detections),
                "avg_confidence": np.mean([d["confidence"] for d in detections]) if detections else 0
            }
        )
        
        # Log results
        self.tracker.log_detection_results(run_id, detections)
        
        return results, run_id
    
    def process_image_aws(self, image_path):
        """Process image using AWS Rekognition and track metrics."""
        # Start MLFlow run
        run_id = self.tracker.start_detection_run(
            service_type="aws",
            model_name="rekognition",
            input_type="image"
        )
        
        # Read image
        with open(image_path, 'rb') as image:
            image_bytes = image.read()
        
        # Process image
        start_time = time.time()
        response = self.rekognition.detect_labels(Image={'Bytes': image_bytes})
        end_time = time.time()
        
        # Calculate metrics
        latency = (end_time - start_time) * 1000  # Convert to ms
        
        # Extract detection results
        detections = []
        for label in response['Labels']:
            detections.append({
                "class": label['Name'],
                "confidence": label['Confidence'],
                "instances": label.get('Instances', [])
            })
        
        # Log metrics
        self.tracker.log_detection_metrics(
            run_id,
            metrics={
                "latency_ms": latency,
                "objects_detected": len(detections),
                "avg_confidence": np.mean([d["confidence"] for d in detections]) if detections else 0
            }
        )
        
        # Log cloud cost
        self.tracker.log_cloud_cost(
            run_id,
            service_type="aws",
            cost_details={
                "api_call_cost": 0.0001,  # Example cost per API call
                "total_cost": 0.0001
            }
        )
        
        # Log results
        self.tracker.log_detection_results(run_id, detections)
        
        return response, run_id
    
    def process_image_azure(self, image_path):
        """Process image using Azure Computer Vision and track metrics."""
        # Start MLFlow run
        run_id = self.tracker.start_detection_run(
            service_type="azure",
            model_name="computervision",
            input_type="image"
        )
        
        # Read image
        with open(image_path, 'rb') as image:
            image_bytes = image.read()
        
        # Process image
        start_time = time.time()
        response = self.azure_client.detect_objects_in_stream(image_bytes)
        end_time = time.time()
        
        # Calculate metrics
        latency = (end_time - start_time) * 1000  # Convert to ms
        
        # Extract detection results
        detections = []
        for obj in response.objects:
            detections.append({
                "class": obj.object_property,
                "confidence": obj.confidence,
                "bbox": [obj.rectangle.x, obj.rectangle.y, 
                        obj.rectangle.x + obj.rectangle.w,
                        obj.rectangle.y + obj.rectangle.h]
            })
        
        # Log metrics
        self.tracker.log_detection_metrics(
            run_id,
            metrics={
                "latency_ms": latency,
                "objects_detected": len(detections),
                "avg_confidence": np.mean([d["confidence"] for d in detections]) if detections else 0
            }
        )
        
        # Log cloud cost
        self.tracker.log_cloud_cost(
            run_id,
            service_type="azure",
            cost_details={
                "api_call_cost": 0.0001,  # Example cost per API call
                "total_cost": 0.0001
            }
        )
        
        # Log results
        self.tracker.log_detection_results(run_id, detections)
        
        return response, run_id
    
    def compare_services(self, image_path):
        """Compare all services on the same image."""
        # Process with all services
        yolo_results, yolo_run_id = self.process_image_yolo(image_path)
        aws_results, aws_run_id = self.process_image_aws(image_path)
        azure_results, azure_run_id = self.process_image_azure(image_path)
        
        # Compare latency
        latency_comparison = self.tracker.compare_services(
            [yolo_run_id, aws_run_id, azure_run_id],
            "latency_ms"
        )
        
        # Compare detection counts
        count_comparison = self.tracker.compare_services(
            [yolo_run_id, aws_run_id, azure_run_id],
            "objects_detected"
        )
        
        # Compare confidence
        confidence_comparison = self.tracker.compare_services(
            [yolo_run_id, aws_run_id, azure_run_id],
            "avg_confidence"
        )
        
        return {
            "latency": latency_comparison,
            "detection_count": count_comparison,
            "confidence": confidence_comparison
        }

def main():
    """Main function to run service comparison."""
    # Initialize comparator
    comparator = CloudServiceComparator()
    
    # Test image path
    image_path = "test_image.jpg"
    
    # Run comparison
    results = comparator.compare_services(image_path)
    
    # Print results
    print("\nService Comparison Results:")
    print("Latency (ms):", results["latency"])
    print("Detection Count:", results["detection_count"])
    print("Average Confidence:", results["confidence"])

if __name__ == "__main__":
    main() 