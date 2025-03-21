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
import tempfile
from pathlib import Path
import mlflow
from io import BytesIO
import shutil

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

        # Initialize rate limiting
        self.last_aws_call = datetime.now()
        self.last_azure_call = datetime.now()
        self.azure_calls_per_minute = 0
        self.azure_minute_start = datetime.now()
        
        # Create output directory for annotated frames
        self.output_dir = Path("cloud_comparison/annotated_frames")
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define relevant classes for detection
        self.relevant_classes = {
            'aws': {
                'vehicles': ['Car', 'Vehicle', 'Automobile', 'Truck', 'Van', 'Bus', 'Motorcycle', 'Transportation'],
                'people': ['Person', 'Human', 'People']
            },
            'azure': {
                'vehicles': ['car', 'vehicle', 'truck', 'van', 'bus', 'motorcycle'],
                'people': ['person', 'people']
            },
            'yolo': {
                'vehicles': [2, 3, 5, 7],  # COCO classes: car, motorcycle, bus, truck
                'people': [0]  # COCO class: person
            }
        }

    def get_run_metrics(self, run_id):
        """Get metrics from an MLFlow run."""
        run = mlflow.get_run(run_id)
        return run.data.metrics

    def can_call_aws(self):
        """Check if we can make an AWS API call (1 call per second)."""
        current_time = datetime.now()
        time_since_last_call = (current_time - self.last_aws_call).total_seconds()
        return time_since_last_call >= 1.0

    def can_call_azure(self):
        """Check if we can make an Azure API call (20 calls per minute)."""
        current_time = datetime.now()
        
        # Reset minute counter if needed
        if (current_time - self.azure_minute_start).total_seconds() >= 60:
            self.azure_calls_per_minute = 0
            self.azure_minute_start = current_time
        
        return self.azure_calls_per_minute < 20

    def extract_frames(self, video_path, num_frames=5):
        """Extract frames from video for cloud processing."""
        frames = []
        frame_paths = []
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Error opening video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise Exception(f"No frames found in video: {video_path}")
        
        # Calculate frame indices to extract
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        
        try:
            for idx in frame_indices:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # Save frame to temporary file
                    frame_path = os.path.join(temp_dir, f"frame_{idx}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames.append(frame)
                    frame_paths.append(frame_path)
        finally:
            cap.release()
        
        return frames, frame_paths, temp_dir
    
    def is_relevant_detection(self, detection, service_name):
        """Check if the detection is a vehicle or person."""
        class_name = detection.get('class')
        if class_name is None:
            return False

        relevant_classes = self.relevant_classes[service_name]
        
        # Convert class name to lowercase for string comparison
        if isinstance(class_name, str):
            class_name = class_name.lower()
            return any(class_name in vehicle_class.lower() for vehicle_class in relevant_classes['vehicles']) or \
                   any(class_name in person_class.lower() for person_class in relevant_classes['people'])
        else:  # YOLO uses numeric classes
            return class_name in relevant_classes['vehicles'] or class_name in relevant_classes['people']

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
                detection = {
                    "class": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                }
                if self.is_relevant_detection(detection, "yolo"):
                    detections.append(detection)
        
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
        # Check rate limit
        if not self.can_call_aws():
            raise Exception("AWS API rate limit exceeded (1 call per second)")

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
        
        # Update rate limiting
        self.last_aws_call = datetime.now()
        
        # Calculate metrics
        latency = (end_time - start_time) * 1000  # Convert to ms
        
        # Extract detection results - only include labels with bounding boxes (actual object detections)
        detections = []
        for label in response['Labels']:
            if label.get('Instances'):  # Only include if it has bounding box instances
                for instance in label['Instances']:
                    detections.append({
                        "class": label['Name'],
                        "confidence": label['Confidence'],
                        "bbox": [
                            instance['BoundingBox']['Left'],
                            instance['BoundingBox']['Top'],
                            instance['BoundingBox']['Left'] + instance['BoundingBox']['Width'],
                            instance['BoundingBox']['Top'] + instance['BoundingBox']['Height']
                        ]
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
        # Check rate limit
        if not self.can_call_azure():
            raise Exception("Azure API rate limit exceeded (20 calls per minute)")

        # Start MLFlow run
        run_id = self.tracker.start_detection_run(
            service_type="azure",
            model_name="computervision",
            input_type="image"
        )
        
        # Read image and create a BytesIO object
        with open(image_path, 'rb') as image:
            image_bytes = image.read()
            image_stream = BytesIO(image_bytes)
            image_stream.seek(0)  # Reset stream position to start
        
        # Process image
        start_time = time.time()
        try:
            response = self.azure_client.detect_objects_in_stream(image_stream)
        except Exception as e:
            print(f"Azure API error: {str(e)}")
            raise
        end_time = time.time()
        
        # Update rate limiting
        self.last_azure_call = datetime.now()
        self.azure_calls_per_minute += 1
        
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

    def draw_detections(self, image, detections, service_name, frame_number):
        """Draw bounding boxes and labels on the image and save it."""
        img = image.copy()
        height, width = img.shape[:2]
        
        # Define colors for different services
        colors = {
            "aws": (0, 255, 0),  # Green for AWS
            "azure": (255, 0, 0),  # Blue for Azure
            "yolo": (0, 0, 255)   # Red for YOLO
        }
        color = colors.get(service_name, (255, 255, 255))
        
        for detection in detections:
            # Get normalized coordinates
            bbox = detection["bbox"]
            if isinstance(bbox[0], float) and bbox[0] <= 1.0:  # Normalized coordinates
                x1, y1, x2, y2 = [
                    int(bbox[0] * width),
                    int(bbox[1] * height),
                    int(bbox[2] * width),
                    int(bbox[3] * height)
                ]
            else:  # Absolute coordinates
                x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{frame_number}_{service_name}_{timestamp}.jpg"
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), img)
        return str(output_path)

    def process_video_frames(self, service_name, frame_paths):
        """Process multiple frames with a given service and aggregate results."""
        total_latency = 0
        total_detections = 0
        total_confidence = 0
        all_detections = []
        processed_frames = 0
        saved_frame_paths = []
        
        for frame_idx, frame_path in enumerate(frame_paths):
            try:
                # Read the original frame
                frame = cv2.imread(frame_path)
                if frame is None:
                    print(f"Error reading frame: {frame_path}")
                    continue
                
                if service_name == "aws":
                    if not self.can_call_aws():
                        print(f"Skipping AWS frame due to rate limit")
                        continue
                    results, run_id = self.process_image_aws(frame_path)
                    # Only count actual object detections with bounding boxes
                    detections = []
                    for label in results['Labels']:
                        if label.get('Instances'):
                            for instance in label['Instances']:
                                detection = {
                                    "class": label['Name'],
                                    "confidence": label['Confidence'] / 100.0,  # Normalize to 0-1
                                    "bbox": [
                                        instance['BoundingBox']['Left'],
                                        instance['BoundingBox']['Top'],
                                        instance['BoundingBox']['Left'] + instance['BoundingBox']['Width'],
                                        instance['BoundingBox']['Top'] + instance['BoundingBox']['Height']
                                    ]
                                }
                                if self.is_relevant_detection(detection, "aws"):
                                    detections.append(detection)
                    
                elif service_name == "azure":
                    if not self.can_call_azure():
                        print(f"Skipping Azure frame due to rate limit")
                        continue
                    results, run_id = self.process_image_azure(frame_path)
                    detections = []
                    for obj in results.objects:
                        detection = {
                            "class": obj.object_property,
                            "confidence": obj.confidence,
                            "bbox": [
                                obj.rectangle.x / frame.shape[1],  # Normalize coordinates
                                obj.rectangle.y / frame.shape[0],
                                (obj.rectangle.x + obj.rectangle.w) / frame.shape[1],
                                (obj.rectangle.y + obj.rectangle.h) / frame.shape[0]
                            ]
                        }
                        if self.is_relevant_detection(detection, "azure"):
                            detections.append(detection)
                
                # Only save frame if it contains relevant detections
                if detections:
                    saved_path = self.draw_detections(frame, detections, service_name, frame_idx)
                    saved_frame_paths.append(saved_path)
                
                total_detections += len(detections)
                total_confidence += sum(d["confidence"] for d in detections)
                
                # Get metrics from MLFlow for this run
                metrics = self.get_run_metrics(run_id)
                total_latency += metrics.get("latency_ms", 0)
                
                all_detections.extend(detections)
                processed_frames += 1
                
                # Add delay to respect rate limits
                if service_name == "aws":
                    time.sleep(1.0)  # 1 second delay for AWS
                elif service_name == "azure" and processed_frames % 3 == 0:
                    time.sleep(1.0)  # Add delay every 3 frames for Azure to stay under 20/minute
                    
            except Exception as e:
                print(f"Error processing frame with {service_name}: {str(e)}")
                continue
        
        if processed_frames == 0:
            return {
                "latency": 0,
                "detection_count": 0,
                "avg_detections_per_frame": 0,
                "frames_processed": 0,
                "confidence": 0,
                "detections": [],
                "cost": 0,
                "saved_frame_paths": []
            }
        
        # Calculate metrics
        avg_latency = total_latency / processed_frames
        avg_confidence = (total_confidence / total_detections) if total_detections > 0 else 0
        
        # Calculate cost based on service
        if service_name == "aws":
            # AWS Rekognition: $1 per 1000 images
            cost = processed_frames * (1.0 / 1000)
        elif service_name == "azure":
            # Azure Computer Vision: $1 per 1000 transactions
            cost = processed_frames * (1.0 / 1000)
        else:
            cost = 0
        
        return {
            "latency": avg_latency,
            "detection_count": total_detections,
            "avg_detections_per_frame": total_detections / processed_frames,
            "frames_processed": processed_frames,
            "confidence": avg_confidence,
            "detections": all_detections,
            "cost": cost,
            "saved_frame_paths": saved_frame_paths
        }
    
    def compare_services(self, video_path):
        """Compare all services on the same video."""
        # Extract frames from video
        frames, frame_paths, temp_dir = self.extract_frames(video_path)
        
        try:
            # Process with YOLO (can handle video directly)
            yolo_results, yolo_run_id = self.process_image_yolo(video_path)
            yolo_metrics = self.get_run_metrics(yolo_run_id)
            
            # Process frames with AWS and Azure
            aws_results = self.process_video_frames("aws", frame_paths)
            azure_results = self.process_video_frames("azure", frame_paths)
            
            return {
                "latency": {
                    "yolo": yolo_metrics.get("latency_ms", 0),
                    "aws": aws_results["latency"],
                    "azure": azure_results["latency"]
                },
                "detections": {
                    "yolo": yolo_metrics.get("objects_detected", 0),
                    "aws": aws_results["detection_count"],
                    "azure": azure_results["detection_count"]
                },
                "frames_processed": {
                    "yolo": "all",  # YOLO processes all frames
                    "aws": aws_results["frames_processed"],
                    "azure": azure_results["frames_processed"]
                },
                "avg_detections_per_frame": {
                    "yolo": yolo_metrics.get("objects_detected", 0) / len(frames) if frames else 0,
                    "aws": aws_results["avg_detections_per_frame"],
                    "azure": azure_results["avg_detections_per_frame"]
                },
                "confidence": {
                    "yolo": yolo_metrics.get("avg_confidence", 0),
                    "aws": aws_results["confidence"],
                    "azure": azure_results["confidence"]
                },
                "costs": {
                    "yolo": 0.0,  # Local processing
                    "aws": aws_results["cost"],
                    "azure": azure_results["cost"]
                },
                "saved_frames": {
                    "aws": aws_results.get("saved_frame_paths", []),
                    "azure": azure_results.get("saved_frame_paths", [])
                }
            }
        finally:
            # Clean up temporary files
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except Exception as e:
                    print(f"Warning: Failed to remove temporary frame {frame_path}: {e}")
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to remove temporary directory {temp_dir}: {e}")

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
    print("Detection Count:", results["detections"])
    print("Average Confidence:", results["confidence"])

if __name__ == "__main__":
    main() 