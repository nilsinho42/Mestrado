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
from .tracking import IoUTracker, Detection, TrackedObject

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

        # Initialize Azure rate limiting
        self.azure_calls_per_minute = 0
        self.azure_minute_start = datetime.now()
        
        # Initialize service latency tracking
        self.service_latency = {
            'yolo': 0.0,
            'aws': 0.0,
            'azure': 0.0
        }
        
        # Initialize object trackers
        self.trackers = {
            'aws': IoUTracker(),
            'azure': IoUTracker(),
            'yolo': IoUTracker()
        }
        
        # Create output directory for annotated frames
        self.output_dir = Path("cloud_comparison/annotated_frames")
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define relevant classes for detection
        self.relevant_classes = {
            'aws': {
                'vehicles': ['Car', 'Vehicle', 'Automobile', 'Truck', 'Van', 'Bus', 'Motorcycle', 'Transportation', 'Taxi', 'Ambulance', 'Police Car'],
                'people': ['Person', 'Human', 'People', 'Pedestrian', 'Man', 'Woman', 'Child']
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

    def get_azure_wait_time(self):
        """Calculate how long to wait for next Azure call."""
        current_time = datetime.now()
        time_since_minute_start = (current_time - self.azure_minute_start).total_seconds()
        
        if self.azure_calls_per_minute >= 20:
            # Wait until the current minute window ends
            return max(60 - time_since_minute_start, 0)
        else:
            # Standard 3-second spacing between calls
            return 3.0

    def can_call_azure(self):
        """Check if we can make an Azure API call (20 calls per minute)."""
        current_time = datetime.now()
        
        # Reset minute counter if needed
        if (current_time - self.azure_minute_start).total_seconds() >= 60:
            self.azure_calls_per_minute = 0
            self.azure_minute_start = current_time
            return True
        
        return self.azure_calls_per_minute < 20

    def extract_frames(self, video_path, fps_reduction_factor=10):
        """Extract frames from video for cloud processing.
        
        Args:
            video_path: Path to the video file
            fps_reduction_factor: Factor by which to reduce FPS (e.g., 10 means 30fps -> 3fps)
        """
        frames = []
        frame_paths = []
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Error opening video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        if total_frames == 0:
            raise Exception(f"No frames found in video: {video_path}")
        
        # Calculate target FPS and frame interval
        target_fps = original_fps / fps_reduction_factor
        frame_interval = fps_reduction_factor  # Take 1 frame every N frames
        
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        
        try:
            frame_count = 0
            frames_sampled = 0
            frame_indices = []
            
            while frame_count < total_frames:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                
                if ret:
                    # Save frame to temporary file
                    frame_path = os.path.join(temp_dir, f"frame_{frames_sampled}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames.append(frame)
                    frame_paths.append(frame_path)
                    frame_indices.append(frame_count)
                    frames_sampled += 1
                
                # Move to next frame based on interval
                frame_count += frame_interval
            
            video_info = {
                "total_frames": total_frames,
                "fps": original_fps,
                "target_fps": target_fps,
                "duration": duration,
                "sampled_frames": frames_sampled,
                "frame_indices": frame_indices
            }
            
            print(f"Video info: {total_frames} total frames at {original_fps} FPS")
            print(f"Sampling at 1/{fps_reduction_factor} of original FPS ({target_fps:.1f} FPS)")
            print(f"Sampled {frames_sampled} frames at indices: {frame_indices}")
            
            return frames, frame_paths, temp_dir, video_info
            
        finally:
            cap.release()

    def is_relevant_detection(self, detection, service_name):
        """Check if the detection is a vehicle or person."""
        class_name = detection.get('class')
        if class_name is None:
            return False

        relevant_classes = self.relevant_classes[service_name]
        
        # For YOLO, which uses numeric class IDs
        if service_name == 'yolo':
            try:
                # If it's already an integer, use it directly
                if isinstance(class_name, int):
                    class_id = class_name
                # If it's a string representing a number, convert it
                elif isinstance(class_name, str) and class_name.isdigit():
                    class_id = int(class_name)
                # If it's a string class name, try to map it to YOLO class ID
                else:
                    # Common class name mappings for YOLO COCO dataset
                    class_map = {
                        'person': 0,
                        'car': 2,
                        'motorcycle': 3,
                        'bus': 5,
                        'truck': 7
                    }
                    class_id = class_map.get(class_name.lower(), -1)
                
                return class_id in relevant_classes['vehicles'] or class_id in relevant_classes['people']
            except (ValueError, AttributeError):
                return False
        
        # For AWS and Azure, which use string class names
        if isinstance(class_name, str):
            class_name = class_name.lower()
            return any(class_name in vehicle_class.lower() for vehicle_class in relevant_classes['vehicles']) or \
                   any(class_name in person_class.lower() for person_class in relevant_classes['people'])
        
        return False

    def process_image_yolo(self, video_path):
        """Process video with YOLO."""
        start_time = time.time()
        
        # Extract frames for tracking
        frames, frame_paths, temp_dir, video_info = self.extract_frames(video_path)
        tracker = self.trackers['yolo']
        
        total_detections = 0
        total_confidence = 0
        saved_frame_paths = []
        frames_processed = 0
        
        try:
            # Load YOLO model
            model = YOLO('yolov8n.pt')
            
            # Process each frame
            for frame_idx, frame in enumerate(frames):
                # Run YOLO detection
                results = model(frame, verbose=False)[0]
                frame_detections = []
                frames_processed += 1
                
                # Process detections
                for r in results.boxes.data.tolist():
                    x1, y1, x2, y2, confidence, class_id = r
                    class_name = results.names[int(class_id)]
                    
                    detection = {
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [x1/frame.shape[1], y1/frame.shape[0], x2/frame.shape[1], y2/frame.shape[0]]
                    }
                    
                    if self.is_relevant_detection(detection, "yolo"):
                        frame_detections.append(Detection(
                            frame_number=frame_idx,
                            class_name=detection["class"],
                            confidence=detection["confidence"],
                            bbox=detection["bbox"],
                            service="yolo"
                        ))
                
                # Update tracker with new detections
                active_tracks = tracker.update(frame_detections)
                
                # Convert tracked objects back to detection format for visualization
                tracked_detections = [
                    {
                        "class": track.class_name,
                        "confidence": track.confidence_history[-1],
                        "bbox": track.bbox_history[-1],
                        "track_id": track.id
                    }
                    for track in active_tracks
                ]
                
                # Only save frame if it contains relevant detections
                if tracked_detections:
                    saved_path = self.draw_detections(frame, tracked_detections, "yolo", frame_idx)
                    saved_frame_paths.append(saved_path)
                    total_detections += len(tracked_detections)
                
                total_confidence += sum(d.confidence for d in frame_detections)
            
            # Get unique objects from tracker
            unique_objects = tracker.get_unique_objects()
            
            # Calculate average confidence
            avg_confidence = total_confidence / total_detections if total_detections > 0 else 0
            
            end_time = time.time()
            self.service_latency['yolo'] = end_time - start_time
            
            return {
                "latency": self.service_latency['yolo'],
                "detection_count": total_detections,
                "unique_objects": len(unique_objects),
                "avg_confidence": avg_confidence,
                "saved_frames": saved_frame_paths,
                "frames_processed": frames_processed
            }, None
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)

    def process_image_aws(self, image_path):
        """Process image using AWS Rekognition and track metrics."""
        # Check rate limit
        if not self.can_call_azure():
            raise Exception("Azure API rate limit exceeded (20 calls per minute)")

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
        latency = end_time - start_time  # Store in seconds for consistency
        latency_ms = latency * 1000  # Convert to ms for MLFlow tracking
        
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
                "latency_ms": latency_ms,
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
        
        return {"Labels": response['Labels'], "latency": latency}, run_id
    
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
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Read image into memory first to ensure it's complete
                with open(image_path, 'rb') as image:
                    image_bytes = image.read()
                
                # Create a fresh BytesIO object for each attempt
                image_stream = BytesIO(image_bytes)
                
                # Process image
                start_time = time.time()
                response = self.azure_client.detect_objects_in_stream(image_stream)
                end_time = time.time()
                
                # If we get here, the call was successful
                break
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Azure API error after {max_retries} attempts: {str(e)}")
                    raise
                print(f"Azure API error (attempt {retry_count}/{max_retries}): {str(e)}")
                time.sleep(1)  # Wait a second before retrying
                continue
            finally:
                # Ensure we close the stream
                if 'image_stream' in locals():
                    image_stream.close()
        
        # Calculate metrics
        latency = end_time - start_time  # Store in seconds for consistency
        latency_ms = latency * 1000  # Convert to ms for MLFlow tracking
        
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
                "latency_ms": latency_ms,
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
        
        # Update Azure call counter after successful processing
        self.azure_calls_per_minute += 1
        
        return {"objects": response.objects, "latency": latency}, run_id

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
            
            # Add label with confidence and track ID
            track_id = detection.get("track_id", "")
            label = f"{detection['class']} ({track_id[-4:]}): {detection['confidence']:.2f}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{frame_number}_{service_name}_{timestamp}.jpg"
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), img)
        return str(output_path)

    def process_video_frames(self, service_name, frame_paths):
        """Process frames with specified cloud service."""
        total_detections = 0
        total_confidence = 0
        saved_frame_paths = []
        frames_processed = 0
        total_latency = 0.0
        
        # Initialize tracker for this service
        tracker = self.trackers[service_name]
        
        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Failed to read frame {frame_path}")
                continue
            
            frame_detections = []
            
            try:
                if service_name == "aws":
                    # AWS: No rate limiting needed, just process the frame
                    results, run_id = self.process_image_aws(frame_path)
                    total_latency += results["latency"]
                    frames_processed += 1
                    
                    # Process AWS detections
                    for label in results['Labels']:
                        if label.get('Instances'):
                            for instance in label['Instances']:
                                detection = {
                                    "class": label['Name'],
                                    "confidence": label['Confidence'] / 100.0,
                                    "bbox": [
                                        instance['BoundingBox']['Left'],
                                        instance['BoundingBox']['Top'],
                                        instance['BoundingBox']['Left'] + instance['BoundingBox']['Width'],
                                        instance['BoundingBox']['Top'] + instance['BoundingBox']['Height']
                                    ]
                                }
                                if self.is_relevant_detection(detection, "aws"):
                                    frame_detections.append(Detection(
                                        frame_number=frame_idx,
                                        class_name=detection["class"],
                                        confidence=detection["confidence"],
                                        bbox=detection["bbox"],
                                        service=service_name
                                    ))
                    
                elif service_name == "azure":
                    # Azure: Handle rate limiting with proper wait times
                    while not self.can_call_azure():
                        wait_time = self.get_azure_wait_time()
                        print(f"Waiting {wait_time:.1f} seconds for Azure rate limit...")
                        time.sleep(wait_time)
                    
                    results, run_id = self.process_image_azure(frame_path)
                    total_latency += results["latency"]
                    frames_processed += 1
                    
                    # Process Azure detections
                    for obj in results["objects"]:
                        detection = {
                            "class": obj.object_property,
                            "confidence": obj.confidence,
                            "bbox": [
                                obj.rectangle.x / frame.shape[1],
                                obj.rectangle.y / frame.shape[0],
                                (obj.rectangle.x + obj.rectangle.w) / frame.shape[1],
                                (obj.rectangle.y + obj.rectangle.h) / frame.shape[0]
                            ]
                        }
                        if self.is_relevant_detection(detection, "azure"):
                            frame_detections.append(Detection(
                                frame_number=frame_idx,
                                class_name=detection["class"],
                                confidence=detection["confidence"],
                                bbox=detection["bbox"],
                                service=service_name
                            ))
                
                # Update tracker with new detections
                active_tracks = tracker.update(frame_detections)
                
                # Convert tracked objects back to detection format for visualization
                tracked_detections = [
                    {
                        "class": track.class_name,
                        "confidence": track.confidence_history[-1],
                        "bbox": track.bbox_history[-1],
                        "track_id": track.id
                    }
                    for track in active_tracks
                ]
                
                # Only save frame if it contains relevant detections
                if tracked_detections:
                    saved_path = self.draw_detections(frame, tracked_detections, service_name, frame_idx)
                    saved_frame_paths.append(saved_path)
                    total_detections += len(tracked_detections)
                
                total_confidence += sum(d.confidence for d in frame_detections)
                
            except Exception as e:
                print(f"Error processing frame {frame_idx} with {service_name}: {str(e)}")
                continue
        
        # Calculate metrics
        avg_confidence = total_confidence / total_detections if total_detections > 0 else 0
        if frames_processed > 0:
            self.service_latency[service_name] = total_latency / frames_processed
        
        print(f"{service_name} processed {frames_processed} frames with {total_detections} total detections")
        
        return {
            "latency": self.service_latency[service_name],
            "detection_count": total_detections,
            "unique_objects": len(tracker.get_unique_objects()),
            "avg_confidence": avg_confidence,
            "saved_frames": saved_frame_paths,
            "frames_processed": frames_processed
        }

    def compare_services(self, video_path):
        """Compare all services on the same video."""
        # Extract frames from video
        frames, frame_paths, temp_dir, video_info = self.extract_frames(video_path)
        
        try:
            # Process with YOLO
            yolo_results, yolo_run_id = self.process_image_yolo(video_path)
            
            # Process frames with AWS and Azure
            aws_results = self.process_video_frames("aws", frame_paths)
            azure_results = self.process_video_frames("azure", frame_paths)
            
            # Get MLFlow dashboard URL
            dashboard_url = "http://localhost:5000"  # Default MLFlow UI URL
            
            return {
                "video_info": {
                    "total_frames": video_info["total_frames"],
                    "fps": video_info["fps"],
                    "duration_seconds": video_info["duration"],
                    "sampled_frames": video_info["sampled_frames"],
                    "frame_indices": video_info["frame_indices"]
                },
                "processing_time": {
                    "yolo": yolo_results["latency"],
                    "aws": aws_results["latency"],
                    "azure": azure_results["latency"]
                },
                "total_detections": {
                    "yolo": yolo_results["detection_count"],
                    "aws": aws_results["detection_count"],
                    "azure": azure_results["detection_count"]
                },
                "frames": {
                    "total": video_info["total_frames"],
                    "sampled": video_info["sampled_frames"],
                    "processed": {
                        "yolo": yolo_results["frames_processed"],
                        "aws": aws_results["frames_processed"],
                        "azure": azure_results["frames_processed"]
                    }
                },
                "costs": {
                    "yolo": 0.0,  # Local processing
                    "aws": 0.0001 * aws_results["frames_processed"],  # Cost per frame
                    "azure": 0.0001 * azure_results["frames_processed"]  # Cost per frame
                },
                "dashboard_url": dashboard_url,
                "saved_frames": {
                    "yolo": yolo_results["saved_frames"],
                    "aws": aws_results["saved_frames"],
                    "azure": azure_results["saved_frames"]
                },
                "detections": {
                    "yolo": [],  # We'll populate these if needed
                    "aws": [],
                    "azure": []
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
    print("Latency (ms):", results["processing_time"])
    print("Detection Count:", results["total_detections"])
    print("Average Confidence:", results["avg_confidence"])

if __name__ == "__main__":
    main() 