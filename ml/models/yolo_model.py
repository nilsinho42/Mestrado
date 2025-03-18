import torch
import torch.nn as nn
from ultralytics import YOLO
import mlflow
import cv2
import numpy as np
from pathlib import Path
import time
from threading import Thread
from queue import Queue

class YOLODetector:
    def __init__(self, config):
        """Initialize YOLO model with configuration."""
        self.config = config["models"]["yolo"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize YOLO model with pre-trained weights
        self.model = YOLO('yolov8n.pt')  # Using smaller model for better performance
        
        # Define class mappings for different datasets
        self.class_mappings = {
            'bdd100k': {
                'car': 2,
                'bus': 5,
                'truck': 7,
                'motorcycle': 3,
                'bicycle': 1
            },
            'citypersons': {
                'person': 0
            }
        }
        
        # Configure model parameters
        self.model.conf = self.config["confidence_threshold"]
        self.model.iou = self.config["nms_threshold"]
        
        # Initialize frame processing queue
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        
    def process_frame(self, frame):
        """Process a single frame."""
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Perform inference
        results = self.model.predict(
            source=frame,
            conf=self.config["confidence_threshold"],
            iou=self.config["nms_threshold"],
            verbose=False
        )[0]
        
        return results, frame
    
    def process_video_stream(self, source=0, display=True, save_path=None):
        """
        Process video stream for real-time inference.
        
        Args:
            source: Integer for webcam device ID or string for video file path
            display: Whether to display the processed stream
            save_path: Path to save the processed video
        """
        # Open video capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source {source}")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer if save_path is provided
        writer = None
        if save_path:
            writer = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
        
        # Initialize counters and FPS calculator
        frame_count = 0
        total_vehicles = 0
        total_persons = 0
        start_time = time.time()
        
        # Set lower resolution for webcam
        if source == 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames to maintain performance
                if frame_count % 2 != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                results, processed_frame = self.process_frame(frame)
                
                # Process detections
                vehicles = 0
                persons = 0
                
                for box in results.boxes:
                    # Get box coordinates and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    # Count detections
                    if cls in [self.class_mappings['bdd100k'][c] for c in 
                             ['car', 'bus', 'truck', 'motorcycle', 'bicycle']]:
                        vehicles += 1
                        color = (0, 255, 0)  # Green for vehicles
                    elif cls == self.class_mappings['citypersons']['person']:
                        persons += 1
                        color = (0, 0, 255)  # Red for persons
                    
                    # Draw bounding box
                    cv2.rectangle(processed_frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                color, 2)
                    
                    # Add label
                    label = f"{results.names[cls]} {conf:.2f}"
                    cv2.putText(processed_frame, label, 
                              (int(x1), int(y1-10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, color, 2)
                
                # Update totals
                total_vehicles += vehicles
                total_persons += persons
                frame_count += 1
                
                # Calculate FPS
                fps = frame_count / (time.time() - start_time)
                
                # Add statistics overlay
                stats = f"FPS: {fps:.1f} | Vehicles: {vehicles} | Persons: {persons}"
                cv2.putText(processed_frame, stats, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Display the frame
                if display:
                    cv2.imshow('Object Detection', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Save the frame if writer is initialized
                if writer:
                    writer.write(processed_frame)
                
        finally:
            # Clean up
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Return statistics
        return {
            'total_frames': frame_count,
            'average_fps': frame_count / (time.time() - start_time),
            'total_vehicles': total_vehicles,
            'total_persons': total_persons,
            'avg_vehicles_per_frame': total_vehicles / frame_count,
            'avg_persons_per_frame': total_persons / frame_count
        }
    
    def train_on_bdd100k(self, data_yaml_path, epochs):
        """Fine-tune on BDD100K for vehicle detection."""
        vehicle_classes = [self.class_mappings['bdd100k'][c] for c in 
                         ['car', 'bus', 'truck', 'motorcycle', 'bicycle']]
        
        return self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=self.config["input_size"][0],
            batch=self.config["batch_size"],
            device=self.device,
            classes=vehicle_classes,
            val=True,
            plots=True
        )
    
    def train_on_citypersons(self, data_yaml_path, epochs):
        """Fine-tune on CityPersons for pedestrian detection."""
        person_class = [self.class_mappings['citypersons']['person']]
        
        return self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=self.config["input_size"][0],
            batch=self.config["batch_size"],
            device=self.device,
            classes=person_class,
            val=True,
            plots=True
        )
    
    def evaluate(self, data_yaml_path, dataset_type):
        """Evaluate the model on specified dataset."""
        classes = None
        if dataset_type == 'bdd100k':
            classes = [self.class_mappings['bdd100k'][c] for c in 
                      ['car', 'bus', 'truck', 'motorcycle', 'bicycle']]
        elif dataset_type == 'citypersons':
            classes = [self.class_mappings['citypersons']['person']]
            
        results = self.model.val(
            data=data_yaml_path,
            classes=classes
        )
        
        metrics = {
            "precision": results.results_dict["metrics/precision(B)"],
            "recall": results.results_dict["metrics/recall(B)"],
            "mAP50": results.results_dict["metrics/mAP50(B)"],
            "mAP50-95": results.results_dict["metrics/mAP50-95(B)"]
        }
        return metrics
    
    def predict(self, image):
        """Run inference on a single image."""
        results = self.model.predict(
            source=image,
            conf=self.config["confidence_threshold"],
            iou=self.config["nms_threshold"]
        )
        return results[0]
    
    def save_model(self, path):
        """Save the model to disk."""
        self.model.save(path)
        
    def load_model(self, path):
        """Load a saved model."""
        self.model = YOLO(path)
        self.model.conf = self.config["confidence_threshold"]
        self.model.iou = self.config["nms_threshold"] 