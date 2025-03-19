import cv2
import numpy as np
import torch
import time
import psutil
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path

class VideoProcessor:
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.to(device)
        self.model.eval()

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the YOLO model from the specified path."""
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def process_video(self, video_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Process a video file and return detection results, comparison metrics, and performance metrics."""
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Initialize metrics
            start_time = time.time()
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Process frames
            detection_results = []
            frame_number = 0
            total_detections = 0
            total_confidence = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                results = self.model(frame)
                
                # Extract detections
                for det in results.xyxy[0]:  # xyxy format
                    x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                    detection_type = results.names[int(cls)]
                    
                    detection_results.append({
                        "frame_number": frame_number,
                        "detection_type": detection_type,
                        "confidence": float(conf),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "metadata": {
                            "class_id": int(cls),
                            "frame_size": [width, height]
                        }
                    })
                    
                    total_detections += 1
                    total_confidence += float(conf)

                frame_number += 1

            # Calculate metrics
            processing_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory
            
            # Calculate accuracy (using confidence as a proxy)
            accuracy = total_confidence / total_detections if total_detections > 0 else 0

            # Create comparison metrics
            comparison = {
                "processing_time": processing_time,
                "memory_usage": memory_usage,
                "accuracy": accuracy,
                "metadata": {
                    "fps": fps,
                    "frame_count": frame_count,
                    "resolution": [width, height],
                    "total_detections": total_detections
                }
            }

            # Create performance metrics
            metric = {
                "metric_type": "processing_time",
                "value": processing_time,
                "metadata": {
                    "memory_usage": memory_usage,
                    "frames_processed": frame_number,
                    "detections_per_frame": total_detections / frame_number if frame_number > 0 else 0
                }
            }

            cap.release()
            return detection_results, comparison, metric

        except Exception as e:
            raise RuntimeError(f"Error processing video: {str(e)}")

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache() 