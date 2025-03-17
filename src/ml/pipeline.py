import cv2
import numpy as np
import torch
from typing import List, Dict, Any
from datetime import datetime
from .models import YOLODetector, FasterRCNNDetector, RCNNDetector
from .preprocessing import preprocess_image
from .utils import calculate_flow, bbox_to_dict

class DetectionPipeline:
    def __init__(self, model_type: str = 'yolo', device: str = 'cuda'):
        self.model_type = model_type
        self.device = device
        self.model = self._initialize_model()
        self.previous_detections = None
        self.previous_timestamp = None

    def _initialize_model(self):
        if self.model_type == 'yolo':
            return YOLODetector(device=self.device)
        elif self.model_type == 'faster-rcnn':
            return FasterRCNNDetector(device=self.device)
        elif self.model_type == 'rcnn':
            return RCNNDetector(device=self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame and return detections with flow metrics"""
        # Preprocess image
        processed_frame = preprocess_image(frame)
        
        # Get current timestamp
        current_timestamp = datetime.now()

        # Perform detection
        detections = self.model.detect(processed_frame)
        
        # Calculate flow metrics if we have previous detections
        flow_metrics = {}
        if self.previous_detections is not None and self.previous_timestamp is not None:
            time_diff = (current_timestamp - self.previous_timestamp).total_seconds()
            flow_metrics = calculate_flow(
                self.previous_detections,
                detections,
                time_diff
            )

        # Prepare results
        results = {
            'timestamp': current_timestamp.isoformat(),
            'detections': [bbox_to_dict(det) for det in detections],
            'counts': {
                'person': len([d for d in detections if d['class'] == 'person']),
                'vehicle': len([d for d in detections if d['class'] in ['car', 'truck', 'bus']])
            },
            'flow_metrics': flow_metrics
        }

        # Update previous state
        self.previous_detections = detections
        self.previous_timestamp = current_timestamp

        return results

    def process_video_stream(self, stream_url: str, callback=None):
        """Process video stream and call callback function with results"""
        cap = cv2.VideoCapture(stream_url)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.process_frame(frame)
                
                if callback:
                    callback(results)

        finally:
            cap.release()

    def process_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process a batch of images"""
        return [self.process_frame(img) for img in images]

    def update_model(self, new_weights_path: str):
        """Update model weights"""
        self.model.load_weights(new_weights_path)

    def get_model_metrics(self) -> Dict[str, float]:
        """Get current model performance metrics"""
        return {
            'inference_time': self.model.average_inference_time,
            'confidence_threshold': self.model.confidence_threshold,
            'device': self.device,
            'model_type': self.model_type
        } 