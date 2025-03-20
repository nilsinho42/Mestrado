import mlflow
import time
from datetime import datetime
import json
import os
from typing import Dict, List, Any
import numpy as np

class DetectionTracker:
    def __init__(self, experiment_name: str = "object-detection-comparison"):
        """Initialize MLFlow tracking for object detection comparison."""
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(experiment_name)
        
    def start_detection_run(self, 
                          service_type: str,  # "yolo", "aws", or "azure"
                          model_name: str,
                          input_type: str,  # "image", "video", or "webcam"
                          batch_size: int = 1):
        """Start a new MLFlow run for detection."""
        run_name = f"{service_type}-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            # Log basic parameters
            mlflow.log_params({
                "service_type": service_type,
                "model_name": model_name,
                "input_type": input_type,
                "batch_size": batch_size,
                "timestamp": datetime.now().isoformat()
            })
            
            return run.info.run_id
    
    def log_detection_metrics(self, 
                            run_id: str,
                            metrics: Dict[str, float],
                            artifacts: Dict[str, Any] = None):
        """Log detection metrics and artifacts."""
        with mlflow.start_run(run_id=run_id, nested=True):
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log artifacts if provided
            if artifacts:
                for name, artifact in artifacts.items():
                    if isinstance(artifact, (np.ndarray, list)):
                        mlflow.log_artifact(f"{name}.npy")
                    elif isinstance(artifact, str):
                        mlflow.log_artifact(artifact)
    
    def log_cloud_cost(self, 
                      run_id: str,
                      service_type: str,
                      cost_details: Dict[str, float]):
        """Log cloud service costs."""
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_params({
                "service_type": service_type,
                "cost_timestamp": datetime.now().isoformat()
            })
            mlflow.log_metrics(cost_details)
    
    def log_detection_results(self,
                            run_id: str,
                            results: List[Dict[str, Any]],
                            save_path: str = None):
        """Log detection results with optional saving."""
        with mlflow.start_run(run_id=run_id, nested=True):
            # Convert results to JSON for logging
            results_json = json.dumps(results, indent=2)
            
            # Save results if path provided
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(results_json)
                mlflow.log_artifact(save_path)
            else:
                mlflow.log_text(results_json, "detection_results.json")
    
    def log_performance_metrics(self,
                              run_id: str,
                              metrics: Dict[str, float]):
        """Log performance metrics like FPS, latency, etc."""
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_metrics(metrics)
    
    def compare_services(self,
                        run_ids: List[str],
                        metric_name: str):
        """Compare metrics across different service runs."""
        results = {}
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            if metric_name in run.data.metrics:
                results[run_id] = run.data.metrics[metric_name]
        return results

# Example usage:
"""
tracker = DetectionTracker()

# Start a YOLO detection run
yolo_run_id = tracker.start_detection_run(
    service_type="yolo",
    model_name="yolov8n",
    input_type="webcam"
)

# Log YOLO metrics
tracker.log_detection_metrics(
    yolo_run_id,
    metrics={
        "fps": 30.0,
        "latency_ms": 33.3,
        "detection_confidence": 0.85,
        "objects_detected": 5
    }
)

# Start an AWS detection run
aws_run_id = tracker.start_detection_run(
    service_type="aws",
    model_name="rekognition",
    input_type="image"
)

# Log AWS metrics and costs
tracker.log_detection_metrics(
    aws_run_id,
    metrics={
        "latency_ms": 150.0,
        "detection_confidence": 0.92,
        "objects_detected": 6
    }
)

tracker.log_cloud_cost(
    aws_run_id,
    service_type="aws",
    cost_details={
        "api_call_cost": 0.0001,
        "total_cost": 0.0001
    }
)

# Compare services
comparison = tracker.compare_services(
    [yolo_run_id, aws_run_id],
    "latency_ms"
)
""" 