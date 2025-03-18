import mlflow
import yaml
import torch
import time
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime

from yolo_model import YOLOModel
from faster_rcnn_model import FasterRCNNModel
from metrics import calculate_metrics
from dataset import get_test_loader

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_model(model, test_loader, device: str) -> Dict:
    """Test a single model and return metrics."""
    model.eval()
    all_predictions = []
    all_ground_truth = []
    total_time = 0
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = [image.to(device) for image in images]
            
            # Measure inference time
            start_time = time.time()
            predictions = model(images)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            all_predictions.extend(predictions)
            all_ground_truth.extend(targets)
    
    metrics = calculate_metrics(all_predictions, all_ground_truth)
    metrics['inference_time'] = total_time / len(test_loader)
    return metrics

def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Set up MLflow
    mlflow.set_experiment("model_comparison")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    yolo_model = YOLOModel(config['models']['yolo'])
    faster_rcnn_model = FasterRCNNModel(config['models']['faster_rcnn'])
    
    # Load test data
    test_loader = get_test_loader(
        dataset_path=config['data']['dataset_path'],
        batch_size=config['training']['batch_size'],
        num_workers=4
    )
    
    # Test models
    with mlflow.start_run(run_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Test YOLO
        yolo_metrics = test_model(yolo_model, test_loader, device)
        for metric_name, value in yolo_metrics.items():
            mlflow.log_metric(f"yolo_{metric_name}", value)
        
        # Test Faster R-CNN
        faster_rcnn_metrics = test_model(faster_rcnn_model, test_loader, device)
        for metric_name, value in faster_rcnn_metrics.items():
            mlflow.log_metric(f"faster_rcnn_{metric_name}", value)
        
        # Log comparison results
        comparison_results = {
            'yolo': yolo_metrics,
            'faster_rcnn': faster_rcnn_metrics
        }
        mlflow.log_dict(comparison_results, "comparison_results.json")
        
        # Print results
        print("\nModel Comparison Results:")
        print("=" * 50)
        print("\nYOLO Metrics:")
        for metric, value in yolo_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nFaster R-CNN Metrics:")
        for metric, value in faster_rcnn_metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 