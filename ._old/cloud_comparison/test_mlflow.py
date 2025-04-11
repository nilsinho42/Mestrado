import os
from mlflow_tracking import DetectionTracker
import time
import numpy as np

def test_mlflow_tracking():
    """Test MLFlow tracking functionality."""
    # Initialize tracker
    tracker = DetectionTracker()
    
    # Test 1: Basic run creation and metric logging
    print("\nTest 1: Basic run creation and metric logging")
    run_id = tracker.start_detection_run(
        service_type="test",
        model_name="test-model",
        input_type="image"
    )
    
    # Log some test metrics
    tracker.log_detection_metrics(
        run_id,
        metrics={
            "test_metric": 0.95,
            "latency_ms": 100.0
        }
    )
    
    # Test 2: Cloud cost logging
    print("\nTest 2: Cloud cost logging")
    tracker.log_cloud_cost(
        run_id,
        service_type="test",
        cost_details={
            "api_call_cost": 0.0001,
            "total_cost": 0.0001
        }
    )
    
    # Test 3: Detection results logging
    print("\nTest 3: Detection results logging")
    test_results = [
        {
            "class": "person",
            "confidence": 0.95,
            "bbox": [0, 0, 100, 100]
        }
    ]
    tracker.log_detection_results(run_id, test_results)
    
    # Test 4: Performance metrics logging
    print("\nTest 4: Performance metrics logging")
    tracker.log_performance_metrics(
        run_id,
        metrics={
            "fps": 30.0,
            "memory_usage_mb": 500.0
        }
    )
    
    # Test 5: Service comparison
    print("\nTest 5: Service comparison")
    # Create a second run for comparison
    run_id2 = tracker.start_detection_run(
        service_type="test2",
        model_name="test-model2",
        input_type="image"
    )
    
    tracker.log_detection_metrics(
        run_id2,
        metrics={
            "latency_ms": 150.0
        }
    )
    
    # Compare latency between runs
    comparison = tracker.compare_services(
        [run_id, run_id2],
        "latency_ms"
    )
    print("Latency comparison:", comparison)
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_mlflow_tracking() 