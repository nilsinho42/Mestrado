import os
import sys
import time
import argparse
import logging
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import uuid
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import custom modules
from video_processor import VideoProcessor
from detectors import YOLODetector, AWSDetector, AzureDetector
from trackers import IoUTracker, DeepSORTTracker, AzureContainerAppTracker
from cloud_storage import CloudStorage
from cost_utils import CostCalculator
from db_utils import Database
from mlflow_tracking import DetectionTracker

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoPipeline:
    def __init__(self, output_dir: str = "./data/object_detection/images"):
        """Initialize video processing pipeline.
        
        Args:
            output_dir: Directory to save extracted images
        """
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.video_processor = VideoProcessor(output_dir=output_dir)
        self.cloud_storage = CloudStorage()
        self.cost_calculator = CostCalculator()
        self.db = Database()
        self.mlflow_tracker = DetectionTracker()
        
        # Initialize detectors
        self.detectors = {
            'yolo': YOLODetector(),
            'aws': AWSDetector(),
            'azure': AzureDetector()
        }
        
        # Initialize trackers
        self.trackers = {
            'yolo': DeepSORTTracker(name="yolo_deepsort"),
            'aws': DeepSORTTracker(name="aws_deepsort"),
            'azure': AzureContainerAppTracker(name="azure_container_app")
        }
        
        # Create database tables if they don't exist
        self.db.create_tables()
        
        logger.info("Initialized video processing pipeline")
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process video through the pipeline.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing video: {video_path}")
        video_path = Path(video_path)
        
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return {'error': 'Video file not found'}
        
        # Generate a unique ID for this processing job
        job_id = str(uuid.uuid4())
        
        # Start MLFlow tracking
        mlflow_run_id = self.mlflow_tracker.start_detection_run(
            service_type="comparison",
            model_name="multiple",
            input_type="video"
        )
        
        try:
            # Process Task A: Image Analysis with Object Detection
            task_a_results = self._process_task_a(video_path)
            
            # Process Task B: Video Processing with Object Tracking
            task_b_results = self._process_task_b(video_path)
            
            # Combine results
            results = {
                'job_id': job_id,
                'mlflow_run_id': mlflow_run_id,
                'video_path': str(video_path),
                'task_a': task_a_results,
                'task_b': task_b_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log final results to MLFlow
            self.mlflow_tracker.log_detection_results(
                run_id=mlflow_run_id,
                results=results,
                save_path=f"results_{job_id}.json"
            )
            
            logger.info(f"Video processing completed successfully for {video_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            return {'error': str(e), 'job_id': job_id}
    
    def _process_task_a(self, video_path: Path) -> Dict[str, Any]:
        """Process Task A: Image Analysis with Object Detection.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with Task A results
        """
        logger.info("Starting Task A: Image Analysis with Object Detection")
        
        # Extract frames from video (1 out of every 5 frames)
        frames, frame_paths, video_info = self.video_processor.extract_frames(
            video_path=str(video_path),
            save_frames=True
        )
        
        logger.info(f"Extracted {len(frames)} frames from video")
        
        # Results dictionary
        results = {
            'video_info': video_info,
            'detectors': {}
        }
        
        # Process each detector
        for detector_name, detector in self.detectors.items():
            logger.info(f"Running detector: {detector_name}")
            
            detector_results = {
                'frames': [],
                'metrics': {
                    'total_latency': 0,
                    'avg_latency': 0,
                    'total_people': 0,
                    'total_vehicles': 0
                }
            }
            
            # Process each frame
            for i, (frame, frame_path) in enumerate(zip(frames, frame_paths)):
                logger.info(f"Processing frame {i+1}/{len(frames)} with {detector_name}")
                
                # Extract image_id from path
                image_id = Path(frame_path).stem
                
                # Run object detection
                try:
                    detections, latency = detector.detect(frame)
                    
                    # Count objects
                    counts = detector.count_objects(detections)
                    
                    # Store results for this image
                    frame_result = {
                        'image_id': image_id,
                        'latency': latency,
                        'people_count': counts['people_count'],
                        'vehicles_count': counts['vehicles_count'],
                        'detections': [d.to_dict() for d in detections]
                    }
                    
                    # Add to detector results
                    detector_results['frames'].append(frame_result)
                    detector_results['metrics']['total_latency'] += latency
                    detector_results['metrics']['total_people'] += counts['people_count']
                    detector_results['metrics']['total_vehicles'] += counts['vehicles_count']
                    
                    # Save detection results to database
                    self.db.save_detection_results({
                        'image_id': image_id,
                        'source': detector_name,
                        'latency': latency,
                        'people_count': counts['people_count'],
                        'vehicles_count': counts['vehicles_count'],
                        'detection_data': frame_result
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing frame with {detector_name}: {e}")
                    frame_result = {
                        'image_id': image_id,
                        'error': str(e)
                    }
                    detector_results['frames'].append(frame_result)
            
            # Calculate metrics
            if len(frames) > 0:
                detector_results['metrics']['avg_latency'] = detector_results['metrics']['total_latency'] / len(frames)
                detector_results['metrics']['avg_people_per_frame'] = detector_results['metrics']['total_people'] / len(frames)
                detector_results['metrics']['avg_vehicles_per_frame'] = detector_results['metrics']['total_vehicles'] / len(frames)
            
            # Calculate cost
            if detector_name == 'aws':
                cost = self.cost_calculator.get_aws_cost_for_task_a(len(frames))
            elif detector_name == 'azure':
                cost = self.cost_calculator.get_azure_cost_for_task_a(len(frames))
            else:
                cost = self.cost_calculator.get_local_cost_for_task_a(len(frames))
            
            detector_results['cost'] = cost
            
            # Save metrics to database
            self.db.save_metrics({
                'image_id': video_info['video_id'],
                'source': detector_name,
                'latency': detector_results['metrics']['avg_latency'],
                'cost_image_processing': cost
            })
            
            # Add to results
            results['detectors'][detector_name] = detector_results
        
        logger.info("Task A processing completed")
        return results
    
    def _process_task_b(self, video_path: Path) -> Dict[str, Any]:
        """Process Task B: Video Processing with Object Tracking.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with Task B results
        """
        logger.info("Starting Task B: Video Processing with Object Tracking")
        
        # Get video file size
        video_size_bytes = video_path.stat().st_size
        video_size_gb = video_size_bytes / (1024 * 1024 * 1024)
        
        # Upload video to cloud storage
        storage_results = self.cloud_storage.upload_to_all(video_path)
        
        # Results dictionary
        results = {
            'video_path': str(video_path),
            'video_size_bytes': video_size_bytes,
            'video_size_gb': video_size_gb,
            'storage_results': storage_results,
            'trackers': {}
        }
        
        # Process each detector/tracker
        for service_name, tracker in self.trackers.items():
            logger.info(f"Running tracking with {service_name}")
            
            # Start timing
            tracking_start_time = time.time()
            
            try:
                # Process video tracking
                if service_name == 'aws':
                    tracking_results = self._run_aws_tracking(video_path, tracker)
                elif service_name == 'azure':
                    tracking_results = self._run_azure_tracking(video_path, tracker)
                else:  # yolo
                    tracking_results = self._run_local_tracking(video_path, tracker)
                
                # Calculate processing time
                processing_time = time.time() - tracking_start_time
                tracking_results['processing_time'] = processing_time
                
                # Calculate cost
                if service_name == 'aws':
                    # Get frame count from tracking results
                    frame_count = tracking_results.get('processed_frames', 0)
                    cost = self.cost_calculator.get_aws_cost_for_task_b(
                        video_size_gb=video_size_gb,
                        frame_count=frame_count,
                        lambda_execution_time_seconds=processing_time
                    )
                elif service_name == 'azure':
                    frame_count = tracking_results.get('processed_frames', 0)
                    cost = self.cost_calculator.get_azure_cost_for_task_b(
                        video_size_gb=video_size_gb,
                        frame_count=frame_count,
                        container_execution_time_seconds=processing_time
                    )
                else:
                    cost = self.cost_calculator.get_local_cost_for_task_b(
                        execution_time_seconds=processing_time
                    )
                
                tracking_results['cost'] = cost
                
                # Save tracking results to database
                self.db.save_tracking_results({
                    'video_id': tracking_results['video_id'],
                    'source': service_name,
                    'processing_time': processing_time,
                    'people_tracked': tracking_results['counts']['people_tracked'],
                    'vehicles_tracked': tracking_results['counts']['vehicles_tracked'],
                    'tracking_data': tracking_results
                })
                
                # Save metrics to database
                self.db.save_metrics({
                    'image_id': tracking_results['video_id'],
                    'source': service_name,
                    'total_processing_time': processing_time,
                    'cost_video_processing': cost['total_cost']
                })
                
                # Add to results
                results['trackers'][service_name] = tracking_results
                
            except Exception as e:
                logger.error(f"Error processing tracking with {service_name}: {e}", exc_info=True)
                results['trackers'][service_name] = {
                    'error': str(e)
                }
        
        logger.info("Task B processing completed")
        return results
    
    def _run_local_tracking(self, video_path: Path, tracker) -> Dict[str, Any]:
        """Run local tracking using YOLO + DeepSORT.
        
        Args:
            video_path: Path to video file
            tracker: Tracker object
            
        Returns:
            Dictionary with tracking results
        """
        # Reset tracker
        tracker.reset()
        
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video ID
        video_id = f"{Path(video_path).stem}_{uuid.uuid4().hex[:8]}"
        
        # Run YOLO detector
        detector = self.detectors['yolo']
        
        # Process each frame
        frame_idx = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections, _ = detector.detect(frame)
            
            # Update tracker
            tracker.update(detections, frame_idx)
            
            frame_idx += 1
            processed_frames += 1
        
        # Release video capture
        cap.release()
        
        # Get tracking results
        tracking_results = tracker.get_results()
        tracking_results['video_id'] = video_id
        tracking_results['video_path'] = str(video_path)
        tracking_results['processed_frames'] = processed_frames
        
        return tracking_results
    
    def _run_aws_tracking(self, video_path: Path, tracker) -> Dict[str, Any]:
        """Run AWS tracking using Rekognition + DeepSORT.
        
        In a real implementation, this would use AWS Rekognition for detection
        and AWS Lambda with DeepSORT for tracking.
        
        Args:
            video_path: Path to video file
            tracker: Tracker object
            
        Returns:
            Dictionary with tracking results
        """
        # For this implementation, we'll simulate AWS tracking
        # using our local detector but with AWS processing time simulation
        
        # Reset tracker
        tracker.reset()
        
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video ID
        video_id = f"{Path(video_path).stem}_{uuid.uuid4().hex[:8]}"
        
        # Run AWS detector
        detector = self.detectors['aws']
        
        # Process each frame
        frame_idx = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections, _ = detector.detect(frame)
            
            # Update tracker
            tracker.update(detections, frame_idx)
            
            # Simulate AWS processing delay
            time.sleep(0.1)  # 100ms simulated delay
            
            frame_idx += 1
            processed_frames += 1
        
        # Release video capture
        cap.release()
        
        # Get tracking results
        tracking_results = tracker.get_results()
        tracking_results['video_id'] = video_id
        tracking_results['video_path'] = str(video_path)
        tracking_results['processed_frames'] = processed_frames
        
        return tracking_results
    
    def _run_azure_tracking(self, video_path: Path, tracker) -> Dict[str, Any]:
        """Run Azure tracking using AI Vision + DeepSORT.
        
        In a real implementation, this would use Azure AI Vision for detection
        and Azure Container App with DeepSORT for tracking.
        
        Args:
            video_path: Path to video file
            tracker: Tracker object
            
        Returns:
            Dictionary with tracking results
        """
        # Check if we're using the Azure Container App tracker
        is_container_app = isinstance(tracker, AzureContainerAppTracker)
        
        # Reset tracker
        tracker.reset()
        
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video ID
        video_id = f"{Path(video_path).stem}_{uuid.uuid4().hex[:8]}"
        
        # Run Azure detector
        detector = self.detectors['azure']
        
        # Process each frame
        frame_idx = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections, _ = detector.detect(frame)
            
            # For Container App tracker, we need to attach the frame to detections
            if is_container_app:
                for detection in detections:
                    # Store frame reference in detection objects for the Container App
                    detection._frame = frame
            
            # Update tracker
            tracker.update(detections, frame_idx)
            
            # If using Container App, the request already includes some delay
            # If using local simulation, add a simulated delay
            if not is_container_app:
                # Simulate Azure processing delay
                time.sleep(0.12)  # 120ms simulated delay
            
            frame_idx += 1
            processed_frames += 1
        
        # Release video capture
        cap.release()
        
        # Get tracking results
        tracking_results = tracker.get_results()
        tracking_results['video_id'] = video_id
        tracking_results['video_path'] = str(video_path)
        tracking_results['processed_frames'] = processed_frames
        
        return tracking_results

def main():
    """Main entry point for video processing pipeline."""
    parser = argparse.ArgumentParser(description='Video Processing Pipeline')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='./data/object_detection/images', help='Output directory for frames')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VideoPipeline(output_dir=args.output)
    
    # Process video
    results = pipeline.process_video(args.video)
    
    # Save results to file
    output_file = Path(args.output) / f"results_{Path(args.video).stem}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")

if __name__ == '__main__':
    main() 