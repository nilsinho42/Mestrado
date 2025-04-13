"""
Video processing functionality.
Provides utilities for video frame extraction, processing, and analysis.
"""

import cv2
import numpy as np
import os
import tempfile
import time
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Base video processor for handling frame extraction and processing."""
    
    def __init__(self, output_dir: str = "./data/object_detection/images"):
        """
        Initialize video processor.
        
        Args:
            output_dir: Directory to save extracted frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Video processor initialized with output directory: {self.output_dir}")
    
    def extract_frames(self, video_path: str, fps_reduction_factor: int = 5, 
                       save_frames: bool = True) -> Tuple[List[np.ndarray], List[str], Dict[str, Any]]:
        """
        Extract frames from video at reduced FPS.
        
        Args:
            video_path: Path to the video file
            fps_reduction_factor: Factor by which to reduce FPS (e.g., 5 means 30fps -> 6fps)
            save_frames: Whether to save frames to disk
            
        Returns:
            Tuple containing:
            - List of extracted frames as numpy arrays
            - List of paths to saved frames (if save_frames=True)
            - Dictionary with video info (fps, frame count, etc.)
        """
        frames = []
        frame_paths = []
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        if total_frames == 0:
            raise ValueError(f"No frames found in video: {video_path}")
        
        # Calculate target FPS and frame interval
        target_fps = original_fps / fps_reduction_factor
        frame_interval = fps_reduction_factor  # Take 1 frame every N frames
        
        # Create temporary directory for frames if needed
        if save_frames:
            temp_dir = tempfile.mkdtemp()
            # Create a subfolder based on video filename
            video_name = Path(video_path).stem
            frames_dir = Path(self.output_dir) / video_name
            frames_dir.mkdir(exist_ok=True)
        
        try:
            frame_count = 0
            frames_sampled = 0
            frame_indices = []
            
            while frame_count < total_frames:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                
                if ret:
                    if save_frames:
                        # Save frame to the output directory with a timestamp
                        frame_filename = f"{video_name}_frame_{frames_sampled:04d}.jpg"
                        frame_path = str(frames_dir / frame_filename)
                        cv2.imwrite(frame_path, frame)
                        frame_paths.append(frame_path)
                    
                    frames.append(frame)
                    frame_indices.append(frame_count)
                    frames_sampled += 1
                
                # Move to next frame based on interval
                frame_count += frame_interval
            
            video_info = {
                "total_frames": total_frames,
                "fps": original_fps,
                "target_fps": target_fps,
                "duration": duration,
                "width": width,
                "height": height,
                "sampled_frames": frames_sampled,
                "frame_indices": frame_indices,
                "video_path": video_path,
                "video_name": Path(video_path).stem
            }
            
            logger.info(f"Extracted {frames_sampled} frames from video: {video_path}")
            logger.info(f"Original: {total_frames} frames at {original_fps:.1f} FPS, "
                        f"sampled at 1/{fps_reduction_factor} ({target_fps:.1f} FPS)")
            
            return frames, frame_paths, video_info
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
        finally:
            cap.release()
    
    def process_video(self, video_path: str, fps_reduction_factor: int = 5, 
                      save_frames: bool = True) -> Dict[str, Any]:
        """
        Process video and extract frames for analysis.
        Base implementation that should be extended by specific processors.
        
        Args:
            video_path: Path to the video file
            fps_reduction_factor: Factor by which to reduce FPS
            save_frames: Whether to save frames to disk
            
        Returns:
            Dictionary with processing results and video info
        """
        start_time = time.time()
        
        # Extract frames from video
        frames, frame_paths, video_info = self.extract_frames(
            video_path, 
            fps_reduction_factor=fps_reduction_factor,
            save_frames=save_frames
        )
        
        processing_time = time.time() - start_time
        
        # Create basic results structure
        results = {
            "video_info": video_info,
            "processing_time": processing_time,
            "frame_count": len(frames),
            "frame_paths": frame_paths,
            "status": "completed"
        }
        
        return results
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]], 
                       confidence_threshold: float = 0.3) -> np.ndarray:
        """
        Draw detection bounding boxes on an image.
        
        Args:
            image: Image as numpy array
            detections: List of detection dictionaries
            confidence_threshold: Minimum confidence to draw
            
        Returns:
            Image with drawn detections
        """
        image_height, image_width = image.shape[:2]
        result_image = image.copy()
        
        # Color map for different classes
        color_map = {
            "person": (0, 255, 0),    # Green for people
            "car": (255, 0, 0),       # Blue for cars
            "truck": (255, 0, 255),   # Magenta for trucks
            "bus": (255, 165, 0),     # Orange for buses
            "motorcycle": (0, 255, 255)  # Yellow for motorcycles
        }
        
        for detection in detections:
            confidence = detection.get("confidence", 0)
            if confidence < confidence_threshold:
                continue
            
            # Get bounding box
            bbox = detection.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            
            # Convert from normalized coordinates if needed
            if max(bbox) <= 1.0:
                x1 = int(x1 * image_width)
                y1 = int(y1 * image_height)
                x2 = int(x2 * image_width)
                y2 = int(y2 * image_height)
            
            # Get class and determine color
            class_name = detection.get("detection_type", "unknown")
            color = color_map.get(class_name.lower(), (0, 0, 255))  # Default to red
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1 = max(y1, label_size[1])
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - baseline), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - baseline), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image


class ImageAnalysisProcessor(VideoProcessor):
    """
    Processor for Task A: Image Analysis with Object Detection.
    Samples frames from video and processes them with different providers.
    """
    
    def __init__(self, output_dir: str = "./data/object_detection/images"):
        super().__init__(output_dir=output_dir)
        # Will be initialized with provider-specific detectors
        self.detectors = {}
    
    def register_detector(self, provider: str, detector: Any) -> None:
        """Register a detector for a specific provider."""
        self.detectors[provider] = detector
        logger.info(f"Registered detector for provider: {provider}")
    
    def process_video(self, video_path: str, fps_reduction_factor: int = 5, 
                     providers: List[str] = None) -> Dict[str, Any]:
        """
        Process video for Task A (Image Analysis).
        
        Args:
            video_path: Path to the video file
            fps_reduction_factor: Factor by which to reduce FPS
            providers: List of providers to use for detection
            
        Returns:
            Dictionary with detection results for each provider
        """
        if providers is None:
            providers = list(self.detectors.keys())
        
        # Start timing
        task_start_time = time.time()
        
        # Extract frames
        frames, frame_paths, video_info = self.extract_frames(
            video_path, 
            fps_reduction_factor=fps_reduction_factor,
            save_frames=True
        )
        
        # Initialize results for each provider
        results = {
            "video_info": video_info,
            "providers": {},
            "summary": {
                "people_count": {},
                "vehicle_count": {},
                "avg_latency": {}
            }
        }
        
        # Process frames with each provider
        for provider in providers:
            if provider not in self.detectors:
                logger.warning(f"No detector registered for provider: {provider}")
                continue
            
            logger.info(f"Processing frames with provider: {provider}")
            provider_start_time = time.time()
            provider_results = []
            
            # Track metrics
            total_latency = 0
            people_count = 0
            vehicle_count = 0
            
            # Process each frame
            for i, (frame, frame_path) in enumerate(zip(frames, frame_paths)):
                # Process frame with provider
                frame_start_time = time.time()
                detections = self.detectors[provider].process_image(frame, frame_path)
                frame_latency = time.time() - frame_start_time
                total_latency += frame_latency
                
                # Count people and vehicles
                frame_people = sum(1 for d in detections if 
                                 d.get("detection_type", "").lower() in ["person", "people"])
                frame_vehicles = sum(1 for d in detections if 
                                   d.get("detection_type", "").lower() in 
                                   ["car", "truck", "bus", "motorcycle", "vehicle"])
                
                people_count += frame_people
                vehicle_count += frame_vehicles
                
                # Store results for this frame
                frame_result = {
                    "frame_number": i,
                    "frame_path": frame_path,
                    "latency": frame_latency,
                    "detections": detections,
                    "people_count": frame_people,
                    "vehicle_count": frame_vehicles,
                    "image_id": Path(frame_path).stem
                }
                provider_results.append(frame_result)
            
            # Calculate provider-level metrics
            provider_time = time.time() - provider_start_time
            avg_latency = total_latency / len(frames) if frames else 0
            
            # Store provider results
            results["providers"][provider] = {
                "results": provider_results,
                "total_time": provider_time,
                "avg_latency": avg_latency,
                "frame_count": len(frames),
                "people_count": people_count,
                "vehicle_count": vehicle_count
            }
            
            # Add to summary
            results["summary"]["people_count"][provider] = people_count
            results["summary"]["vehicle_count"][provider] = vehicle_count
            results["summary"]["avg_latency"][provider] = avg_latency
        
        # Calculate overall processing time
        results["total_processing_time"] = time.time() - task_start_time
        
        return results


class VideoTrackingProcessor(VideoProcessor):
    """
    Processor for Task B: Video Processing with Object Tracking.
    Uploads video to cloud providers and tracks objects using various services.
    """
    
    def __init__(self, output_dir: str = "./data/object_detection/tracks"):
        super().__init__(output_dir=output_dir)
        # Will be initialized with provider-specific trackers
        self.trackers = {}
        # Storage providers for uploading/downloading videos
        self.storage_providers = {}
    
    def register_tracker(self, provider: str, tracker: Any) -> None:
        """Register a tracker for a specific provider."""
        self.trackers[provider] = tracker
        logger.info(f"Registered tracker for provider: {provider}")
    
    def register_storage_provider(self, provider: str, storage_provider: Any) -> None:
        """Register a storage provider for a specific provider."""
        self.storage_providers[provider] = storage_provider
        logger.info(f"Registered storage provider for provider: {provider}")
    
    def process_video(self, video_path: str, 
                     providers: List[str] = None) -> Dict[str, Any]:
        """
        Process video for Task B (Video Tracking).
        
        Args:
            video_path: Path to the video file
            providers: List of providers to use for tracking
            
        Returns:
            Dictionary with tracking results for each provider
        """
        if providers is None:
            providers = list(self.trackers.keys())
        
        # Start timing
        task_start_time = time.time()
        
        # Initialize results structure
        results = {
            "video_path": video_path,
            "video_name": Path(video_path).stem,
            "providers": {},
            "summary": {
                "people_tracked": {},
                "vehicles_tracked": {},
                "processing_time": {},
                "cost": {}
            }
        }
        
        # Process video with each provider
        for provider in providers:
            if provider not in self.trackers:
                logger.warning(f"No tracker registered for provider: {provider}")
                continue
            
            logger.info(f"Processing video with provider: {provider}")
            provider_start_time = time.time()
            
            # Process video with provider's tracker
            tracker_results = self.trackers[provider].process_video(video_path)
            
            # Calculate provider metrics
            provider_time = time.time() - provider_start_time
            
            # Add to results
            results["providers"][provider] = tracker_results
            
            # Add to summary
            people_count = tracker_results.get("summary", {}).get("people_count", 0)
            vehicle_count = tracker_results.get("summary", {}).get("vehicle_count", 0)
            processing_cost = tracker_results.get("summary", {}).get("cost", 0.0)
            
            results["summary"]["people_tracked"][provider] = people_count
            results["summary"]["vehicles_tracked"][provider] = vehicle_count
            results["summary"]["processing_time"][provider] = provider_time
            results["summary"]["cost"][provider] = processing_cost
        
        # Calculate overall processing time
        results["total_processing_time"] = time.time() - task_start_time
        
        return results
