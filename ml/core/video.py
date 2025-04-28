"""
Video processing functionality.
Provides utilities for video frame extraction, processing, and analysis.
"""

import cv2
import numpy as np
import time
import logging
import subprocess
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

logger = logging.getLogger(__name__)

def get_video_rotation(video_path: str) -> int:
    """
    Detect video rotation from metadata.
    
    Note: The rotation value refers to how much the image needs to be rotated
    to be displayed correctly. For example, a rotation of 90 means the device
    was rotated 90 degrees counterclockwise, so we need to rotate the image
    90 degrees counterclockwise to correct it.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Rotation angle in degrees (0, 90, 180, 270, or -90)
    """
    logger.info(f"Detecting rotation for video: {video_path}")
    
    # First try using ffmpeg-python if available
    if ffmpeg:
        try:
            logger.info(f"Attempting rotation detection using ffmpeg-python library")
            probe = ffmpeg.probe(video_path)
            # Look for rotation metadata in video stream
            for stream in probe['streams']:
                if stream['codec_type'] == 'video':
                    # Check different possible metadata locations
                    # 1. Standard rotation tag
                    rotation = stream.get('tags', {}).get('rotate', 0)
                    if rotation:
                        logger.info(f"Found rotation metadata in tags: {rotation} degrees")
                        return int(rotation)
                    
                    # 2. Side data rotation matrix
                    side_data = stream.get('side_data_list', [])
                    for data in side_data:
                        if 'rotation' in data:
                            logger.info(f"Found rotation in side_data: {data['rotation']} degrees")
                            return int(data['rotation'])
                        
            logger.info("No rotation metadata found via ffmpeg-python")
        except Exception as e:
            logger.warning(f"Error using ffmpeg-python for rotation detection: {e}")
    else:
        logger.info("ffmpeg-python library not available")
    
    # Fallback to ffprobe command line
    try:
        logger.info(f"Attempting rotation detection using ffprobe command-line tool")
        # Check if ffprobe is available
        try:
            subprocess.check_output(["ffprobe", "-version"], stderr=subprocess.STDOUT)
            logger.info("ffprobe command is available")
        except Exception as e:
            logger.warning(f"ffprobe command not available: {e}")
            raise  # Re-raise to move to next fallback
            
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
               "-show_entries", "stream_tags=rotate", "-of", "json", video_path]
        logger.debug(f"Running ffprobe command: {' '.join(cmd)}")
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
        import json
        metadata = json.loads(output)
        logger.debug(f"ffprobe output: {metadata}")
        rotation = metadata.get('streams', [{}])[0].get('tags', {}).get('rotate')
        if rotation:
            logger.info(f"Found rotation from ffprobe: {rotation} degrees")
            return int(rotation)
        else:
            logger.info("No rotation metadata found via ffprobe")
    except Exception as e:
        logger.warning(f"Error using ffprobe for rotation detection: {e}")
    
    # Final fallback - check dimensions
    try:
        logger.info("Attempting rotation detection based on video dimensions")
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        logger.info(f"Video dimensions: {width}x{height}")
        # If height > width and significantly so, it might be portrait
        if height > width * 1.2:  # 20% threshold
            logger.info(f"Portrait video detected based on dimensions (h/w ratio: {height/width:.2f})")
            return 90  # Assume needs 90 degree rotation for portrait
        else:
            logger.info(f"Landscape or square video detected (h/w ratio: {height/width:.2f})")
    except Exception as e:
        logger.warning(f"Error checking video dimensions: {e}")
    
    logger.info("No rotation detected, using default 0 degrees")
    return 0


def apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """
    Apply rotation to a frame.
    
    Args:
        frame: Image as numpy array
        rotation: Rotation angle in degrees
        
    Returns:
        Rotated frame
    """
    if rotation == 0:
        return frame
    elif rotation == 90:
        # Apply counterclockwise rotation for 90 degrees
        # Most mobile devices need counterclockwise rotation when metadata says 90
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == -90 or rotation == 270:
        # Apply clockwise rotation for -90 or 270 degrees
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    else:
        logger.warning(f"Unsupported rotation angle: {rotation}")
        return frame


class VideoProcessor:
    """Base video processor for handling frame extraction and processing."""

    # KEEP THIS FUNCTION
    def __init__(self, output_dir: str = "./data/object_detection/images"):
        """
        Initialize video processor.
        
        Args:
            output_dir: Directory to save extracted frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Video processor initialized with output directory: {self.output_dir}")
    
    # KEEP THIS FUNCTION
    def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[str], Dict[str, Any]]:
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
        
        # Detect rotation from video metadata
        rotation = get_video_rotation(video_path)
        if rotation != 0:
            logger.info(f"Detected video rotation: {rotation} degrees for {video_path}")
        
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
        
        # Create a subfolder based on video filename
        video_name = Path(video_path).stem
        frames_dir = Path(self.output_dir)
        frames_dir.mkdir(exist_ok=True)
        
        try:
            frame_count = 0

            while frame_count < total_frames:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                
                if ret:
                    # Apply rotation correction if needed
                    if rotation != 0:
                        frame = apply_rotation(frame, rotation)
                    
                    # Save frame to the output directory with a timestamp
                    frame_filename = f"{video_name}_frame_{frame_count:04d}.jpg"

                    frame_path = str(frames_dir / frame_filename)
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)

                    frames.append(frame)
                
                # Move to next frame
                frame_count += 1
            
            video_info = {
                "total_frames": total_frames,
                "fps": original_fps,
                "duration": duration,
                "width": width,
                "height": height,
                "video_path": video_path,
                "video_name": Path(video_path).stem,
                "rotation": rotation
            }

            if rotation != 0:
                logger.info(f"Applied {rotation} degree rotation correction to frames")
            
            return frames, frame_paths, video_info
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
        finally:
            cap.release()
    
    # def process_video(self, video_path: str, fps_reduction_factor: int = 5, 
    #                   save_frames: bool = True) -> Dict[str, Any]:
    #     """
    #     Process video and extract frames for analysis.
    #     Base implementation that should be extended by specific processors.
        
    #     Args:
    #         video_path: Path to the video file
    #         fps_reduction_factor: Factor by which to reduce FPS
    #         save_frames: Whether to save frames to disk
            
    #     Returns:
    #         Dictionary with processing results and video info
    #     """
    #     # Extract frames from video
    #     frames, frame_paths, video_info = self.extract_frames(
    #         video_path, 
    #         fps_reduction_factor=fps_reduction_factor,
    #         save_frames=save_frames
    #     )
        
    #     # Create basic results structure
    #     results = {
    #         "video_info": video_info,
    #         "frame_paths": frame_paths,
    #         "status": "completed"
    #     }
        
    #     return results
    
    # def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]], 
    #                    confidence_threshold: float = 0.3) -> np.ndarray:
    #     """
    #     Draw detection bounding boxes on an image.
        
    #     Args:
    #         image: Image as numpy array
    #         detections: List of detection dictionaries
    #         confidence_threshold: Minimum confidence to draw
            
    #     Returns:
    #         Image with drawn detections
    #     """
    #     image_height, image_width = image.shape[:2]
    #     result_image = image.copy()
        
    #     # Color map for different classes
    #     color_map = {
    #         "person": (0, 255, 0),    # Green for people
    #         "car": (255, 0, 0),       # Blue for cars
    #         "truck": (255, 0, 255),   # Magenta for trucks
    #         "bus": (255, 165, 0),     # Orange for buses
    #         "motorcycle": (0, 255, 255)  # Yellow for motorcycles
    #     }
        
    #     for detection in detections:
    #         confidence = detection.get("confidence", 0)
    #         if confidence < confidence_threshold:
    #             continue
            
    #         # Get bounding box
    #         bbox = detection.get("bbox", [0, 0, 0, 0])
    #         x1, y1, x2, y2 = bbox
            
    #         # Convert from normalized coordinates if needed
    #         if max(bbox) <= 1.0:
    #             x1 = int(x1 * image_width)
    #             y1 = int(y1 * image_height)
    #             x2 = int(x2 * image_width)
    #             y2 = int(y2 * image_height)
            
    #         # Get class and determine color
    #         class_name = detection.get("detection_type", "unknown")
    #         color = color_map.get(class_name.lower(), (0, 0, 255))  # Default to red
            
    #         # Draw bounding box
    #         cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
    #         # Draw label
    #         label = f"{class_name}: {confidence:.2f}"
    #         label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    #         y1 = max(y1, label_size[1])
    #         cv2.rectangle(result_image, (x1, y1 - label_size[1] - baseline), 
    #                      (x1 + label_size[0], y1), color, -1)
    #         cv2.putText(result_image, label, (x1, y1 - baseline), 
    #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    #     return result_image


class ImageAnalysisProcessor(VideoProcessor):
    """
    Processor for Task A: Image Analysis with Object Detection.
    Samples frames from video and processes them with different providers.
    """
    # KEEP THIS FUNCTION
    def __init__(self, output_dir: str = "./data/object_detection/images"):
        super().__init__(output_dir=output_dir)
        # Will be initialized with provider-specific detectors
        self.detectors = {}
    
    # KEEP THIS FUNCTION
    def register_detector(self, provider: str, detector: Any) -> None:
        """Register a detector for a specific provider."""
        self.detectors[provider] = detector
        logger.info(f"Registered detector for provider: {provider}")
    
    # KEEP THIS FUNCTION
    def process_image(self, image_path: str, provider: str) -> List[Dict[str, Any]]:
        """
        Process a single image using the specified provider's detector.
        
        Args:
            image_path: Path to the image file
            provider: Provider name for detection ('local', 'aws', 'azure')
            
        Returns:
            List of detection dictionaries
        """
        if provider not in self.detectors:
            logger.warning(f"No detector registered for provider: {provider}")
            return []
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return []
        
        # Process image with detector
        logger.info(f"Processing image with {provider} detector: {image_path}")
        start_time = time.time()
        detections = self.detectors[provider].detect(image)

        latency = time.time() - start_time
        logger.info(f"Processed image with {provider} detector: {image_path}, latency: {latency:.3f}s, detections: {len(detections)}")
        return detections, latency, image
    
    # def process_video(self, video_path: str, fps_reduction_factor: int = 5, 
    #                  providers: List[str] = None) -> Dict[str, Any]:
    #     """
    #     Process video for Task A (Image Analysis).
        
    #     Args:
    #         video_path: Path to the video file
    #         fps_reduction_factor: Factor by which to reduce FPS
    #         providers: List of providers to use for detection
            
    #     Returns:
    #         Dictionary with detection results for each provider
    #     """
    #     if providers is None:
    #         providers = list(self.detectors.keys())
        
    #     # Start timing
    #     task_start_time = time.time()
        
    #     # Extract frames
    #     frames, frame_paths, video_info = self.extract_frames(
    #         video_path, 
    #         fps_reduction_factor=fps_reduction_factor,
    #         save_frames=True
    #     )
        
    #     # Initialize results for each provider
    #     results = {
    #         "video_info": video_info,
    #         "providers": {},
    #         "summary": {
    #             "people_count": {},
    #             "vehicle_count": {},
    #             "avg_latency": {}
    #         }
    #     }
        
    #     # Process frames with each provider
    #     for provider in providers:
    #         if provider not in self.detectors:
    #             logger.warning(f"No detector registered for provider: {provider}")
    #             continue
            
    #         logger.info(f"Processing frames with provider: {provider}")
    #         provider_start_time = time.time()
    #         provider_results = []
            
    #         # Track metrics
    #         total_latency = 0
    #         people_count = 0
    #         vehicle_count = 0
            
    #         # Process each frame
    #         for i, (frame, frame_path) in enumerate(zip(frames, frame_paths)):
    #             # Process frame with provider
    #             frame_start_time = time.time()
    #             logger.info(f"Processing frame {i} with provider {provider}: {frame_path}")
    #             detections = self.detectors[provider].process_image(frame, frame_path)
    #             frame_latency = time.time() - frame_start_time
    #             total_latency += frame_latency
                
    #             # Log detection info
    #             logger.info(f"{provider} detected {len(detections)} objects in frame {i}, latency: {frame_latency:.3f}s")
                
    #             # Count people and vehicles
    #             frame_people = sum(1 for d in detections if 
    #                              d.get("detection_type", d.get("class_name", "")).lower() in 
    #                              ["person", "people", "pedestrian", "man", "woman", "child"])
    #             frame_vehicles = sum(1 for d in detections if 
    #                                d.get("detection_type", d.get("class_name", "")).lower() in 
    #                                ["car", "truck", "bus", "motorcycle", "vehicle", "automobile", "transportation"])
                
    #             logger.info(f"{provider} frame {i}: found {frame_people} people, {frame_vehicles} vehicles")
                
    #             people_count += frame_people
    #             vehicle_count += frame_vehicles
                
    #             # Store results for this frame
    #             frame_result = {
    #                 "frame_number": i,
    #                 "frame_path": frame_path,
    #                 "latency": frame_latency,
    #                 "detections": detections,
    #                 "people_count": frame_people,
    #                 "vehicle_count": frame_vehicles,
    #                 "image_id": Path(frame_path).stem
    #             }
    #             provider_results.append(frame_result)
            
    #         # Calculate provider-level metrics
    #         provider_time = time.time() - provider_start_time
    #         avg_latency = total_latency / len(frames) if frames else 0
            
    #         # Store provider results
    #         results["providers"][provider] = {
    #             "results": provider_results,
    #             "total_time": provider_time,
    #             "avg_latency": avg_latency,
    #             "frame_count": len(frames),
    #             "people_count": people_count,
    #             "vehicle_count": vehicle_count
    #         }
            
    #         # Add to summary
    #         results["summary"]["people_count"][provider] = people_count
    #         results["summary"]["vehicle_count"][provider] = vehicle_count
    #         results["summary"]["avg_latency"][provider] = avg_latency
        
    #     # Calculate overall processing time
    #     results["total_processing_time"] = time.time() - task_start_time
        
    #     return results


# class VideoTrackingProcessor(VideoProcessor):
#     """
#     Processor for Task B: Video Processing with Object Tracking.
#     Uploads video to cloud providers and tracks objects using various services.
#     """
    
#     def __init__(self, output_dir: str = "./data/object_detection/tracks"):
#         super().__init__(output_dir=output_dir)
#         # Will be initialized with provider-specific trackers
#         self.trackers = {}
#         # Storage providers for uploading/downloading videos
#         self.storage_providers = {}
    
#     def register_tracker(self, provider: str, tracker: Any) -> None:
#         """Register a tracker for a specific provider."""
#         self.trackers[provider] = tracker
#         logger.info(f"Registered tracker for provider: {provider}")
    
#     def register_storage_provider(self, provider: str, storage_provider: Any) -> None:
#         """Register a storage provider for a specific provider."""
#         self.storage_providers[provider] = storage_provider
#         logger.info(f"Registered storage provider for provider: {provider}")
    
#     def process_video(self, video_path: str, 
#                      providers: List[str] = None) -> Dict[str, Any]:
#         """
#         Process video for Task B (Video Tracking).
        
#         Args:
#             video_path: Path to the video file
#             providers: List of providers to use for tracking
            
#         Returns:
#             Dictionary with tracking results for each provider
#         """
#         if providers is None:
#             providers = list(self.trackers.keys())
        
#         # Start timing
#         task_start_time = time.time()
        
#         # Initialize results structure
#         results = {
#             "video_path": video_path,
#             "video_name": Path(video_path).stem,
#             "providers": {},
#             "summary": {
#                 "people_tracked": {},
#                 "vehicles_tracked": {},
#                 "processing_time": {},
#                 "cost": {}
#             }
#         }
        
#         # Process video with each provider
#         for provider in providers:
#             if provider not in self.trackers:
#                 logger.warning(f"No tracker registered for provider: {provider}")
#                 continue
            
#             logger.info(f"Processing video with provider: {provider}")
#             provider_start_time = time.time()
            
#             # Process video with provider's tracker
#             tracker_results = self.trackers[provider].process_video(video_path)
            
#             # Calculate provider metrics
#             provider_time = time.time() - provider_start_time
            
#             # Add to results
#             results["providers"][provider] = tracker_results
            
#             # Add to summary
#             people_count = tracker_results.get("summary", {}).get("people_count", 0)
#             vehicle_count = tracker_results.get("summary", {}).get("vehicle_count", 0)
#             processing_cost = tracker_results.get("summary", {}).get("cost", 0.0)
            
#             results["summary"]["people_tracked"][provider] = people_count
#             results["summary"]["vehicles_tracked"][provider] = vehicle_count
#             results["summary"]["processing_time"][provider] = provider_time
#             results["summary"]["cost"][provider] = processing_cost
        
#         # Calculate overall processing time
#         results["total_processing_time"] = time.time() - task_start_time
        
#         return results
