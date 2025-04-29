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
            return []
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return []
        
        # Process image with detector
        start_time = time.time()
        detections = self.detectors[provider].detect(image)

        image_id = image_path.split("_")[-1].split(".")[0]

        latency = time.time() - start_time
        logger.info(f"[{provider}][{image_id}] Latency: {latency:.3f}s.")
        
        return detections, latency, image