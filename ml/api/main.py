from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import jwt
import os
from datetime import datetime, timedelta
import time
import json
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
import shutil
import uuid

from .processing import VideoProcessor
from .storage import ProcessingStorage
from .tracing import setup_tracing
from .metrics import MetricsCollector
from .discovery import ServiceDiscovery
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Service API")
security = HTTPBearer()

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET")
SERVICE_NAME = os.getenv("SERVICE_NAME", "ml_service")
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov5s.pt")
VIDEO_UPLOAD_DIR = os.getenv("VIDEO_UPLOAD_DIR", "uploads")
MAX_VIDEO_SIZE = int(os.getenv("MAX_VIDEO_SIZE", "500")) * 1024 * 1024  # 500MB default
ALLOWED_VIDEO_TYPES = ["video/mp4", "video/avi", "video/mov", "video/quicktime"]

# Ensure upload directory exists
Path(VIDEO_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Initialize video processor and storage
try:
    video_processor = VideoProcessor(MODEL_PATH)
    storage = ProcessingStorage()
    logger.info(f"Video processor initialized with model: {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to initialize services: {str(e)}")
    raise

metrics = MetricsCollector()
service_discovery = ServiceDiscovery()

# Set up tracing
setup_tracing(app, settings.SERVICE_NAME)

@app.on_event("startup")
async def startup_event():
    """Register service with Consul on startup."""
    try:
        await service_discovery.register(
            port=settings.PORT,
            tags=["ml", "video-processing"]
        )
        logger.info("Service registered with Consul")
    except Exception as e:
        logger.error(f"Failed to register service with Consul: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Deregister service from Consul on shutdown."""
    try:
        await service_discovery.deregister()
        logger.info("Service deregistered from Consul")
    except Exception as e:
        logger.error(f"Failed to deregister service from Consul: {str(e)}")

class VideoRequest(BaseModel):
    video_path: str
    user_id: int

class DetectionResult(BaseModel):
    frame_number: int
    detection_type: str
    confidence: float
    bbox: List[float]
    metadata: Optional[Dict[str, Any]] = None

class ServiceComparison(BaseModel):
    processing_time: float
    memory_usage: float
    accuracy: float
    metadata: Optional[Dict[str, Any]] = None

class PerformanceMetric(BaseModel):
    metric_type: str
    value: float
    metadata: Optional[Dict[str, Any]] = None

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def save_upload_file(upload_file: UploadFile, user_id: int) -> str:
    """Save uploaded file and return the path."""
    try:
        # Validate file type
        if upload_file.content_type not in ALLOWED_VIDEO_TYPES:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Generate unique filename
        file_extension = Path(upload_file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = Path(VIDEO_UPLOAD_DIR) / str(user_id) / unique_filename
        
        # Create user directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error saving upload file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save file")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def process_video_background(processing_id: str, video_path: str, user_id: int):
    """Background task to process video with retry mechanism."""
    try:
        # Update status to processing
        storage.update_status(processing_id, "processing")
        
        # Process video
        detection_results, comparison, metric = video_processor.process_video(video_path)
        
        # Update status with results
        storage.update_status(
            processing_id,
            "completed",
            results={
                "detection_results": detection_results,
                "comparison": comparison,
                "metric": metric
            }
        )
        
        logger.info(f"Successfully processed video {processing_id}")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing video {processing_id}: {error_msg}")
        storage.update_status(processing_id, "failed", error=error_msg)
        raise

@app.post("/upload_video")
async def upload_video(
    background_tasks: BackgroundTasks,
    token: dict = Depends(verify_token),
    file: UploadFile = File(...)
):
    try:
        # Get user ID from token
        user_id = token.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing user_id")
        
        # Save uploaded file
        video_path = await save_upload_file(file, user_id)
        
        # Create processing ID
        processing_id = f"proc_{int(time.time() * 1000)}"
        
        # Create initial status
        storage.create_status(processing_id, video_path, user_id)
        
        # Add background task
        background_tasks.add_task(
            process_video_background,
            processing_id,
            video_path,
            user_id
        )
        
        return {
            "processing_id": processing_id,
            "status": "pending",
            "video_path": video_path
        }
        
    except Exception as e:
        logger.error(f"Error handling video upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_video")
async def process_video(
    request: VideoRequest,
    background_tasks: BackgroundTasks,
    token: dict = Depends(verify_token)
):
    try:
        # Validate video path
        video_path = Path(request.video_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Create processing ID
        processing_id = f"proc_{int(time.time() * 1000)}"
        
        # Create initial status
        storage.create_status(processing_id, str(video_path), request.user_id)
        
        # Add background task
        background_tasks.add_task(
            process_video_background,
            processing_id,
            str(video_path),
            request.user_id
        )
        
        return {
            "processing_id": processing_id,
            "status": "pending"
        }
        
    except Exception as e:
        logger.error(f"Error initiating video processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video_status/{processing_id}")
async def get_video_status(
    processing_id: str,
    token: dict = Depends(verify_token)
):
    try:
        status = storage.get_status(processing_id)
        if not status:
            raise HTTPException(status_code=404, detail="Processing ID not found")
        
        if status["status"] == "failed":
            raise HTTPException(status_code=500, detail=status["error"])
        
        return {
            "status": status["status"],
            "processing_id": processing_id,
            **status
        }
        
    except Exception as e:
        logger.error(f"Error getting video status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user_statuses")
async def get_user_statuses(
    limit: int = 10,
    token: dict = Depends(verify_token)
):
    try:
        user_id = token.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing user_id")
        
        statuses = storage.get_user_statuses(user_id, limit)
        return {"statuses": statuses}
        
    except Exception as e:
        logger.error(f"Error getting user statuses: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/detection_stats")
async def get_detection_stats(
    token: dict = Depends(verify_token)
):
    try:
        user_id = token.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing user_id")
        
        # Get user's completed processes
        statuses = storage.get_user_statuses(user_id)
        completed_statuses = [s for s in statuses if s["status"] == "completed"]
        
        # Calculate stats
        total_detections = 0
        total_confidence = 0
        detection_types = {}
        
        for status in completed_statuses:
            if status["results"] and "detection_results" in status["results"]:
                for detection in status["results"]["detection_results"]:
                    total_detections += 1
                    total_confidence += detection["confidence"]
                    detection_type = detection["detection_type"]
                    detection_types[detection_type] = detection_types.get(detection_type, 0) + 1
        
        return {
            "total_detections": total_detections,
            "average_confidence": total_confidence / total_detections if total_detections > 0 else 0,
            "detection_types": detection_types
        }
        
    except Exception as e:
        logger.error(f"Error calculating detection stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for service discovery and monitoring."""
    try:
        # Check if model is loaded
        if not hasattr(video_processor, 'model'):
            raise RuntimeError("Model not loaded")
        
        # Check storage
        storage.get_status("test")  # This will test the database connection
        
        return {
            "status": "healthy",
            "service": settings.SERVICE_NAME,
            "model_loaded": True,
            "device": video_processor.device,
            "storage": "ok"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": settings.SERVICE_NAME,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 