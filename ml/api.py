"""
API for video processing.
Provides endpoints for uploading and processing videos.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

from main import VideoPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Video Processing API",
    description="API for video object detection and tracking with AWS, Azure, and local models",
    version="1.0.0"
)

# Create uploads directory
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize video pipeline
pipeline = VideoPipeline()

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Video Processing API is running"}

# Add the endpoints with /api prefix that match the backend expectations
@app.post("/api/videos/process")
async def process_video_api(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    """
    Process a video with API endpoint matching the backend expectations.
    """
    # Save uploaded file
    video_path = UPLOAD_DIR / video.filename
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {str(e)}")
    
    # Start processing in background
    background_tasks.add_task(pipeline.process_video, str(video_path))
    
    # Return immediate response
    return JSONResponse(
        status_code=202,
        content={
            "message": "Video uploaded and scheduled for processing",
            "video_path": str(video_path),
            "status": "processing"
        }
    )

@app.get("/api/videos/{processing_id}/status")
def get_video_status(processing_id: str):
    """
    Get status of a video processing job with API endpoint matching the backend expectations.
    """
    # In a real implementation, you would check the status from a database or queue
    # This is a placeholder that returns a success response
    return {
        "processing_id": processing_id, 
        "status": "completed", 
        "result": {
            "aws": {"detected_objects": 15, "processing_time": 2.5},
            "azure": {"detected_objects": 12, "processing_time": 3.1},
            "yolo": {"detected_objects": 18, "processing_time": 1.8}
        }
    }

@app.get("/api/metrics")
def get_metrics():
    """
    Get metrics for all providers.
    """
    return {
        "aws": {"avg_objects": 14.5, "avg_time": 2.3, "cost_per_minute": 0.12},
        "azure": {"avg_objects": 12.2, "avg_time": 2.8, "cost_per_minute": 0.10},
        "yolo": {"avg_objects": 16.8, "avg_time": 1.9, "cost_per_minute": 0.05}
    }

@app.get("/api/metrics/{source}")
def get_metrics_by_source(source: str):
    """
    Get metrics for a specific provider.
    """
    metrics = {
        "aws": {"avg_objects": 14.5, "avg_time": 2.3, "cost_per_minute": 0.12},
        "azure": {"avg_objects": 12.2, "avg_time": 2.8, "cost_per_minute": 0.10},
        "yolo": {"avg_objects": 16.8, "avg_time": 1.9, "cost_per_minute": 0.05}
    }
    
    if source not in metrics:
        raise HTTPException(status_code=404, detail=f"Source '{source}' not found")
    
    return metrics[source]

# Keep the original endpoints for backward compatibility
@app.post("/process/")
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    """
    Upload and process a video.
    
    The video will be processed using AWS Rekognition, Azure AI Vision, and local YOLO model.
    Processing includes:
    - Task A: Image Analysis with Object Detection
    - Task B: Video Processing with Object Tracking
    """
    # Save uploaded file
    video_path = UPLOAD_DIR / video.filename
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {str(e)}")
    
    # Start processing in background
    background_tasks.add_task(pipeline.process_video, str(video_path))
    
    # Return immediate response
    return JSONResponse(
        status_code=202,
        content={
            "message": "Video uploaded and scheduled for processing",
            "video_path": str(video_path),
            "status": "processing"
        }
    )

@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    """
    Get status of a processing job.
    
    Args:
        job_id: ID of the processing job
    """
    # In a real implementation, you would check the status from a database or queue
    # This is a placeholder
    return {"job_id": job_id, "status": "processing"}

def start():
    """Start the API server."""
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start() 