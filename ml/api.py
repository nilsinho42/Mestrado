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

@app.post("/process/")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload and process a video.
    
    The video will be processed using AWS Rekognition, Azure AI Vision, and local YOLO model.
    Processing includes:
    - Task A: Image Analysis with Object Detection
    - Task B: Video Processing with Object Tracking
    """
    # Save uploaded file
    video_path = UPLOAD_DIR / file.filename
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
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