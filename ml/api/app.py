import os
import sys
import time
import uuid
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import aiofiles
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from video_pipeline import VideoPipeline
from db_utils import Database

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Video Processing API",
    description="API for processing videos with object detection and tracking",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temporary directory for uploads
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Create output directory for results
OUTPUT_DIR = Path("./data/object_detection/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize pipeline and database
pipeline = VideoPipeline(output_dir=str(OUTPUT_DIR))
db = Database()

# Dictionary to store background task status
task_status = {}

# Response models
class JobStatus(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    progress: Optional[float] = None

class VideoDetail(BaseModel):
    video_id: str
    filename: str
    size_bytes: int
    upload_time: str
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None

async def process_video_task(video_path: str, job_id: str):
    """Background task to process video."""
    try:
        # Update task status
        task_status[job_id] = {
            'status': 'processing',
            'message': 'Processing video...',
            'start_time': datetime.now().isoformat(),
            'progress': 0.0
        }
        
        # Process video
        results = pipeline.process_video(video_path)
        
        # Update task status
        task_status[job_id] = {
            'status': 'completed',
            'message': 'Video processing completed',
            'end_time': datetime.now().isoformat(),
            'progress': 100.0,
            'results': results
        }
        
        # Save results to file
        output_file = OUTPUT_DIR / f"results_{job_id}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Video processing completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Error processing video for job {job_id}: {e}", exc_info=True)
        
        # Update task status
        task_status[job_id] = {
            'status': 'failed',
            'message': f'Error processing video: {str(e)}',
            'end_time': datetime.now().isoformat(),
            'progress': 0.0
        }

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Video Processing API"}

@app.post("/api/videos/upload", response_model=JobStatus)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    detect_objects: bool = Form(True),
    track_objects: bool = Form(True)
):
    """Upload and process video."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file extension
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create destination file path
        dest_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
        
        # Save uploaded file
        async with aiofiles.open(dest_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Initialize task status
        task_status[job_id] = {
            'status': 'uploading',
            'message': 'Video uploaded, waiting for processing',
            'start_time': datetime.now().isoformat(),
            'progress': 0.0
        }
        
        # Start background processing task
        background_tasks.add_task(process_video_task, str(dest_path), job_id)
        
        return JSONResponse(status_code=202, content={
            'job_id': job_id,
            'status': 'uploading',
            'message': 'Video uploaded, processing started',
            'start_time': task_status[job_id]['start_time']
        })
        
    except Exception as e:
        logger.error(f"Error uploading video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status."""
    if job_id not in task_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return task_status[job_id]

@app.get("/api/jobs")
async def get_all_jobs():
    """Get all jobs status."""
    return task_status

@app.get("/api/results/{job_id}")
async def get_job_results(job_id: str):
    """Get job results."""
    if job_id not in task_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = task_status[job_id]
    
    if status['status'] != 'completed':
        return {
            'job_id': job_id,
            'status': status['status'],
            'message': status['message'],
            'progress': status.get('progress', 0.0)
        }
    
    if 'results' in status:
        return status['results']
    
    # Try to load results from file
    output_file = OUTPUT_DIR / f"results_{job_id}.json"
    if output_file.exists():
        with open(output_file, 'r') as f:
            results = json.load(f)
        return results
    
    raise HTTPException(status_code=404, detail="Results not found")

@app.get("/api/metrics")
async def get_metrics():
    """Get all metrics from database."""
    try:
        metrics = db.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/{source}")
async def get_metrics_by_source(source: str):
    """Get metrics for a specific source from database."""
    try:
        metrics = db.get_metrics(source=source)
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics for source {source}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/detection/{image_id}")
async def get_detection_results(image_id: str):
    """Get detection results for a specific image."""
    try:
        results = db.get_detection_results(image_id=image_id)
        if not results:
            raise HTTPException(status_code=404, detail="Detection results not found")
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting detection results for image {image_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tracking/{video_id}")
async def get_tracking_results(video_id: str):
    """Get tracking results for a specific video."""
    try:
        results = db.get_tracking_results(video_id=video_id)
        if not results:
            raise HTTPException(status_code=404, detail="Tracking results not found")
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tracking results for video {video_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthcheck")
async def healthcheck():
    """Health check endpoint."""
    return {"status": "healthy"}

def start():
    """Start the server."""
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start() 