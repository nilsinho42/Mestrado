"""
API for video processing.
Provides endpoints for uploading and processing videos.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Depends
from fastapi.responses import JSONResponse
import uvicorn
import json
import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uuid
import numpy as np

from main import VideoPipeline

# Initialize logger
logger = logging.getLogger(__name__)

# Configure lifespan context to keep the server running
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Application startup
    logger.info("Application starting up...")
    yield
    # Application shutdown
    logger.info("Application shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Video Processing API",
    description="API for video object detection and tracking with AWS, Azure, and local models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["Content-Type", "Content-Length", "Access-Control-Allow-Origin"],
    max_age=86400,  # Cache CORS responses for 24 hours
)

# Create uploads directory
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize video pipeline
pipeline = VideoPipeline()

# Define ID generation and normalization functions
def generate_processing_id():
    """Generate a simple, consistent processing ID."""
    return f"proc_{uuid.uuid4().hex}"

def normalize_id(processing_id):
    """
    Process the ID received from frontend.
    Always preserves the original ID without modification.
    """
    if processing_id:
        # Always preserve the original ID from the frontend
        logger.info(f"Using original processing ID from frontend: {processing_id}")
        return processing_id
    else:
        # Only generate a new ID if none was provided
        # Use timestamp format for compatibility with frontend
        new_id = f"proc_{int(time.time() * 1000000000)}"
        logger.info(f"No ID provided, generated new timestamp ID: {new_id}")
        return new_id

@app.get("/")
def read_root():
    """Root endpoint."""
    return JSONResponse(
        content={"message": "Video Processing API is running"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )

# Add the endpoints with /api prefix that match the backend expectations
@app.post("/api/videos/process")
async def process_video_api(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    processing_id: Optional[str] = Form(None),
    expected_vehicles: int = Form(0),
    expected_people: int = Form(0)
):
    """
    Process a video with API endpoint matching the backend expectations.
    Uses FastAPI's background tasks for async processing.
    
    Parameters:
        - video: Uploaded video file
        - processing_id: Optional ID for the processing job
        - expected_vehicles: Expected number of vehicles in the video
        - expected_people: Expected number of people in the video
    """
    logger.info(f"Using processing ID from frontend: {processing_id}")
    logger.info(f"Expected vehicles: {expected_vehicles}, Expected people: {expected_people}")

    # Save uploaded file with processing ID in the name
    original_filename = video.filename
    file_extension = os.path.splitext(original_filename)[1]
    video_path = UPLOAD_DIR / f"{processing_id}{file_extension}"
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        logger.info(f"Video saved to {video_path}")
    except Exception as e:
        logger.error(f"Could not save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {str(e)}")
    
    # Start processing in background
    try:
        # Add task to background queue with the exact ID as provided
        background_tasks.add_task(
            _process_video_in_background, 
            str(video_path), 
            processing_id,
            expected_vehicles,
            expected_people
        )
        logger.info(f"Added processing task for {processing_id} to background queue")
        
        # Return immediate response with the exact processing ID
        return JSONResponse(
            status_code=202,
            content={
                "message": "Video uploaded and scheduled for processing",
                "video_path": str(video_path),
                "processing_id": processing_id,
                "status": "processing",
                "expected_vehicles": expected_vehicles,
                "expected_people": expected_people
            }
        )
    except Exception as e:
        logger.error(f"Error scheduling video processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error scheduling video processing: {str(e)}")

async def _process_video_in_background(
    video_path: str, 
    processing_id: str,
    expected_vehicles: int = 0,
    expected_people: int = 0
):
    """
    Process video in background and save results.
    
    Args:
        video_path: Path to video file
        processing_id: Processing ID for this job (normalized format)
        expected_vehicles: Expected number of vehicles in the video
        expected_people: Expected number of people in the video
    """
    try:
        logger.info(f"Processing video in background: {video_path}, ID: {processing_id}")
        logger.info(f"Expected vehicles: {expected_vehicles}, Expected people: {expected_people}")
        
        # Process video
        logger.info(f"Starting video processing for {processing_id}")
        results = pipeline.process_video(
            video_path, 
            processing_id,
            expected_vehicles=expected_vehicles,
            expected_people=expected_people
        )
        
        # Add the processing ID to the results
        if isinstance(results, dict):
            results["processing_id"] = processing_id
            results["expected_vehicles"] = expected_vehicles
            results["expected_people"] = expected_people
        
        # Recursively convert numpy arrays to Python types for JSON serialization
        def convert_numpy_to_python(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_to_python(item) for item in obj]
            else:
                return str(obj)
            
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = convert_numpy_to_python(results)
        logger.info(f"Results:")
        logger.info(repr(serializable_results))

        # Save results with a single, consistent filename
        result_path = Path("./data/results") / f"results_{processing_id}.json"
        Path("./data/results").mkdir(parents=True, exist_ok=True)
        
        with open(result_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Results saved to {result_path}")
        
        logger.info(f"Background processing completed for {processing_id}")
    except Exception as e:
        logger.error(f"Error in background processing for {processing_id}: {e}", exc_info=True)
        
        # Save error results
        error_result = {
            "error": str(e),
            "processing_id": processing_id,
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }
        
        # Ensure results directory exists
        Path("./data/results").mkdir(parents=True, exist_ok=True)
        
        # Save error results
        error_path = Path("./data/results") / f"results_{processing_id}.json"
        with open(error_path, "w") as f:
            json.dump(error_result, f, indent=2)
        logger.info(f"Saved error results to {error_path}")

@app.get("/api/videos/{processing_id}/status")
def get_video_status(processing_id: str, include_predictions: bool = False):
    """
    Get status of a video processing job.
    
    Parameters:
        - processing_id: ID of the processing job
        - include_predictions: Whether to include predictions in the response (default: False)
    """
    logger.info(f"Checking status for processing_id: {processing_id}, include_predictions: {include_predictions}")
    
    # Look for results file with the exact ID only
    result_file = Path("./data/results") / f"results_{processing_id}.json"
    
    if result_file.exists():
        try:
            with open(result_file, "r") as f:
                results = json.load(f)
            logger.info(f"Found results at {result_file}")
            
            # Reset counter on success
            if hasattr(app.state, "poll_counters"):
                if processing_id in app.state.poll_counters:
                    del app.state.poll_counters[processing_id]
            
            # Create a response object
            predictions_count = len(results.get("predictions", [])) if isinstance(results, dict) and "predictions" in results else 0
            
            response_data = {
                "processing_id": processing_id, 
                "status": "completed", 
                "result": {
                    "processing_id": processing_id,
                    "success": True,
                    "predictions_count": predictions_count,
                    "api_guidance": {
                        "preferred_endpoints": [
                            f"/api/videos/{processing_id}/metadata",
                            f"/api/videos/{processing_id}/predictions?chunk_size=100&offset=0"
                        ],
                        "message": "For better performance with large result sets, use the metadata and predictions endpoints."
                    }
                }
            }
            
            # Include predictions only if explicitly requested and count is reasonable
            if include_predictions and predictions_count <= 200:
                # Safe to include all predictions
                response_data["result"]["predictions"] = results.get("predictions", [])
            elif include_predictions:
                # Include a truncated set of predictions with a warning
                response_data["result"]["predictions"] = results.get("predictions", [])[:100]
                response_data["result"]["predictions_truncated"] = True
                response_data["result"]["warning"] = "Predictions truncated to 100 items. Use the dedicated endpoints for complete data."
            
            return JSONResponse(
                content=response_data,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Content-Type": "application/json"
                }
            )
        except Exception as e:
            logger.error(f"Error reading results file {result_file}: {str(e)}")
            return JSONResponse(
                content={
                    "processing_id": processing_id,
                    "status": "error",
                    "error": f"Error reading results file: {str(e)}"
                },
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Content-Type": "application/json"
                }
            )
    
    # Check if video file exists (processing still ongoing)
    video_file_mp4 = UPLOAD_DIR / f"{processing_id}.mp4"
    video_file_MP4 = UPLOAD_DIR / f"{processing_id}.MP4"
    
    if video_file_mp4.exists() or video_file_MP4.exists():
        logger.info(f"Video file exists, processing is ongoing for {processing_id}")
        return JSONResponse(
            content={
                "processing_id": processing_id,
                "status": "processing",
                "message": "Processing is still ongoing. Please try again later."
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
                "Content-Type": "application/json"
            }
        )
    
    # Prevent infinite polling by tracking how many times this ID has been checked
    if not hasattr(app.state, "poll_counters"):
        app.state.poll_counters = {}
    
    count = app.state.poll_counters.get(processing_id, 0) + 1
    app.state.poll_counters[processing_id] = count
    
    # After 3 attempts, return a terminal state to stop the polling
    if count >= 3:
        logger.warning(f"Processing ID {processing_id} not found after {count} attempts. Returning terminal state.")
        return JSONResponse(
            content={
                "processing_id": processing_id,
                "status": "failed",  # Terminal state that will stop polling
                "message": "Processing job not found. The job may not have been created or may have failed."
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
                "Content-Type": "application/json"
            }
        )
    
    # If no files were found and it's one of the first 3 attempts, return not_found
    logger.warning(f"No results or video file found for processing_id: {processing_id} (attempt {count})")
    return JSONResponse(
        content={
            "processing_id": processing_id,
            "status": "not_found",
            "message": "No processing job found with this ID. Please check the processing ID."
        },
        headers={
            "Access-Control-Allow-Origin": "*", 
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Content-Type": "application/json"
        }
    )

@app.get("/api/metrics")
def get_metrics():
    """
    Get metrics for all providers.
    """
    return JSONResponse(
        content={
            "aws": {"avg_objects": 14.5, "avg_time": 2.3, "cost_per_minute": 0.12},
            "azure": {"avg_objects": 12.2, "avg_time": 2.8, "cost_per_minute": 0.10},
            "yolo": {"avg_objects": 16.8, "avg_time": 1.9, "cost_per_minute": 0.05}
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )

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
    
    return JSONResponse(
        content=metrics[source],
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )

# Keep the original endpoints for backward compatibility
@app.post("/process/")
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    processing_id: Optional[str] = Form(None),
    expected_vehicles: int = Form(0),
    expected_people: int = Form(0)
):
    """
    Upload and process a video.
    
    The video will be processed using AWS Rekognition, Azure AI Vision, and local YOLO model.
    Processing includes:
    - Task A: Image Analysis with Object Detection
    - Task B: Video Processing with Object Tracking
    
    Parameters:
        - video: Uploaded video file
        - processing_id: Optional ID for the processing job
        - expected_vehicles: Expected number of vehicles in the video
        - expected_people: Expected number of people in the video
        
    Uses background tasks to avoid blocking the server.
    """
    # Use the exact processing_id from the frontend if provided, or generate a new one
    if processing_id:
        logger.info(f"Using processing ID from frontend: {processing_id}")
    else:
        # Generate a new ID in the same format the frontend expects (nanosecond timestamp)
        processing_id = f"proc_{int(time.time() * 1000000000)}"
        logger.info(f"Generated new processing ID: {processing_id}")
    
    logger.info(f"Expected vehicles: {expected_vehicles}, Expected people: {expected_people}")
    
    # Save uploaded file with processing ID in the name
    original_filename = video.filename
    file_extension = os.path.splitext(original_filename)[1]
    video_path = UPLOAD_DIR / f"{processing_id}{file_extension}"
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        logger.info(f"Video saved to {video_path}")
    except Exception as e:
        logger.error(f"Could not save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {str(e)}")
    
    # Start processing in background
    try:
        # Add task to background queue with the exact ID as provided
        background_tasks.add_task(
            _process_video_in_background, 
            str(video_path), 
            processing_id,
            expected_vehicles,
            expected_people
        )
        logger.info(f"Added processing task for {processing_id} to background queue")
        
        # Return immediate response with the exact processing ID
        return JSONResponse(
            status_code=202,
            content={
                "message": "Video uploaded and scheduled for processing",
                "video_path": str(video_path),
                "processing_id": processing_id,
                "status": "processing",
                "expected_vehicles": expected_vehicles,
                "expected_people": expected_people
            }
        )
    except Exception as e:
        logger.error(f"Error scheduling video processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error scheduling video processing: {str(e)}")

@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    """
    Get status of a processing job (legacy endpoint).
    Uses exact ID matching and prevents infinite polling.
    
    Args:
        job_id: ID of the processing job
    """
    # Try to find results file with exact match only
    result_file = Path("./data/results") / f"results_{job_id}.json"
    
    if result_file.exists():
        try:
            with open(result_file, "r") as f:
                results = json.load(f)
                
            # Optimize results size if needed
            if isinstance(results, dict) and "predictions" in results and isinstance(results["predictions"], list) and len(results["predictions"]) > 50:
                logger.info(f"Optimizing results size: truncating predictions from {len(results['predictions'])} to 50 items")
                results["predictions"] = results["predictions"][:50]
                results["predictions_truncated"] = True
                results["original_prediction_count"] = len(results["predictions"])
            
            # Reset counter on success
            if hasattr(app.state, "legacy_poll_counters"):
                if job_id in app.state.legacy_poll_counters:
                    del app.state.legacy_poll_counters[job_id]
                
            return JSONResponse(
                content={
                    "job_id": job_id, 
                    "status": "completed", 
                    "results": results
                },
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Content-Type": "application/json"
                }
            )
        except Exception as e:
            return JSONResponse(
                content={
                    "job_id": job_id,
                    "status": "error",
                    "error": f"Error reading results file: {str(e)}"
                },
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Content-Type": "application/json"
                }
            )
    else:
        # Check if video file exists
        video_file_mp4 = UPLOAD_DIR / f"{job_id}.mp4"
        video_file_MP4 = UPLOAD_DIR / f"{job_id}.MP4"
        
        if video_file_mp4.exists() or video_file_MP4.exists():
            logger.info(f"Video file exists, processing is ongoing for {job_id}")
            return JSONResponse(
                content={"job_id": job_id, "status": "processing"},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Content-Type": "application/json"
                }
            )
        else:
            # Prevent infinite polling
            if not hasattr(app.state, "legacy_poll_counters"):
                app.state.legacy_poll_counters = {}
            
            count = app.state.legacy_poll_counters.get(job_id, 0) + 1
            app.state.legacy_poll_counters[job_id] = count
            
            # After 3 attempts, return a terminal state
            if count >= 3:
                logger.warning(f"Legacy job ID {job_id} not found after {count} attempts. Returning terminal state.")
                return JSONResponse(
                    content={
                        "job_id": job_id, 
                        "status": "failed",
                        "message": "Processing job not found after multiple attempts."
                    },
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type",
                        "Content-Type": "application/json"
                    }
                )
            
            # For the first few attempts, return not_found
            logger.warning(f"No results or video file found for job_id: {job_id} (attempt {count})")
            return JSONResponse(
                content={"job_id": job_id, "status": "not_found"},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Content-Type": "application/json"
                }
            )

# Add a separate endpoint for metadata only to avoid timeouts
@app.get("/api/videos/{processing_id}/metadata")
def get_video_metadata(processing_id: str):
    """
    Get only metadata for a completed video processing job.
    This is a lightweight endpoint to avoid timeouts.
    """
    logger.info(f"Fetching metadata for processing_id: {processing_id}")
    
    # Look for results file
    result_file = Path("./data/results") / f"results_{processing_id}.json"
    
    if result_file.exists():
        try:
            with open(result_file, "r") as f:
                results = json.load(f)
            
            # Count predictions
            predictions_count = len(results.get("predictions", [])) if isinstance(results, dict) and "predictions" in results else 0
            
            # Calculate recommended chunk size based on prediction count
            recommended_chunk_size = 100
            if predictions_count > 1000:
                recommended_chunk_size = 200
            elif predictions_count > 500:
                recommended_chunk_size = 100
            elif predictions_count > 100:
                recommended_chunk_size = 50
            else:
                recommended_chunk_size = predictions_count  # All at once for small results
            
            # Create a lightweight response with just metadata
            metadata = {
                "processing_id": processing_id,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "predictions_count": predictions_count,
                "has_predictions": predictions_count > 0,
                "recommended_chunk_size": recommended_chunk_size,
                "chunks_required": (predictions_count + recommended_chunk_size - 1) // recommended_chunk_size if recommended_chunk_size > 0 else 0
            }
            
            return JSONResponse(
                content=metadata,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Content-Type": "application/json"
                }
            )
        except Exception as e:
            logger.error(f"Error reading metadata from {result_file}: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "processing_id": processing_id,
                    "status": "error",
                    "error": f"Error reading results metadata: {str(e)}"
                },
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Content-Type": "application/json"
                }
            )
    
    # Check if video file exists (processing still ongoing)
    video_file_mp4 = UPLOAD_DIR / f"{processing_id}.mp4"
    video_file_MP4 = UPLOAD_DIR / f"{processing_id}.MP4"
    
    if video_file_mp4.exists() or video_file_MP4.exists():
        return JSONResponse(
            content={
                "processing_id": processing_id,
                "status": "processing",
                "message": "Processing is still ongoing."
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
                "Content-Type": "application/json"
            }
        )
    
    # Not found response (simpler for metadata endpoint)
    return JSONResponse(
        status_code=404,
        content={
            "processing_id": processing_id,
            "status": "not_found",
            "message": "No processing job found with this ID."
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Content-Type": "application/json"
        }
    )

# Add endpoint for full predictions (can be called after metadata confirms completion)
@app.get("/api/videos/{processing_id}/predictions")
def get_video_predictions(processing_id: str, chunk_size: int = 100, offset: int = 0):
    """
    Get only the predictions array for a completed video processing job.
    This should be called after metadata confirms the job is complete.
    
    Parameters:
        - processing_id: ID of the processing job
        - chunk_size: Number of predictions to return in one response (default: 100)
        - offset: Starting index for pagination (default: 0)
    """
    logger.info(f"Fetching predictions for processing_id: {processing_id} (offset: {offset}, chunk_size: {chunk_size})")
    
    # Look for results file
    result_file = Path("./data/results") / f"results_{processing_id}.json"
    
    if result_file.exists():
        try:
            with open(result_file, "r") as f:
                results = json.load(f)
            
            # Extract just the predictions
            all_predictions = results.get("predictions", []) if isinstance(results, dict) else []
            total_count = len(all_predictions)
            
            # Get the requested chunk of predictions
            end_idx = min(offset + chunk_size, total_count)
            predictions_chunk = all_predictions[offset:end_idx]
            
            logger.info(f"Returning predictions chunk {offset}-{end_idx} of {total_count} total")
            
            return JSONResponse(
                content={
                    "processing_id": processing_id,
                    "predictions": predictions_chunk,
                    "count": len(predictions_chunk),
                    "total_count": total_count,
                    "offset": offset,
                    "has_more": end_idx < total_count
                },
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Content-Type": "application/json"
                }
            )
        except Exception as e:
            logger.error(f"Error reading predictions from {result_file}: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "processing_id": processing_id,
                    "status": "error",
                    "error": f"Error reading predictions: {str(e)}"
                }
            )
    
    # Not found response
    return JSONResponse(
        status_code=404,
        content={
            "processing_id": processing_id,
            "status": "not_found",
            "message": "No predictions found for this ID."
        }
    )

def start():
    """Start the API server."""
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        reload=False,  # Set to False in production environment
        timeout_keep_alive=300,  # Increase keep-alive timeout to 5 minutes
        workers=1,  # Use a single worker process
        limit_concurrency=20,  # Limit concurrent connections
        timeout_graceful_shutdown=10,  # Grace period for shutdown
        http="httptools"  # Use httptools for better performance
    )

if __name__ == "__main__":
    start() 