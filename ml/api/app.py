from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from pathlib import Path
import sys
import os
from datetime import datetime
import logging
import boto3
from dotenv import load_dotenv
import tempfile
import shutil

# Load environment variables
load_dotenv()

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cloud_comparison.compare_services import CloudServiceComparator

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure AWS
os.environ['AWS_DEFAULT_REGION'] = os.getenv('AWS_DEFAULT_REGION', 'us-east-2')  # Use region from .env
if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
    logger.warning("AWS credentials not found in environment variables")

# Verify Azure credentials
if not os.getenv('AZURE_ENDPOINT') or not os.getenv('AZURE_KEY'):
    logger.warning("Azure credentials not found in environment variables")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ComparisonRequest(BaseModel):
    videoPath: str

@app.post("/start_comparison")
async def start_comparison(video: UploadFile = File(...)):
    """Start video comparison between services."""
    try:
        # Create temporary file to store uploaded video
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, video.filename)
        
        try:
            # Save uploaded file
            with open(temp_video_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
            
            # Initialize comparator and process video
            comparator = CloudServiceComparator()
            results = comparator.compare_services(temp_video_path)
            
            # Format results for response
            formatted_results = {
                "video_info": results["video_info"],
                "processing_time": results["processing_time"],
                "total_detections": results["total_detections"],
                "frames": results["frames"],
                "costs": results["costs"],
                "dashboard_url": results["dashboard_url"],
                "saved_frames": results["saved_frames"],
                "detections": results["detections"]
            }
            
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = Path("cloud_comparison/results") / f"comparison_results_{timestamp}.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_path, "w") as f:
                json.dump(formatted_results, f, indent=2)
            
            return formatted_results
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 