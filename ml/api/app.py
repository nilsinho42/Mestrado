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

@app.post("/api/comparison/start")
async def start_comparison(video: UploadFile = File(...)):
    try:
        # Verify cloud credentials
        if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
            raise HTTPException(
                status_code=500,
                detail="AWS credentials not configured. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
            )
        
        if not os.getenv('AZURE_ENDPOINT') or not os.getenv('AZURE_KEY'):
            raise HTTPException(
                status_code=500,
                detail="Azure credentials not configured. Please set AZURE_ENDPOINT and AZURE_KEY environment variables."
            )

        # Create a temporary directory to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as temp_video:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(video.file, temp_video)
            temp_video_path = temp_video.name

        try:
            # Initialize the comparator
            comparator = CloudServiceComparator()
            
            # Process with all services using the temporary file path
            results = comparator.compare_services(temp_video_path)
            
            # Format results for frontend
            formatted_results = {
                "processing_time": {
                    "yolo": results["latency"]["yolo"],
                    "aws": results["latency"]["aws"],
                    "azure": results["latency"]["azure"]
                },
                "detections": {
                    "yolo": results["detections"]["yolo"],
                    "aws": results["detections"]["aws"],
                    "azure": results["detections"]["azure"]
                },
                "costs": {
                    "yolo": 0.0,  # Local processing cost
                    "aws": 0.0001,  # Example cost from AWS
                    "azure": 0.0001  # Example cost from Azure
                },
                "dashboard_url": "http://localhost:8501",  # Your Streamlit dashboard URL
                "saved_frames": results.get("saved_frames", {})  # Add saved frame paths
            }

            # Save results for the Streamlit dashboard
            results_dir = Path(__file__).parent.parent / "cloud_comparison" / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = results_dir / f"comparison_results_{timestamp}.json"
            
            # Log the results before saving
            logger.debug(f"Saving results to {result_file}:")
            logger.debug(json.dumps(formatted_results, indent=2))
            
            with open(result_file, "w") as f:
                json.dump(formatted_results, f, indent=2)
                
            # Verify the saved file
            with open(result_file) as f:
                saved_data = json.load(f)
                logger.debug("Verified saved data:")
                logger.debug(json.dumps(saved_data, indent=2))

            return formatted_results
            
        finally:
            # Clean up: remove the temporary file
            try:
                os.unlink(temp_video_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_video_path}: {e}")
            
    except Exception as e:
        logger.error(f"Error in comparison: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 