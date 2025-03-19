from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from cloud_comparison.compare import compare_services
from cloud_comparison.streamlit_app import get_streamlit_url

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ComparisonRequest(BaseModel):
    videoPath: str

class ComparisonResponse(BaseModel):
    processingTime: dict
    detections: dict
    costs: dict
    dashboardUrl: str

@app.post("/api/comparison/start")
async def start_comparison(request: ComparisonRequest) -> ComparisonResponse:
    try:
        # Ensure the video file exists
        if not os.path.exists(request.videoPath):
            raise HTTPException(status_code=404, detail="Video file not found")

        # Run the comparison
        results = compare_services(request.videoPath)
        
        # Get the Streamlit dashboard URL
        dashboard_url = get_streamlit_url()

        return ComparisonResponse(
            processingTime={
                "yolo": results["processing_time"]["yolo"],
                "aws": results["processing_time"]["aws"],
                "azure": results["processing_time"]["azure"]
            },
            detections={
                "yolo": results["detections"]["yolo"],
                "aws": results["detections"]["aws"],
                "azure": results["detections"]["azure"]
            },
            costs={
                "yolo": results["costs"]["yolo"],
                "aws": results["costs"]["aws"],
                "azure": results["costs"]["azure"]
            },
            dashboardUrl=dashboard_url
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 