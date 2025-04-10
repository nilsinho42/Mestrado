# Video Processing Pipeline

This project implements a video processing pipeline that analyzes videos using multiple object detection and tracking services:
- AWS Rekognition
- Azure AI Vision
- Local YOLO

## Features

### Task A — Image Analysis with Object Detection
- Samples 1 out of every 5 frames from input videos
- Stores images in `./data/object_detection/images`
- Runs object detection using AWS, Azure, and local YOLO models
- For each service, calculates and stores:
  - Latency per image
  - Number of people detected
  - Number of vehicles detected

### Task B — Video Processing with Object Tracking
- Stores videos in AWS S3 and Azure Blob Storage (if configured)
- Runs object tracking using:
  - AWS: Rekognition + DeepSORT
  - Azure: AI Vision + DeepSORT
  - Local: YOLOv8 + DeepSORT
- For each service, measures:
  - Processing time from upload to result
  - Count of people and vehicles tracked
  - Cost estimation based on cloud provider pricing

## Setup

### Prerequisites
- Python 3.8+
- PostgreSQL database
- AWS account (for AWS services)
- Azure account (for Azure services)

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ml
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure environment variables in `.env` file:
   ```
   # Database Configuration
   DB_HOST=localhost
   DB_PORT=5432
   DB_USER=postgres
   DB_PASSWORD=postgres
   DB_NAME=ml_comparison

   # AWS Configuration
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_DEFAULT_REGION=us-east-1
   AWS_BUCKET_NAME=your-bucket-name

   # Azure Configuration
   AZURE_ENDPOINT=https://your-vision-service.cognitiveservices.azure.com/
   AZURE_KEY=your_azure_key
   AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string
   AZURE_STORAGE_ACCOUNT=your_storage_account
   AZURE_CONTAINER_NAME=your-container-name
   AZURE_DEEPSORT_ENDPOINT=https://your-deepsort-app.region.azurecontainerapps.io
   ```

4. Update cost parameters in `cost_config.ini` file with actual cloud provider pricing.

5. Create necessary directories:
   ```
   mkdir -p data/object_detection/images
   mkdir -p uploads
   ```

### Database Setup

The application automatically creates the necessary tables when started, but you need to create the database:

```sql
CREATE DATABASE ml_comparison;
```

### (Optional) Azure Container App for DeepSORT tracking

The pipeline can use an Azure Container App for DeepSORT tracking. To deploy it:

```bash
cd cloud/azure
docker build -t your-dockerhub-username/deepsort-tracker:latest .
docker push your-dockerhub-username/deepsort-tracker:latest
```

Then create an Azure Container App using the Docker Hub image.

## Usage

### Command Line Interface

Process a video directly from the command line:

```bash
python video_pipeline.py --video /path/to/your/video.mp4 --output ./data/object_detection/images
```

### Web API

Start the web API:

```bash
python run_api.py
```

The API will be available at http://localhost:8000.

### API Endpoints

- `POST /api/videos/upload` - Upload a video for processing
- `GET /api/jobs/{job_id}` - Get job status
- `GET /api/results/{job_id}` - Get job results
- `GET /api/metrics` - Get all metrics
- `GET /api/metrics/{source}` - Get metrics for a specific source
- `GET /api/detection/{image_id}` - Get detection results for a specific image
- `GET /api/tracking/{video_id}` - Get tracking results for a specific video

## Architecture

The project is built with a modular design to easily support new providers:

- `video_processor.py` - Base video processing functionality
- `detectors.py` - Object detection implementations
- `trackers.py` - Object tracking implementations
- `cloud_storage.py` - Cloud storage utilities
- `cost_utils.py` - Cost calculation utilities
- `db_utils.py` - Database utilities
- `video_pipeline.py` - Main pipeline implementation
- `api/app.py` - FastAPI web application

## Data Flow

1. Video is uploaded via API or specified via command line
2. Task A extracts frames and runs object detection on each provider
3. Task B uploads video to cloud storage and runs object tracking
4. Results are stored in PostgreSQL database and as JSON files
5. Metrics are collected and stored for performance comparison

## License

This project is licensed under the MIT License - see the LICENSE file for details. 