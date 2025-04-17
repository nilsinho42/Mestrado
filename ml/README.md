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
   pip install -r ../requirements.txt
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
   AWS_FARGATE_ENDPOINT=https://your-fargate-endpoint.region.amazonaws.com
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

### Cloud Tracker Services

This pipeline can use containerized object tracking services in the cloud for better scalability and performance:

#### Azure Container App for DeepSORT tracking

The pipeline can use an Azure Container App for DeepSORT tracking. Set the endpoint in your `.env` file:
```
AZURE_DEEPSORT_ENDPOINT=https://your-deepsort-app.region.azurecontainerapps.io
```

#### AWS Fargate for DeepSORT tracking

The pipeline can use AWS Fargate for DeepSORT tracking. Set the endpoint in your `.env` file:
```
AWS_FARGATE_ENDPOINT=https://your-fargate-endpoint.region.amazonaws.com
```

### Deploying the Cloud Tracker Services

For improved performance, the DeepSORT tracking can be offloaded to cloud container services:

#### Azure Container App Deployment

1. Build the Docker image:
   ```bash
   cd cloud/azure
   docker build -t your-dockerhub-username/deepsort-tracker:latest .
   docker push your-dockerhub-username/deepsort-tracker:latest
   ```

2. Create an Azure Container App using the Azure Portal or Azure CLI:
   ```bash
   az containerapp create \
     --name deepsort-tracker \
     --resource-group your-resource-group \
     --environment your-environment \
     --image your-dockerhub-username/deepsort-tracker:latest \
     --target-port 8080 \
     --ingress external \
     --cpu 0.5 \
     --memory 1.0Gi
   ```

3. Update `.env` with the Container App URL

#### AWS Fargate Deployment

1. Build the Docker image:
   ```bash
   cd cloud/aws
   docker build -t your-dockerhub-username/deepsort-tracker-aws:latest .
   docker push your-dockerhub-username/deepsort-tracker-aws:latest
   ```

2. Create an AWS ECR repository (optional):
   ```bash
   aws ecr create-repository --repository-name deepsort-tracker
   ```

3. Create an ECS task definition and service using Fargate:
   ```bash
   # Create task definition
   aws ecs register-task-definition \
     --family deepsort-tracker \
     --requires-compatibilities FARGATE \
     --network-mode awsvpc \
     --cpu 0.5 \
     --memory 1GB \
     --execution-role-arn arn:aws:iam::your-account:role/ecsTaskExecutionRole \
     --container-definitions '[{
         "name": "deepsort-tracker",
         "image": "your-dockerhub-username/deepsort-tracker-aws:latest",
         "essential": true,
         "portMappings": [{
           "containerPort": 8080,
           "hostPort": 8080,
           "protocol": "tcp"
         }]
       }]'
   
   # Create service
   aws ecs create-service \
     --cluster your-cluster \
     --service-name deepsort-tracker \
     --task-definition deepsort-tracker \
     --desired-count 1 \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[subnet-id],securityGroups=[sg-id],assignPublicIp=ENABLED}"
   ```

4. Set up an Application Load Balancer to route traffic to your Fargate service

5. Update `.env` with the ALB endpoint URL

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

## Recent Updates

### Architecture Refactoring (June 2023)

We've made several significant improvements to the architecture:

1. **Unified Processing Pipeline**: 
   - Task A (detection) and Task B (tracking) now share a single pipeline to avoid processing the same frame twice
   - The DeepSORT implementation has been updated to use YOLOv11n detection

2. **Local Processing Only**:
   - Removed dependency on S3 and Azure Blob Storage
   - All video frames are now stored and processed locally
   - Improved performance by reducing network overhead

3. **Improved Models**:
   - Now using YOLOv11n for object tracking (Task B)
   - Using DeepSORT with proper implementation (not placeholder)
   
4. **Bug Fixes**:
   - Fixed issue with results files not being saved properly
   - Results now correctly written to `data/results/` directory
   - API now waits for processing to complete before returning response

### Setup and Usage

1. Download required models:
```bash
python download_models.py
```

2. Run the API server:
```bash
python api.py
```

3. Process a video:
```bash
python main.py --video /path/to/your/video.mp4
```

### API Changes

The API now waits for video processing to complete before returning a response. This ensures that clients receive results immediately without having to poll the status endpoint.

Example API response:
```json
{
  "message": "Video processing completed",
  "video_path": "uploads/video.mp4",
  "status": "completed",
  "results": {
    "job_id": "...",
    "video_path": "...",
    "task_a": { ... },
    "task_b": { ... },
    "timestamp": "..."
  }
}
```

### Configuration

- Model selection is now handled automatically
- YOLOv11n is used for tracking (Task B)
- YOLOv8n is used for detection (Task A) by default 

# Model Requirements

## YOLOv11n

This application exclusively uses YOLOv11n for object detection and tracking. YOLOv8n is not used at all.

### Important Notes:
- YOLOv11n.pt must be available in the application root directory
- No fallback models are used; the application will fail if YOLOv11n is not found
- The model will be downloaded automatically during installation

### Manual Download Instructions

If automatic download fails, you must manually download YOLOv11n.pt and place it in the application root directory:

1. Download YOLOv11n from: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
2. **Important**: Rename the downloaded file from `yolo11n.pt` to `yolov11n.pt` (add the 'v')
3. Place the renamed file in the application root directory

### Troubleshooting

If you encounter this error:
```
AttributeError: Can't get attribute 'C3k2' on <module 'ultralytics.nn.modules.block' ...
```

The issue is either:
1. The ultralytics version is not compatible with YOLOv11n. Make sure you have ultralytics 8.3.2 or newer installed.
2. The model file is not correctly named. It should be named `yolov11n.pt` (with 'v').

To fix:
```bash
pip install ultralytics==8.3.2
python download_models.py
```

## Running the Application

Before starting the application, ensure YOLOv11n.pt is available. The application will check for its presence on startup and exit with an error if it's not found.

To download the model manually:
```bash
python download_models.py
```

If you see an error message about YOLOv11n.pt being required, follow the manual download instructions above. 