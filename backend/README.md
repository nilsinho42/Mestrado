# Video Processing API

A lightweight Go API that serves as a proxy between the frontend and the Python ML service for video processing.

## Overview

This API provides endpoints for:
- Uploading videos for processing
- Checking the status of video processing
- Retrieving metrics from the ML service

## Architecture

The backend service is designed with a simple architecture:
- **controllers**: Handle HTTP requests and responses
- **services**: Business logic for communicating with the ML service
- **db**: Database models and migrations
- **router**: Route definitions

## API Endpoints

### Video Processing
- `POST /api/videos/upload` - Upload a video for processing
- `GET /api/videos/:id/status` - Get processing status for a video

### Metrics
- `GET /api/metrics` - Get metrics for all providers
- `GET /api/metrics/:source` - Get metrics for a specific provider (aws, azure, local)

### Health Check
- `GET /health` - Check if the service is running

## Setup

1. Set environment variables:
```bash
export DB_HOST=localhost
export DB_USER=postgres
export DB_PASSWORD=postgres
export DB_NAME=ml_comparison
export ML_SERVICE_URL=http://localhost:8000
```

2. Install dependencies:
```bash
go mod download
```

3. Run the server:
```bash
go run main.go
```

## Docker

To build and run the service using Docker:

```bash
docker build -t video-processing-api .
docker run -p 8080:8080 --env-file .env video-processing-api
```

## Communication with ML Service

This backend communicates with the Python ML service to:
1. Upload videos for processing
2. Check processing status
3. Retrieve metrics and results

The ML service performs the actual video processing using AWS, Azure, and local implementations. 