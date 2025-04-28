# Cloud DeepSORT Tracker

This directory contains a unified, cloud-agnostic implementation of the DeepSORT tracker that can be deployed to any cloud provider (AWS, Azure, GCP) or other container platforms.

## Overview

The tracker provides a REST API for object tracking with the following endpoints:

- `GET /` - Basic info about the API
- `POST /api/track` - Main tracking endpoint
- `POST /api/reset/{video_id}` - Reset a tracking session
- `GET /api/status/{video_id}` - Get statistics from a tracking session
- `GET /healthcheck` - Simple health check endpoint

## Files

- `tracker_app.py` - The main FastAPI application with DeepSORT implementation
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container definition for deployment

## Cloud Provider Specific Files

For cloud provider specific credentials, place them in the `credentials` directory:

- GCP: `credentials/gcp-service-account.json`
- AWS: `credentials/aws-credentials.json`
- Azure: `credentials/azure-credentials.json`

## Deployment

### Prerequisites

- Docker installed
- Cloud provider CLI configured (e.g., AWS CLI, Azure CLI, or Google Cloud SDK)

### Building the Docker Image

```bash
cd cloud
docker build -t deepsort-tracker .
```

### Running Locally

```bash
docker run -p 8080:8080 deepsort-tracker
```

### Deploying to Cloud Providers

#### AWS Fargate

```bash
aws ecr create-repository --repository-name deepsort-tracker
aws ecr get-login-password | docker login --username AWS --password-stdin <aws-account-id>.dkr.ecr.<region>.amazonaws.com
docker tag deepsort-tracker <aws-account-id>.dkr.ecr.<region>.amazonaws.com/deepsort-tracker
docker push <aws-account-id>.dkr.ecr.<region>.amazonaws.com/deepsort-tracker
```

Then deploy using AWS Fargate from the ECR repository.

#### Google Cloud Run

```bash
docker tag deepsort-tracker gcr.io/<project-id>/deepsort-tracker
docker push gcr.io/<project-id>/deepsort-tracker
gcloud run deploy --image gcr.io/<project-id>/deepsort-tracker --platform managed
```

#### Azure Container Instances

```bash
az acr create --resource-group myResourceGroup --name myContainerRegistry --sku Basic
az acr login --name myContainerRegistry
docker tag deepsort-tracker myContainerRegistry.azurecr.io/deepsort-tracker
docker push myContainerRegistry.azurecr.io/deepsort-tracker
az container create --resource-group myResourceGroup --name deepsort-tracker --image myContainerRegistry.azurecr.io/deepsort-tracker --dns-name-label deepsort-tracker --ports 8080
```

## Environment Variables

The application supports the following environment variables:

- `PORT` - Port to listen on (default: 8080)
- `LOG_LEVEL` - Logging level (default: info)

## Performance Considerations

The DeepSORT tracker is designed to run efficiently in cloud environments. For better performance:

- Increase the container's resources (CPU/memory) for handling multiple parallel tracking sessions
- Adjust the `max_age` and `n_init` parameters based on your specific use case
- Consider using cloud provider-specific optimizations for the container hosting 