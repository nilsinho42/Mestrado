# Technical Context: Cloud-Based Object Detection and Flow Analysis
*Version: 1.0*
*Created: 2023-11-15*
*Last Updated: 2023-11-15*

## Technology Stack
- Frontend: React.js with TypeScript (Vite build tool)
- Backend: Go (Golang) RESTful API
- ML Services: Python with YOLO models
- Database: PostgreSQL
- Infrastructure: Docker and Docker Compose for containerization
- Cloud Services: AWS and Azure object detection services

## Development Environment Setup
1. Clone the repository
2. Install prerequisites:
   - Python 3.8+
   - Docker and Docker Compose
   - Node.js 16+ and npm
   - Go 1.18+ (for backend)
3. Install Python dependencies with `pip install -r requirements.txt`
4. Set up environment variables by copying `.env.example` to `.env` and configuring
5. Run the system with `docker-compose up -d`

## Dependencies
- ML Service: YOLO (YOLOv8 and YOLOv11), TensorFlow/PyTorch, OpenCV
- Backend: Go modules as defined in go.mod/go.sum
- Frontend: React, TypeScript, Vite as defined in package.json
- Infrastructure: Docker, Docker Compose
- Cloud: AWS SDK, Azure SDK

## Technical Constraints
- Must be containerized for reproducible deployment
- ML models must be optimized for both cloud and local processing
- API must handle both synchronous and asynchronous processing requests
- Frontend must visualize comparison metrics effectively
- System must log performance and cost metrics for analysis

## Build and Deployment
- Build Process: Docker images built for each service
- Deployment Procedure: Docker Compose for local development, potential cloud deployment options
- CI/CD: Not fully implemented yet, but directory structure suggests preparation

## Testing Approach
- Unit Testing: Individual components are tested separately
- Integration Testing: API endpoints and services communication
- E2E Testing: Full system testing with sample data
- Performance Testing: Metrics collection during object detection operations

---

*This document describes the technologies used in the project and how they're configured.* 