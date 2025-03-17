# Object Detection API

A RESTful API for object detection using machine learning models deployed across different cloud platforms.

## Features

- User authentication and authorization
- Location management
- Image analysis with object detection
- Model management and deployment
- Cloud platform comparison
- Performance metrics and monitoring
- Rate limiting and security measures

## Tech Stack

- Go 1.21+
- PostgreSQL 14+
- Gin web framework
- JWT for authentication
- Zap for logging

## Project Structure

```
backend/
├── controllers/     # Request handlers
├── db/             # Database models and migrations
├── middleware/     # Middleware components
├── migrations/     # SQL migration files
├── router/         # Route definitions
└── main.go         # Application entry point
```

## Setup

1. Install dependencies:
   ```bash
   go mod download
   ```

2. Set up environment variables:
   ```bash
   export DATABASE_URL="postgres://user:password@localhost:5432/dbname?sslmode=disable"
   export JWT_SECRET="your-secret-key"
   export AWS_MODEL_ENDPOINT="your-aws-endpoint"
   export AZURE_MODEL_ENDPOINT="your-azure-endpoint"
   export GCP_MODEL_ENDPOINT="your-gcp-endpoint"
   ```

3. Run migrations:
   ```bash
   go run main.go
   ```

4. Start the server:
   ```bash
   go run main.go
   ```

## API Endpoints

### Authentication
- `POST /api/login` - User login
- `POST /api/register` - User registration

### Locations
- `GET /api/locations` - List all locations
- `POST /api/locations` - Create a new location
- `GET /api/locations/:id` - Get a specific location
- `PUT /api/locations/:id` - Update a location
- `DELETE /api/locations/:id` - Delete a location

### Detections
- `POST /api/detections/analyze` - Analyze an image
- `GET /api/detections/stats` - Get detection statistics

### Models
- `GET /api/models` - List all models
- `POST /api/models` - Register a new model
- `GET /api/models/compare` - Compare model performance
- `GET /api/models/:id/metrics` - Get model metrics
- `PUT /api/models/:id` - Update a model
- `POST /api/models/:id/deploy` - Deploy a model

### Cloud
- `GET /api/cloud/costs` - Get cloud costs
- `GET /api/cloud/performance` - Get cloud performance metrics

## Rate Limiting

- Global rate limit: 100 requests per second
- Detection endpoints: 10 requests per second

## Security

- JWT-based authentication
- Password hashing with bcrypt
- CORS enabled
- Rate limiting
- Input validation

## Database Schema

The application uses PostgreSQL with the following main tables:
- users
- locations
- images
- detections
- models
- model_metrics
- cloud_metrics

See `migrations/001_initial_schema.sql` for the complete schema.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 