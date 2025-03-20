-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create locations table
CREATE TABLE IF NOT EXISTS locations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    person_count INTEGER DEFAULT 0,
    vehicle_count INTEGER DEFAULT 0,
    avg_flow DOUBLE PRECISION DEFAULT 0.0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create images table
CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    source VARCHAR(255) NOT NULL,
    cloud_platform VARCHAR(50) NOT NULL,
    processing_time DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create detections table
CREATE TABLE IF NOT EXISTS detections (
    id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(id),
    class VARCHAR(100) NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    bounding_box JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create models table
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    type VARCHAR(50) NOT NULL,
    framework VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    cloud_platform VARCHAR(50) NOT NULL,
    endpoint_url VARCHAR(255) NOT NULL,
    accuracy DOUBLE PRECISION NOT NULL,
    avg_inference_time DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deployed_at TIMESTAMP WITH TIME ZONE,
    created_by INTEGER NOT NULL REFERENCES users(id)
);

-- Create model_metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES models(id),
    date DATE NOT NULL,
    inference_count BIGINT NOT NULL DEFAULT 0,
    avg_inference_time DOUBLE PRECISION NOT NULL,
    avg_confidence DOUBLE PRECISION NOT NULL,
    error_count BIGINT NOT NULL DEFAULT 0,
    UNIQUE(model_id, date)
);

-- Create cloud_metrics table
CREATE TABLE IF NOT EXISTS cloud_metrics (
    id SERIAL PRIMARY KEY,
    platform VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    request_count BIGINT NOT NULL DEFAULT 0,
    avg_latency DOUBLE PRECISION NOT NULL,
    cost DOUBLE PRECISION NOT NULL DEFAULT 0,
    UNIQUE(platform, date)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_detections_image_id ON detections(image_id);
CREATE INDEX IF NOT EXISTS idx_model_metrics_model_id ON model_metrics(model_id);
CREATE INDEX IF NOT EXISTS idx_cloud_metrics_platform ON cloud_metrics(platform);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Drop and recreate trigger for users
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Drop and recreate trigger for locations
DROP TRIGGER IF EXISTS update_locations_updated_at ON locations;
CREATE TRIGGER update_locations_updated_at
BEFORE UPDATE ON locations
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Drop and recreate trigger for models
DROP TRIGGER IF EXISTS update_models_updated_at ON models;
CREATE TRIGGER update_models_updated_at
BEFORE UPDATE ON models
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();
