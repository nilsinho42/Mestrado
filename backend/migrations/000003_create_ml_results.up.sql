-- Create processed_videos table
CREATE TABLE IF NOT EXISTS processed_videos (
    id SERIAL PRIMARY KEY,
    video_path TEXT NOT NULL,
    processing_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER REFERENCES users(id),
    error_message TEXT
);

-- Create detection_results table
CREATE TABLE IF NOT EXISTS detection_results (
    id SERIAL PRIMARY KEY,
    video_id INTEGER REFERENCES processed_videos(id),
    frame_number INTEGER NOT NULL,
    detection_type TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    bbox JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create service_comparisons table
CREATE TABLE IF NOT EXISTS service_comparisons (
    id SERIAL PRIMARY KEY,
    video_id INTEGER REFERENCES processed_videos(id),
    service_name TEXT NOT NULL,
    processing_time FLOAT NOT NULL,
    memory_usage FLOAT NOT NULL,
    accuracy FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create performance_metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    service_name TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    value FLOAT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create indexes
CREATE INDEX idx_processed_videos_user_id ON processed_videos(user_id);
CREATE INDEX idx_processed_videos_status ON processed_videos(status);
CREATE INDEX idx_detection_results_video_id ON detection_results(video_id);
CREATE INDEX idx_service_comparisons_video_id ON service_comparisons(video_id);
CREATE INDEX idx_performance_metrics_service_name ON performance_metrics(service_name);
CREATE INDEX idx_performance_metrics_timestamp ON performance_metrics(timestamp);

-- Add updated_at trigger function if not exists
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add trigger to processed_videos
CREATE TRIGGER update_processed_videos_updated_at
    BEFORE UPDATE ON processed_videos
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column(); 