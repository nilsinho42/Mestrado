-- Drop triggers
DROP TRIGGER IF EXISTS update_processed_videos_updated_at ON processed_videos;

-- Drop function
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Drop indexes
DROP INDEX IF EXISTS idx_processed_videos_user_id;
DROP INDEX IF EXISTS idx_processed_videos_status;
DROP INDEX IF EXISTS idx_detection_results_video_id;
DROP INDEX IF EXISTS idx_service_comparisons_video_id;
DROP INDEX IF EXISTS idx_performance_metrics_service_name;
DROP INDEX IF EXISTS idx_performance_metrics_timestamp;

-- Drop tables
DROP TABLE IF EXISTS performance_metrics;
DROP TABLE IF EXISTS service_comparisons;
DROP TABLE IF EXISTS detection_results;
DROP TABLE IF EXISTS processed_videos; 