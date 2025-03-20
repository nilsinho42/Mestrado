-- Drop triggers first
DROP TRIGGER IF EXISTS update_processed_videos_updated_at ON processed_videos;

-- Drop indexes
DROP INDEX IF EXISTS idx_processed_videos_user_id;
DROP INDEX IF EXISTS idx_processed_videos_status;
DROP INDEX IF EXISTS idx_detection_results_video_id;
DROP INDEX IF EXISTS idx_service_comparisons_video_id;
DROP INDEX IF EXISTS idx_performance_metrics_service_name;
DROP INDEX IF EXISTS idx_performance_metrics_timestamp;

-- Drop tables in reverse order of creation
DROP TABLE IF EXISTS performance_metrics;
DROP TABLE IF EXISTS service_comparisons;
DROP TABLE IF EXISTS detection_results;
DROP TABLE IF EXISTS processed_videos;

-- Drop the view last
DROP VIEW IF EXISTS ml_results_summary; 