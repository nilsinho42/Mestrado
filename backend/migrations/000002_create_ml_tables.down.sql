-- Drop the view first
DROP VIEW IF EXISTS ml_results_summary;

-- Drop triggers
DROP TRIGGER IF EXISTS update_processed_videos_updated_at ON processed_videos;
DROP TRIGGER IF EXISTS update_ml_experiments_updated_at ON ml_experiments;

-- Drop the function
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Drop indexes
DROP INDEX IF EXISTS idx_processed_videos_user_id;
DROP INDEX IF EXISTS idx_processed_videos_created_at;
DROP INDEX IF EXISTS idx_detection_results_video_id;
DROP INDEX IF EXISTS idx_detection_results_service_name;
DROP INDEX IF EXISTS idx_detection_results_created_at;
DROP INDEX IF EXISTS idx_object_counts_detection_id;
DROP INDEX IF EXISTS idx_object_counts_object_type;
DROP INDEX IF EXISTS idx_service_costs_detection_id;
DROP INDEX IF EXISTS idx_service_costs_service_name;
DROP INDEX IF EXISTS idx_ml_experiments_status;
DROP INDEX IF EXISTS idx_ml_experiments_created_at;

-- Drop tables in reverse order of creation
DROP TABLE IF EXISTS service_costs;
DROP TABLE IF EXISTS object_counts;
DROP TABLE IF EXISTS detection_results;
DROP TABLE IF EXISTS ml_experiments;
DROP TABLE IF EXISTS processed_videos; 