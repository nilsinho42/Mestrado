package operations

import (
	"database/sql"
	"time"

	"github.com/nilsinho42/Mestrado/db/models"
)

// MLOperations handles database operations for ML-related data
type MLOperations struct {
	db *sql.DB
}

// NewMLOperations creates a new ML operations instance
func NewMLOperations(db *sql.DB) *MLOperations {
	return &MLOperations{db: db}
}

// CreateProcessedVideo creates a new processed video record
func (o *MLOperations) CreateProcessedVideo(video *models.ProcessedVideo) error {
	query := `
		INSERT INTO processed_videos (video_path, processing_id, status, user_id, error_message)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id, created_at, updated_at`

	return o.db.QueryRow(
		query,
		video.VideoPath,
		video.ProcessingID,
		video.Status,
		video.UserID,
		video.ErrorMessage,
	).Scan(&video.ID, &video.CreatedAt, &video.UpdatedAt)
}

// UpdateProcessedVideoStatus updates the status of a processed video
func (o *MLOperations) UpdateProcessedVideoStatus(processingID string, status string, errorMessage string) error {
	query := `
		UPDATE processed_videos
		SET status = $1, error_message = $2, updated_at = $3
		WHERE processing_id = $4`

	result, err := o.db.Exec(query, status, errorMessage, time.Now(), processingID)
	if err != nil {
		return err
	}

	rows, err := result.RowsAffected()
	if err != nil {
		return err
	}

	if rows == 0 {
		return sql.ErrNoRows
	}

	return nil
}

// GetProcessedVideo retrieves a processed video by its processing ID
func (o *MLOperations) GetProcessedVideo(processingID string) (*models.ProcessedVideo, error) {
	video := &models.ProcessedVideo{}
	query := `
		SELECT id, video_path, processing_id, status, created_at, updated_at, user_id, error_message
		FROM processed_videos
		WHERE processing_id = $1`

	err := o.db.QueryRow(query, processingID).Scan(
		&video.ID,
		&video.VideoPath,
		&video.ProcessingID,
		&video.Status,
		&video.CreatedAt,
		&video.UpdatedAt,
		&video.UserID,
		&video.ErrorMessage,
	)
	if err != nil {
		return nil, err
	}

	return video, nil
}

// CreateDetectionResult creates a new detection result
func (o *MLOperations) CreateDetectionResult(detection *models.DetectionResult) error {
	query := `
		INSERT INTO detection_results (video_id, frame_number, detection_type, confidence, bbox, metadata)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING id, created_at`

	return o.db.QueryRow(
		query,
		detection.VideoID,
		detection.FrameNumber,
		detection.DetectionType,
		detection.Confidence,
		detection.BBox,
		detection.Metadata,
	).Scan(&detection.ID, &detection.CreatedAt)
}

// GetDetectionResults retrieves all detection results for a video
func (o *MLOperations) GetDetectionResults(videoID int) ([]models.DetectionResult, error) {
	query := `
		SELECT id, video_id, frame_number, detection_type, confidence, bbox, created_at, metadata
		FROM detection_results
		WHERE video_id = $1
		ORDER BY frame_number`

	rows, err := o.db.Query(query, videoID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []models.DetectionResult
	for rows.Next() {
		var result models.DetectionResult
		err := rows.Scan(
			&result.ID,
			&result.VideoID,
			&result.FrameNumber,
			&result.DetectionType,
			&result.Confidence,
			&result.BBox,
			&result.CreatedAt,
			&result.Metadata,
		)
		if err != nil {
			return nil, err
		}
		results = append(results, result)
	}

	return results, rows.Err()
}

// CreateServiceComparison creates a new service comparison record
func (o *MLOperations) CreateServiceComparison(comparison *models.ServiceComparison) error {
	query := `
		INSERT INTO service_comparisons (video_id, service_name, processing_time, memory_usage, accuracy, metadata)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING id, created_at`

	return o.db.QueryRow(
		query,
		comparison.VideoID,
		comparison.ServiceName,
		comparison.ProcessingTime,
		comparison.MemoryUsage,
		comparison.Accuracy,
		comparison.Metadata,
	).Scan(&comparison.ID, &comparison.CreatedAt)
}

// CreatePerformanceMetric creates a new performance metric record
func (o *MLOperations) CreatePerformanceMetric(metric *models.PerformanceMetric) error {
	query := `
		INSERT INTO performance_metrics (service_name, metric_type, value, timestamp, metadata)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id`

	return o.db.QueryRow(
		query,
		metric.ServiceName,
		metric.MetricType,
		metric.Value,
		metric.Timestamp,
		metric.Metadata,
	).Scan(&metric.ID)
}

// GetPerformanceMetrics retrieves performance metrics for a service
func (o *MLOperations) GetPerformanceMetrics(serviceName string, startTime, endTime time.Time) ([]models.PerformanceMetric, error) {
	query := `
		SELECT id, service_name, metric_type, value, timestamp, metadata
		FROM performance_metrics
		WHERE service_name = $1 AND timestamp BETWEEN $2 AND $3
		ORDER BY timestamp`

	rows, err := o.db.Query(query, serviceName, startTime, endTime)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var metrics []models.PerformanceMetric
	for rows.Next() {
		var metric models.PerformanceMetric
		err := rows.Scan(
			&metric.ID,
			&metric.ServiceName,
			&metric.MetricType,
			&metric.Value,
			&metric.Timestamp,
			&metric.Metadata,
		)
		if err != nil {
			return nil, err
		}
		metrics = append(metrics, metric)
	}

	return metrics, rows.Err()
}
