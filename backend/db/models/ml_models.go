package models

import (
	"time"
)

// ProcessedVideo represents a video that has been processed by the ML service
type ProcessedVideo struct {
	ID           int       `json:"id" db:"id"`
	VideoPath    string    `json:"video_path" db:"video_path"`
	ProcessingID string    `json:"processing_id" db:"processing_id"`
	Status       string    `json:"status" db:"status"`
	CreatedAt    time.Time `json:"created_at" db:"created_at"`
	UpdatedAt    time.Time `json:"updated_at" db:"updated_at"`
	UserID       int       `json:"user_id" db:"user_id"`
	ErrorMessage string    `json:"error_message,omitempty" db:"error_message"`
}

// DetectionResult represents a single detection from the ML service
type DetectionResult struct {
	ID            int       `json:"id" db:"id"`
	VideoID       int       `json:"video_id" db:"video_id"`
	FrameNumber   int       `json:"frame_number" db:"frame_number"`
	DetectionType string    `json:"detection_type" db:"detection_type"`
	Confidence    float64   `json:"confidence" db:"confidence"`
	BBox          []float64 `json:"bbox" db:"bbox"`
	CreatedAt     time.Time `json:"created_at" db:"created_at"`
	Metadata      []byte    `json:"metadata,omitempty" db:"metadata"`
}

// ServiceComparison represents a comparison between different ML services
type ServiceComparison struct {
	ID             int       `json:"id" db:"id"`
	VideoID        int       `json:"video_id" db:"video_id"`
	ServiceName    string    `json:"service_name" db:"service_name"`
	ProcessingTime float64   `json:"processing_time" db:"processing_time"`
	MemoryUsage    float64   `json:"memory_usage" db:"memory_usage"`
	Accuracy       float64   `json:"accuracy,omitempty" db:"accuracy"`
	CreatedAt      time.Time `json:"created_at" db:"created_at"`
	Metadata       []byte    `json:"metadata,omitempty" db:"metadata"`
}

// PerformanceMetric represents a performance metric for a service
type PerformanceMetric struct {
	ID          int       `json:"id" db:"id"`
	ServiceName string    `json:"service_name" db:"service_name"`
	MetricType  string    `json:"metric_type" db:"metric_type"`
	Value       float64   `json:"value" db:"value"`
	Timestamp   time.Time `json:"timestamp" db:"timestamp"`
	Metadata    []byte    `json:"metadata,omitempty" db:"metadata"`
}
