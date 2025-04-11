package db

import (
	"time"
)

// Model represents a machine learning model in the system
type Model struct {
	ID               int64      `json:"id" db:"id"`
	Name             string     `json:"name" db:"name"`
	Version          string     `json:"version" db:"version"`
	Type             string     `json:"type" db:"type"`
	Framework        string     `json:"framework" db:"framework"`
	Status           string     `json:"status" db:"status"`
	CloudPlatform    string     `json:"cloud_platform" db:"cloud_platform"`
	EndpointURL      string     `json:"endpoint_url" db:"endpoint_url"`
	Accuracy         float64    `json:"accuracy" db:"accuracy"`
	AvgInferenceTime float64    `json:"avg_inference_time" db:"avg_inference_time"`
	CreatedAt        time.Time  `json:"created_at" db:"created_at"`
	UpdatedAt        time.Time  `json:"updated_at" db:"updated_at"`
	DeployedAt       *time.Time `json:"deployed_at,omitempty" db:"deployed_at"`
	CreatedBy        int64      `json:"created_by" db:"created_by"`
}

// ModelMetrics represents performance metrics for a model
type ModelMetrics struct {
	ID               int64     `json:"id" db:"id"`
	ModelID          int64     `json:"model_id" db:"model_id"`
	Date             time.Time `json:"date" db:"date"`
	InferenceCount   int64     `json:"inference_count" db:"inference_count"`
	AvgInferenceTime float64   `json:"avg_inference_time" db:"avg_inference_time"`
	AvgConfidence    float64   `json:"avg_confidence" db:"avg_confidence"`
	ErrorCount       int64     `json:"error_count" db:"error_count"`
}

// Detection represents a single object detection result
type Detection struct {
	ID          int64     `json:"id" db:"id"`
	ImageID     int64     `json:"image_id" db:"image_id"`
	Class       string    `json:"class" db:"class"`
	Score       float64   `json:"score" db:"score"`
	BoundingBox BBox      `json:"bounding_box" db:"bounding_box"`
	CreatedAt   time.Time `json:"created_at" db:"created_at"`
}

// BBox represents a bounding box for detected objects
type BBox struct {
	X      float64 `json:"x" db:"x"`
	Y      float64 `json:"y" db:"y"`
	Width  float64 `json:"width" db:"width"`
	Height float64 `json:"height" db:"height"`
}

// Image represents a processed image or video frame
type Image struct {
	ID             int64     `json:"id" db:"id"`
	Source         string    `json:"source" db:"source"`
	ProcessedAt    time.Time `json:"processed_at" db:"processed_at"`
	CloudPlatform  string    `json:"cloud_platform" db:"cloud_platform"`
	ProcessingTime float64   `json:"processing_time" db:"processing_time"`
}

// CloudMetrics represents performance and cost metrics for cloud platforms
type CloudMetrics struct {
	ID           int64     `json:"id" db:"id"`
	Platform     string    `json:"platform" db:"platform"`
	RequestCount int64     `json:"request_count" db:"request_count"`
	AvgLatency   float64   `json:"avg_latency" db:"avg_latency"`
	Cost         float64   `json:"cost" db:"cost"`
	Date         time.Time `json:"date" db:"date"`
}

// User represents an authenticated user
type User struct {
	ID        int64     `json:"id" db:"id"`
	Email     string    `json:"email" db:"email"`
	Password  string    `json:"-" db:"password_hash"`
	Role      string    `json:"role" db:"role"`
	CreatedAt time.Time `json:"created_at" db:"created_at"`
	UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
}

// Location represents a location in the system
type Location struct {
	ID          int64     `json:"id"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Latitude    float64   `json:"latitude"`
	Longitude   float64   `json:"longitude"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}
