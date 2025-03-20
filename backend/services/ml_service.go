package services

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/nilsinho42/Mestrado/db/models"
	"github.com/nilsinho42/Mestrado/db/operations"
)

// MLServiceConfig contains configuration for the ML service
type MLServiceConfig struct {
	BaseURL     string
	JWTSecret   string
	ServiceName string
	DB          *sql.DB
}

// MLService handles communication with the ML service
type MLService struct {
	config MLServiceConfig
	ops    *operations.MLOperations
	client *http.Client
}

// NewMLService creates a new ML service instance
func NewMLService(config MLServiceConfig) *MLService {
	return &MLService{
		config: config,
		ops:    operations.NewMLOperations(config.DB),
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// ProcessVideo sends a video processing request to the ML service
func (s *MLService) ProcessVideo(videoPath string, userID int) (string, error) {
	// Create a unique processing ID
	processingID := fmt.Sprintf("proc_%d", time.Now().UnixNano())

	// Create initial video record
	video := &models.ProcessedVideo{
		VideoPath:    videoPath,
		ProcessingID: processingID,
		Status:       "pending",
		UserID:       userID,
	}

	// Store in database
	if err := s.ops.CreateProcessedVideo(video); err != nil {
		return "", fmt.Errorf("failed to create video record: %w", err)
	}

	// Send request to ML service
	reqBody := map[string]interface{}{
		"video_path": videoPath,
		"user_id":    userID,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", fmt.Sprintf("%s/process_video", s.config.BaseURL), bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	// Add JWT token
	token, err := s.generateJWT()
	if err != nil {
		return "", fmt.Errorf("failed to generate JWT: %w", err)
	}
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ML service returned error: %s", string(body))
	}

	var result struct {
		ProcessingID     string                   `json:"processing_id"`
		Status           string                   `json:"status"`
		DetectionResults []models.DetectionResult `json:"detection_results"`
		Comparison       struct {
			ProcessingTime float64         `json:"processing_time"`
			MemoryUsage    float64         `json:"memory_usage"`
			Accuracy       float64         `json:"accuracy"`
			Metadata       json.RawMessage `json:"metadata"`
		} `json:"comparison"`
		Metric struct {
			MetricType string          `json:"metric_type"`
			Value      float64         `json:"value"`
			Metadata   json.RawMessage `json:"metadata"`
		} `json:"metric"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	// Update video status
	if err := s.ops.UpdateProcessedVideoStatus(processingID, result.Status, ""); err != nil {
		return "", fmt.Errorf("failed to update video status: %w", err)
	}

	// Store detection results
	for _, detection := range result.DetectionResults {
		detection.VideoID = video.ID
		if err := s.ops.CreateDetectionResult(&detection); err != nil {
			return "", fmt.Errorf("failed to store detection result: %w", err)
		}
	}

	// Store service comparison
	comparison := &models.ServiceComparison{
		VideoID:        video.ID,
		ServiceName:    s.config.ServiceName,
		ProcessingTime: result.Comparison.ProcessingTime,
		MemoryUsage:    result.Comparison.MemoryUsage,
		Accuracy:       result.Comparison.Accuracy,
		Metadata:       result.Comparison.Metadata,
	}

	if err := s.ops.CreateServiceComparison(comparison); err != nil {
		return "", fmt.Errorf("failed to store service comparison: %w", err)
	}

	// Store performance metric
	metric := &models.PerformanceMetric{
		ServiceName: s.config.ServiceName,
		MetricType:  result.Metric.MetricType,
		Value:       result.Metric.Value,
		Timestamp:   time.Now(),
		Metadata:    result.Metric.Metadata,
	}

	if err := s.ops.CreatePerformanceMetric(metric); err != nil {
		return "", fmt.Errorf("failed to store performance metric: %w", err)
	}

	return processingID, nil
}

// GetVideoStatus checks the status of a video processing job
func (s *MLService) GetVideoStatus(processingID string) (string, error) {
	// First check database
	video, err := s.ops.GetProcessedVideo(processingID)
	if err != nil {
		return "", fmt.Errorf("failed to get video status from database: %w", err)
	}

	// If video is completed or failed, return status from database
	if video.Status == "completed" || video.Status == "failed" {
		return video.Status, nil
	}

	// Otherwise, check ML service
	req, err := http.NewRequest("GET", fmt.Sprintf("%s/video_status/%s", s.config.BaseURL, processingID), nil)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	// Add JWT token
	token, err := s.generateJWT()
	if err != nil {
		return "", fmt.Errorf("failed to generate JWT: %w", err)
	}
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

	resp, err := s.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ML service returned error: %s", string(body))
	}

	var result struct {
		Status string `json:"status"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	// Update database with new status
	if err := s.ops.UpdateProcessedVideoStatus(processingID, result.Status, ""); err != nil {
		return "", fmt.Errorf("failed to update video status: %w", err)
	}

	return result.Status, nil
}

// GetDetectionStats retrieves statistics about detections
func (s *MLService) GetDetectionStats() (map[string]interface{}, error) {
	req, err := http.NewRequest("GET", fmt.Sprintf("%s/detection_stats", s.config.BaseURL), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Add JWT token
	token, err := s.generateJWT()
	if err != nil {
		return nil, fmt.Errorf("failed to generate JWT: %w", err)
	}
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML service returned error: %s", string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result, nil
}

// generateJWT creates a JWT token for ML service authentication
func (s *MLService) generateJWT() (string, error) {
	claims := map[string]interface{}{
		"service": s.config.ServiceName,
		"exp":     time.Now().Add(time.Hour).Unix(),
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims(claims))
	return token.SignedString([]byte(s.config.JWTSecret))
}
