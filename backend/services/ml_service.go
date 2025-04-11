package services

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// MLServiceConfig contains configuration for the ML service
type MLServiceConfig struct {
	BaseURL string
	DB      *sql.DB
}

// MLService handles communication with the ML service
type MLService struct {
	config MLServiceConfig
	client *http.Client
}

// NewMLService creates a new ML service instance
func NewMLService(config MLServiceConfig) *MLService {
	return &MLService{
		config: config,
		client: &http.Client{
			Timeout: 60 * time.Second, // Longer timeout for video processing
		},
	}
}

// UploadVideo uploads a video file to the ML service for processing
func (s *MLService) UploadVideo(file *multipart.FileHeader) (string, error) {
	// Create a unique processing ID
	processingID := fmt.Sprintf("proc_%d", time.Now().UnixNano())

	// Save the file locally first
	uploadDir := "uploads"
	if err := os.MkdirAll(uploadDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create upload directory: %w", err)
	}

	dst := filepath.Join(uploadDir, filepath.Base(file.Filename))
	src, err := file.Open()
	if err != nil {
		return "", fmt.Errorf("failed to open uploaded file: %w", err)
	}
	defer src.Close()

	dstFile, err := os.Create(dst)
	if err != nil {
		return "", fmt.Errorf("failed to create destination file: %w", err)
	}
	defer dstFile.Close()

	if _, err = io.Copy(dstFile, src); err != nil {
		return "", fmt.Errorf("failed to copy file content: %w", err)
	}

	// Create a multipart form for the API request
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// Add the processing ID to the form
	if err := writer.WriteField("processing_id", processingID); err != nil {
		return "", fmt.Errorf("failed to add processing ID to form: %w", err)
	}

	// Add the file to the form
	src.Seek(0, 0) // Reset file position
	part, err := writer.CreateFormFile("video", filepath.Base(file.Filename))
	if err != nil {
		return "", fmt.Errorf("failed to create form file: %w", err)
	}

	if _, err = io.Copy(part, src); err != nil {
		return "", fmt.Errorf("failed to copy file to form: %w", err)
	}

	// Close the writer
	if err := writer.Close(); err != nil {
		return "", fmt.Errorf("failed to close form writer: %w", err)
	}

	// Create and send the request
	req, err := http.NewRequest("POST", fmt.Sprintf("%s/api/videos/process", s.config.BaseURL), body)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := s.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request to ML service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		bodyContent, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ML service returned error (status %d): %s", resp.StatusCode, string(bodyContent))
	}

	// Store initial record in database as pending
	// This would normally be handled by the database operations but we're simplifying

	return processingID, nil
}

// GetVideoStatus checks the status of a video processing job
func (s *MLService) GetVideoStatus(processingID string) (map[string]interface{}, error) {
	req, err := http.NewRequest("GET", fmt.Sprintf("%s/api/videos/%s/status", s.config.BaseURL, processingID), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to ML service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyContent, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML service returned error (status %d): %s", resp.StatusCode, string(bodyContent))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result, nil
}

// GetMetrics retrieves metrics for all services
func (s *MLService) GetMetrics() (map[string]interface{}, error) {
	req, err := http.NewRequest("GET", fmt.Sprintf("%s/api/metrics", s.config.BaseURL), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to ML service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyContent, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML service returned error (status %d): %s", resp.StatusCode, string(bodyContent))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result, nil
}

// GetMetricsBySource retrieves metrics for a specific source
func (s *MLService) GetMetricsBySource(source string) (map[string]interface{}, error) {
	req, err := http.NewRequest("GET", fmt.Sprintf("%s/api/metrics/%s", s.config.BaseURL, source), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to ML service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyContent, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML service returned error (status %d): %s", resp.StatusCode, string(bodyContent))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result, nil
}
