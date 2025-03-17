package controllers

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/lib/pq"
	"go.uber.org/zap"
	"golang.org/x/crypto/bcrypt"

	"github.com/nilsinho42/Mestrado/db"
	"github.com/nilsinho42/Mestrado/middleware"
)

var database *sql.DB
var logger *zap.Logger

func InitHandlers(db *sql.DB, log *zap.Logger) {
	database = db
	logger = log
}

type LoginRequest struct {
	Email    string `json:"email" binding:"required,email"`
	Password string `json:"password" binding:"required"`
}

type RegisterRequest struct {
	Email    string `json:"email" binding:"required,email"`
	Password string `json:"password" binding:"required,min=8"`
}

func HandleLogin(c *gin.Context) {
	var req LoginRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	var user db.User
	err := database.QueryRow(`
		SELECT id, email, password_hash, role
		FROM users
		WHERE email = $1
	`, req.Email).Scan(&user.ID, &user.Email, &user.Password, &user.Role)

	if err == sql.ErrNoRows {
		c.JSON(401, gin.H{"error": "Invalid credentials"})
		return
	} else if err != nil {
		logger.Error("Failed to query user", zap.Error(err))
		c.JSON(500, gin.H{"error": "Internal server error"})
		return
	}

	// Verify password
	err = bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(req.Password))
	if err != nil {
		c.JSON(401, gin.H{"error": "Invalid credentials"})
		return
	}

	// Generate token
	token, err := middleware.GenerateToken(user.ID, user.Role)
	if err != nil {
		logger.Error("Failed to generate token", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to generate token"})
		return
	}

	c.JSON(200, gin.H{
		"token": token,
		"user": gin.H{
			"id":    user.ID,
			"email": user.Email,
			"role":  user.Role,
		},
	})
}

func HandleRegister(c *gin.Context) {
	var req RegisterRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	// Check if user already exists
	var exists bool
	err := database.QueryRow("SELECT EXISTS(SELECT 1 FROM users WHERE email = $1)", req.Email).Scan(&exists)
	if err != nil {
		logger.Error("Failed to check user existence", zap.Error(err))
		c.JSON(500, gin.H{"error": "Internal server error"})
		return
	}

	if exists {
		c.JSON(400, gin.H{"error": "User already exists"})
		return
	}

	// Hash password
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
	if err != nil {
		logger.Error("Failed to hash password", zap.Error(err))
		c.JSON(500, gin.H{"error": "Internal server error"})
		return
	}

	// Create user
	var userID int64
	err = database.QueryRow(`
		INSERT INTO users (email, password_hash, role)
		VALUES ($1, $2, 'user')
		RETURNING id
	`, req.Email, hashedPassword).Scan(&userID)

	if err != nil {
		logger.Error("Failed to create user", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to create user"})
		return
	}

	// Generate token
	token, err := middleware.GenerateToken(userID, "user")
	if err != nil {
		logger.Error("Failed to generate token", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to generate token"})
		return
	}

	c.JSON(201, gin.H{
		"token": token,
		"user": gin.H{
			"id":    userID,
			"email": req.Email,
			"role":  "user",
		},
	})
}

type AnalyzeRequest struct {
	ImageURL      string `json:"image_url" binding:"required"`
	CloudPlatform string `json:"cloud_platform" binding:"required,oneof=aws azure gcp"`
}

// MLResponse represents the response from the ML model API
type MLResponse struct {
	Detections []struct {
		Class       string  `json:"class"`
		Confidence  float64 `json:"confidence"`
		BoundingBox struct {
			X      float64 `json:"x"`
			Y      float64 `json:"y"`
			Width  float64 `json:"width"`
			Height float64 `json:"height"`
		} `json:"bbox"`
	} `json:"detections"`
	ProcessingTime float64 `json:"processing_time"`
}

func HandleAnalyzeImage(c *gin.Context) {
	var req AnalyzeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	// Record start time for processing
	startTime := time.Now()

	// Get ML model endpoint based on cloud platform
	modelEndpoint := getModelEndpoint(req.CloudPlatform)

	// Download image
	resp, err := http.Get(req.ImageURL)
	if err != nil {
		logger.Error("Failed to download image", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to download image"})
		return
	}
	defer resp.Body.Close()

	// Read image data
	imageData, err := io.ReadAll(resp.Body)
	if err != nil {
		logger.Error("Failed to read image data", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to read image data"})
		return
	}

	// Call ML model API
	mlResp, err := http.Post(modelEndpoint, "application/octet-stream", bytes.NewReader(imageData))
	if err != nil {
		logger.Error("Failed to call ML model", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to process image"})
		return
	}
	defer mlResp.Body.Close()

	// Parse ML model response
	var mlResult MLResponse
	if err := json.NewDecoder(mlResp.Body).Decode(&mlResult); err != nil {
		logger.Error("Failed to parse ML response", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to parse model response"})
		return
	}

	// Collect confidence scores for metrics
	confidenceScores := make([]float64, len(mlResult.Detections))
	for i, d := range mlResult.Detections {
		confidenceScores[i] = d.Confidence
	}

	// Calculate processing time
	processingTime := time.Since(startTime).Seconds()

	// Save image record
	var imageID int64
	err = database.QueryRow(`
		INSERT INTO images (source, cloud_platform, processing_time)
		VALUES ($1, $2, $3)
		RETURNING id
	`, req.ImageURL, req.CloudPlatform, processingTime).Scan(&imageID)

	if err != nil {
		logger.Error("Failed to save image record", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to save image record"})
		return
	}

	// Convert ML detections to our format and save them
	var detections []db.Detection
	for _, d := range mlResult.Detections {
		detection := db.Detection{
			ImageID: imageID,
			Class:   d.Class,
			Score:   d.Confidence,
			BoundingBox: db.BBox{
				X:      d.BoundingBox.X,
				Y:      d.BoundingBox.Y,
				Width:  d.BoundingBox.Width,
				Height: d.BoundingBox.Height,
			},
		}
		detections = append(detections, detection)

		// Save detection
		boundingBoxJSON, _ := json.Marshal(detection.BoundingBox)
		_, err = database.Exec(`
			INSERT INTO detections (image_id, class, score, bounding_box)
			VALUES ($1, $2, $3, $4)
		`, imageID, detection.Class, detection.Score, boundingBoxJSON)

		if err != nil {
			logger.Error("Failed to save detection", zap.Error(err))
		}
	}

	// Update cloud metrics
	_, err = database.Exec(`
		INSERT INTO cloud_metrics (platform, request_count, avg_latency, cost)
		VALUES ($1, 1, $2, $3)
		ON CONFLICT (platform, date)
		DO UPDATE SET
			request_count = cloud_metrics.request_count + 1,
			avg_latency = (cloud_metrics.avg_latency * cloud_metrics.request_count + $2) / (cloud_metrics.request_count + 1),
			cost = cloud_metrics.cost + $3
	`, req.CloudPlatform, processingTime, calculateCost(req.CloudPlatform, processingTime))

	if err != nil {
		logger.Error("Failed to update cloud metrics", zap.Error(err))
	}

	// Update model metrics
	_, err = database.Exec(`
		INSERT INTO model_metrics (
			model_id, date, inference_count, 
			avg_inference_time, avg_confidence
		)
		VALUES (
			(SELECT id FROM models WHERE cloud_platform = $1 AND status = 'deployed' LIMIT 1),
			CURRENT_DATE,
			1,
			$2,
			(SELECT AVG(score) FROM unnest($3::float[]) AS score)
		)
		ON CONFLICT (model_id, date)
		DO UPDATE SET
			inference_count = model_metrics.inference_count + 1,
			avg_inference_time = (model_metrics.avg_inference_time * model_metrics.inference_count + $2) 
				/ (model_metrics.inference_count + 1),
			avg_confidence = (model_metrics.avg_confidence * model_metrics.inference_count + 
				(SELECT AVG(score) FROM unnest($3::float[]) AS score)) 
				/ (model_metrics.inference_count + 1)
	`, req.CloudPlatform, processingTime, pq.Array(confidenceScores))

	if err != nil {
		logger.Error("Failed to update model metrics", zap.Error(err))
	}

	c.JSON(200, gin.H{
		"image_id":        imageID,
		"detections":      detections,
		"processing_time": processingTime,
	})
}

func getModelEndpoint(platform string) string {
	endpoints := map[string]string{
		"aws":   os.Getenv("AWS_MODEL_ENDPOINT"),
		"azure": os.Getenv("AZURE_MODEL_ENDPOINT"),
		"gcp":   os.Getenv("GCP_MODEL_ENDPOINT"),
	}
	return endpoints[platform]
}

func HandleGetStats(c *gin.Context) {
	// Get detection statistics from database
	rows, err := database.Query(`
		SELECT 
			COUNT(DISTINCT i.id) as total_images,
			COUNT(d.id) as total_detections,
			AVG(i.processing_time) as avg_processing_time
		FROM images i
		LEFT JOIN detections d ON i.id = d.image_id
	`)

	if err != nil {
		logger.Error("Failed to get statistics", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to get statistics"})
		return
	}
	defer rows.Close()

	var stats struct {
		TotalImages       int     `json:"total_images"`
		TotalDetections   int     `json:"total_detections"`
		AvgProcessingTime float64 `json:"avg_processing_time"`
	}

	if rows.Next() {
		err = rows.Scan(&stats.TotalImages, &stats.TotalDetections, &stats.AvgProcessingTime)
		if err != nil {
			logger.Error("Failed to scan statistics", zap.Error(err))
			c.JSON(500, gin.H{"error": "Failed to scan statistics"})
			return
		}
	}

	c.JSON(200, stats)
}

func HandleListModels(c *gin.Context) {
	// Get query parameters
	status := c.DefaultQuery("status", "")
	platform := c.DefaultQuery("platform", "")

	// Build query
	query := `
		SELECT id, name, version, type, framework, status, 
		       cloud_platform, endpoint_url, accuracy, avg_inference_time,
		       created_at, updated_at, deployed_at
		FROM models
		WHERE 1=1
	`
	var args []interface{}
	var argNum int

	if status != "" {
		argNum++
		query += fmt.Sprintf(" AND status = $%d", argNum)
		args = append(args, status)
	}

	if platform != "" {
		argNum++
		query += fmt.Sprintf(" AND cloud_platform = $%d", argNum)
		args = append(args, platform)
	}

	query += " ORDER BY created_at DESC"

	// Execute query
	rows, err := database.Query(query, args...)
	if err != nil {
		logger.Error("Failed to list models", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to list models"})
		return
	}
	defer rows.Close()

	var models []db.Model
	for rows.Next() {
		var m db.Model
		err := rows.Scan(
			&m.ID, &m.Name, &m.Version, &m.Type, &m.Framework, &m.Status,
			&m.CloudPlatform, &m.EndpointURL, &m.Accuracy, &m.AvgInferenceTime,
			&m.CreatedAt, &m.UpdatedAt, &m.DeployedAt,
		)
		if err != nil {
			logger.Error("Failed to scan model", zap.Error(err))
			continue
		}
		models = append(models, m)
	}

	c.JSON(200, models)
}

func HandleCompareModels(c *gin.Context) {
	// Get model comparison metrics from database
	rows, err := database.Query(`
		SELECT 
			cloud_platform,
			AVG(processing_time) as avg_processing_time,
			COUNT(*) as total_requests
		FROM images
		GROUP BY cloud_platform
	`)

	if err != nil {
		logger.Error("Failed to get model comparison", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to get model comparison"})
		return
	}
	defer rows.Close()

	comparisons := make(map[string]map[string]interface{})
	for rows.Next() {
		var platform string
		var avgTime float64
		var totalReqs int

		err = rows.Scan(&platform, &avgTime, &totalReqs)
		if err != nil {
			logger.Error("Failed to scan comparison", zap.Error(err))
			continue
		}

		comparisons[platform] = map[string]interface{}{
			"avg_processing_time": avgTime,
			"total_requests":      totalReqs,
		}
	}

	c.JSON(200, comparisons)
}

func HandleGetCloudCosts(c *gin.Context) {
	// Get cloud costs from database
	rows, err := database.Query(`
		SELECT 
			platform,
			SUM(cost) as total_cost,
			SUM(request_count) as total_requests,
			AVG(avg_latency) as avg_latency
		FROM cloud_metrics
		GROUP BY platform
	`)

	if err != nil {
		logger.Error("Failed to get cloud costs", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to get cloud costs"})
		return
	}
	defer rows.Close()

	costs := make(map[string]map[string]interface{})
	for rows.Next() {
		var platform string
		var totalCost, avgLatency float64
		var totalReqs int64

		err = rows.Scan(&platform, &totalCost, &totalReqs, &avgLatency)
		if err != nil {
			logger.Error("Failed to scan costs", zap.Error(err))
			continue
		}

		costs[platform] = map[string]interface{}{
			"total_cost":     totalCost,
			"total_requests": totalReqs,
			"avg_latency":    avgLatency,
		}
	}

	c.JSON(200, costs)
}

func HandleGetCloudPerformance(c *gin.Context) {
	// Get detailed performance metrics
	rows, err := database.Query(`
		SELECT 
			platform,
			date,
			request_count,
			avg_latency,
			cost
		FROM cloud_metrics
		ORDER BY date DESC
		LIMIT 30
	`)

	if err != nil {
		logger.Error("Failed to get cloud performance", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to get cloud performance"})
		return
	}
	defer rows.Close()

	var metrics []db.CloudMetrics
	for rows.Next() {
		var m db.CloudMetrics
		err = rows.Scan(&m.Platform, &m.Date, &m.RequestCount, &m.AvgLatency, &m.Cost)
		if err != nil {
			logger.Error("Failed to scan performance metrics", zap.Error(err))
			continue
		}
		metrics = append(metrics, m)
	}

	c.JSON(200, metrics)
}

func calculateCost(platform string, processingTime float64) float64 {
	// Mock cost calculation based on processing time and platform
	// In reality, this would use actual cloud pricing
	baseCost := map[string]float64{
		"aws":   0.0000015, // per 100ms
		"azure": 0.0000014, // per 100ms
		"gcp":   0.0000016, // per 100ms
	}

	return baseCost[platform] * (processingTime * 10) // Convert to 100ms units
}

// ModelRegistrationRequest represents the request to register a new model
type ModelRegistrationRequest struct {
	Name          string  `json:"name" binding:"required"`
	Version       string  `json:"version" binding:"required"`
	Type          string  `json:"type" binding:"required,oneof=yolo rcnn faster-rcnn"`
	Framework     string  `json:"framework" binding:"required"`
	CloudPlatform string  `json:"cloud_platform" binding:"required,oneof=aws azure gcp"`
	EndpointURL   string  `json:"endpoint_url" binding:"required,url"`
	Accuracy      float64 `json:"accuracy" binding:"required"`
}

func HandleRegisterModel(c *gin.Context) {
	var req ModelRegistrationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	// Get user ID from context
	userID := c.GetInt64("user_id")

	// Check if model version already exists
	var exists bool
	err := database.QueryRow(`
		SELECT EXISTS(
			SELECT 1 FROM models 
			WHERE name = $1 AND version = $2 AND cloud_platform = $3
		)
	`, req.Name, req.Version, req.CloudPlatform).Scan(&exists)

	if err != nil {
		logger.Error("Failed to check model existence", zap.Error(err))
		c.JSON(500, gin.H{"error": "Internal server error"})
		return
	}

	if exists {
		c.JSON(400, gin.H{"error": "Model version already exists"})
		return
	}

	// Create new model
	var modelID int64
	err = database.QueryRow(`
		INSERT INTO models (
			name, version, type, framework, cloud_platform, 
			endpoint_url, accuracy, created_by
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		RETURNING id
	`, req.Name, req.Version, req.Type, req.Framework, req.CloudPlatform,
		req.EndpointURL, req.Accuracy, userID).Scan(&modelID)

	if err != nil {
		logger.Error("Failed to create model", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to register model"})
		return
	}

	c.JSON(201, gin.H{
		"id":      modelID,
		"message": "Model registered successfully",
	})
}

func HandleDeployModel(c *gin.Context) {
	modelID := c.Param("id")

	// Update model status to deployed
	result, err := database.Exec(`
		UPDATE models 
		SET status = 'deployed',
		    deployed_at = CURRENT_TIMESTAMP
		WHERE id = $1
	`, modelID)

	if err != nil {
		logger.Error("Failed to deploy model", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to deploy model"})
		return
	}

	rowsAffected, _ := result.RowsAffected()
	if rowsAffected == 0 {
		c.JSON(404, gin.H{"error": "Model not found"})
		return
	}

	c.JSON(200, gin.H{"message": "Model deployed successfully"})
}

func HandleGetModelMetrics(c *gin.Context) {
	modelID := c.Param("id")
	timeRange := c.DefaultQuery("range", "7d") // Default to 7 days

	// Calculate date range
	var startDate time.Time
	switch timeRange {
	case "24h":
		startDate = time.Now().AddDate(0, 0, -1)
	case "7d":
		startDate = time.Now().AddDate(0, 0, -7)
	case "30d":
		startDate = time.Now().AddDate(0, 0, -30)
	default:
		c.JSON(400, gin.H{"error": "Invalid time range. Supported values: 24h, 7d, 30d"})
		return
	}

	// Get metrics
	rows, err := database.Query(`
		SELECT date, inference_count, avg_inference_time, 
		       avg_confidence, error_count
		FROM model_metrics
		WHERE model_id = $1 AND date >= $2
		ORDER BY date DESC
	`, modelID, startDate)

	if err != nil {
		logger.Error("Failed to get model metrics", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to get model metrics"})
		return
	}
	defer rows.Close()

	var metrics []db.ModelMetrics
	for rows.Next() {
		var m db.ModelMetrics
		err := rows.Scan(
			&m.Date, &m.InferenceCount, &m.AvgInferenceTime,
			&m.AvgConfidence, &m.ErrorCount,
		)
		if err != nil {
			logger.Error("Failed to scan metrics", zap.Error(err))
			continue
		}
		metrics = append(metrics, m)
	}

	c.JSON(200, metrics)
}

type ModelUpdateRequest struct {
	Status      *string  `json:"status"`
	EndpointURL *string  `json:"endpoint_url"`
	Accuracy    *float64 `json:"accuracy"`
}

func HandleUpdateModel(c *gin.Context) {
	modelID := c.Param("id")

	var req ModelUpdateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	// Build update query
	query := "UPDATE models SET updated_at = CURRENT_TIMESTAMP"
	var args []interface{}
	args = append(args, modelID)
	argNum := 1

	if req.Status != nil {
		argNum++
		query += fmt.Sprintf(", status = $%d", argNum)
		args = append(args, *req.Status)
	}

	if req.EndpointURL != nil {
		argNum++
		query += fmt.Sprintf(", endpoint_url = $%d", argNum)
		args = append(args, *req.EndpointURL)
	}

	if req.Accuracy != nil {
		argNum++
		query += fmt.Sprintf(", accuracy = $%d", argNum)
		args = append(args, *req.Accuracy)
	}

	query += " WHERE id = $1"

	// Execute update
	result, err := database.Exec(query, args...)
	if err != nil {
		logger.Error("Failed to update model", zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to update model"})
		return
	}

	rowsAffected, _ := result.RowsAffected()
	if rowsAffected == 0 {
		c.JSON(404, gin.H{"error": "Model not found"})
		return
	}

	c.JSON(200, gin.H{"message": "Model updated successfully"})
}
