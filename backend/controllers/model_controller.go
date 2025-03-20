package controllers

import (
	"database/sql"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

type ModelController struct {
	db *sql.DB
}

func NewModelController(db *sql.DB) *ModelController {
	return &ModelController{db: db}
}

func (c *ModelController) ListModels(ctx *gin.Context) {
	// Get user ID from context (set by auth middleware)
	userID := ctx.GetInt64("user_id")

	// Get status filter from query parameters
	status := ctx.Query("status")

	var rows *sql.Rows
	var err error

	// Build query based on whether status filter is provided
	if status != "" {
		// Query models for the user with status filter
		rows, err = c.db.Query(`
			SELECT id, name, version, type, framework, status, cloud_platform, 
				   endpoint_url, accuracy, avg_inference_time, deployed_at
			FROM models 
			WHERE created_by = $1 AND status = $2
			ORDER BY created_at DESC`, userID, status)
	} else {
		// Query all models for the user
		rows, err = c.db.Query(`
			SELECT id, name, version, type, framework, status, cloud_platform, 
				   endpoint_url, accuracy, avg_inference_time, deployed_at
			FROM models 
			WHERE created_by = $1
			ORDER BY created_at DESC`, userID)
	}

	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch models"})
		return
	}
	defer rows.Close()

	// Initialize with empty array to ensure we don't return null
	var models []gin.H = []gin.H{}

	for rows.Next() {
		var (
			id            int
			name          string
			version       string
			modelType     string
			framework     string
			status        string
			cloudPlatform string
			endpointURL   string
			accuracy      float64
			avgInfTime    sql.NullFloat64
			deployedAt    sql.NullTime
		)

		err := rows.Scan(&id, &name, &version, &modelType, &framework, &status,
			&cloudPlatform, &endpointURL, &accuracy, &avgInfTime, &deployedAt)
		if err != nil {
			ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to scan model data"})
			return
		}

		model := gin.H{
			"id":            id,
			"name":          name,
			"version":       version,
			"type":          modelType,
			"framework":     framework,
			"status":        status,
			"cloudPlatform": cloudPlatform,
			"endpointUrl":   endpointURL,
			"accuracy":      accuracy,
		}

		if avgInfTime.Valid {
			model["avgInferenceTime"] = avgInfTime.Float64
		}
		if deployedAt.Valid {
			model["deployedAt"] = deployedAt.Time
		}

		models = append(models, model)
	}

	ctx.JSON(http.StatusOK, models)
}

func (c *ModelController) RegisterModel(ctx *gin.Context) {
	var model struct {
		Name          string  `json:"name" binding:"required"`
		Version       string  `json:"version" binding:"required"`
		Type          string  `json:"type" binding:"required"`
		Framework     string  `json:"framework" binding:"required"`
		CloudPlatform string  `json:"cloudPlatform" binding:"required"`
		EndpointURL   string  `json:"endpointUrl" binding:"required"`
		Accuracy      float64 `json:"accuracy" binding:"required"`
	}

	if err := ctx.ShouldBindJSON(&model); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	userID := ctx.GetInt64("user_id")

	result, err := c.db.Exec(`
		INSERT INTO models (name, version, type, framework, status, cloud_platform, 
						   endpoint_url, accuracy, created_by)
		VALUES ($1, $2, $3, $4, 'pending', $5, $6, $7, $8)
		RETURNING id`,
		model.Name, model.Version, model.Type, model.Framework,
		model.CloudPlatform, model.EndpointURL, model.Accuracy, userID)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to register model"})
		return
	}

	id, _ := result.LastInsertId()
	ctx.JSON(http.StatusCreated, gin.H{
		"id":     id,
		"status": "pending",
	})
}

func (c *ModelController) DeployModel(ctx *gin.Context) {
	modelID := ctx.Param("id")
	userID := ctx.GetInt64("user_id")

	// Verify ownership
	var count int
	err := c.db.QueryRow("SELECT COUNT(*) FROM models WHERE id = $1 AND created_by = $2",
		modelID, userID).Scan(&count)
	if err != nil || count == 0 {
		ctx.JSON(http.StatusNotFound, gin.H{"error": "Model not found"})
		return
	}

	// Update model status
	_, err = c.db.Exec(`
		UPDATE models 
		SET status = 'active', deployed_at = $1
		WHERE id = $2`,
		time.Now(), modelID)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to deploy model"})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{"status": "active"})
}

func (c *ModelController) GetModelMetrics(ctx *gin.Context) {
	modelID := ctx.Param("id")
	userID := ctx.GetInt64("user_id")
	timeRange := ctx.DefaultQuery("range", "7d")

	// Verify ownership
	var count int
	err := c.db.QueryRow("SELECT COUNT(*) FROM models WHERE id = $1 AND created_by = $2",
		modelID, userID).Scan(&count)
	if err != nil || count == 0 {
		ctx.JSON(http.StatusNotFound, gin.H{"error": "Model not found"})
		return
	}

	// Calculate date range
	var days int
	switch timeRange {
	case "24h":
		days = 1
	case "7d":
		days = 7
	case "30d":
		days = 30
	default:
		days = 7
	}

	rows, err := c.db.Query(`
		SELECT date, inference_count, avg_inference_time, avg_confidence, error_count
		FROM model_metrics
		WHERE model_id = $1
		  AND date >= CURRENT_DATE - $2::interval
		ORDER BY date DESC`,
		modelID, days)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch metrics"})
		return
	}
	defer rows.Close()

	var metrics []gin.H
	for rows.Next() {
		var (
			date           time.Time
			inferenceCount int64
			avgInfTime     float64
			avgConfidence  float64
			errorCount     int64
		)

		err := rows.Scan(&date, &inferenceCount, &avgInfTime, &avgConfidence, &errorCount)
		if err != nil {
			ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to scan metrics"})
			return
		}

		metrics = append(metrics, gin.H{
			"date":             date,
			"inferenceCount":   inferenceCount,
			"avgInferenceTime": avgInfTime,
			"avgConfidence":    avgConfidence,
			"errorCount":       errorCount,
		})
	}

	ctx.JSON(http.StatusOK, metrics)
}

func (c *ModelController) CompareModels(ctx *gin.Context) {
	userID := ctx.GetInt64("user_id")

	rows, err := c.db.Query(`
		SELECT m.name,
			   COALESCE(AVG(mm.avg_inference_time), 0) as avg_processing_time,
			   COALESCE(SUM(mm.inference_count), 0) as total_requests
		FROM models m
		LEFT JOIN model_metrics mm ON m.id = mm.model_id
		WHERE m.created_by = $1
		GROUP BY m.id, m.name`,
		userID)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to compare models"})
		return
	}
	defer rows.Close()

	comparison := make(map[string]gin.H)
	for rows.Next() {
		var (
			name              string
			avgProcessingTime float64
			totalRequests     int64
		)

		err := rows.Scan(&name, &avgProcessingTime, &totalRequests)
		if err != nil {
			ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to scan comparison data"})
			return
		}

		comparison[name] = gin.H{
			"avgProcessingTime": avgProcessingTime,
			"totalRequests":     totalRequests,
		}
	}

	ctx.JSON(http.StatusOK, comparison)
}
