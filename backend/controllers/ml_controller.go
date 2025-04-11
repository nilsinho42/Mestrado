package controllers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/nilsinho42/Mestrado/services"
	"go.uber.org/zap"
)

// MLController handles ML-related HTTP requests
type MLController struct {
	mlService *services.MLService
	logger    *zap.Logger
}

// NewMLController creates a new ML controller instance
func NewMLController(mlService *services.MLService, logger *zap.Logger) *MLController {
	return &MLController{
		mlService: mlService,
		logger:    logger,
	}
}

// UploadVideo handles video upload requests
func (c *MLController) UploadVideo(ctx *gin.Context) {
	// Get file from request
	file, err := ctx.FormFile("video")
	if err != nil {
		c.logger.Error("Failed to get file from request", zap.Error(err))
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "No video file provided"})
		return
	}

	// Check file size (limit to 100MB)
	if file.Size > 100*1024*1024 {
		c.logger.Warn("File too large", zap.Int64("size", file.Size))
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "File too large, max size is 100MB"})
		return
	}

	// Process the video
	processingID, err := c.mlService.UploadVideo(file)
	if err != nil {
		c.logger.Error("Failed to process video", zap.Error(err))
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(http.StatusAccepted, gin.H{
		"processing_id": processingID,
		"status":        "processing",
		"message":       "Video upload successful and processing has started",
	})
}

// GetVideoStatus handles video status requests
func (c *MLController) GetVideoStatus(ctx *gin.Context) {
	processingID := ctx.Param("id")
	if processingID == "" {
		c.logger.Warn("Missing processing ID in request")
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "processing_id is required"})
		return
	}

	status, err := c.mlService.GetVideoStatus(processingID)
	if err != nil {
		c.logger.Error("Failed to get video status", zap.Error(err), zap.String("processingID", processingID))
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(http.StatusOK, status)
}

// GetMetrics returns metrics for all detection sources
func (c *MLController) GetMetrics(ctx *gin.Context) {
	metrics, err := c.mlService.GetMetrics()
	if err != nil {
		c.logger.Error("Failed to get metrics", zap.Error(err))
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(http.StatusOK, metrics)
}

// GetMetricsBySource returns metrics for a specific detection source
func (c *MLController) GetMetricsBySource(ctx *gin.Context) {
	source := ctx.Param("source")
	if source == "" {
		c.logger.Warn("Missing source in request")
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "source parameter is required"})
		return
	}

	metrics, err := c.mlService.GetMetricsBySource(source)
	if err != nil {
		c.logger.Error("Failed to get metrics by source", zap.Error(err), zap.String("source", source))
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(http.StatusOK, metrics)
}
