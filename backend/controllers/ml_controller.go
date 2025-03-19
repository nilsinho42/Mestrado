package controllers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/nilsinho42/Mestrado/services"
)

// MLController handles ML-related HTTP requests
type MLController struct {
	mlService *services.MLService
}

// NewMLController creates a new ML controller instance
func NewMLController(mlService *services.MLService) *MLController {
	return &MLController{
		mlService: mlService,
	}
}

// ProcessVideo handles video processing requests
func (c *MLController) ProcessVideo(ctx *gin.Context) {
	var req struct {
		VideoPath string `json:"video_path" binding:"required"`
	}

	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Get user ID from context (set by auth middleware)
	userID, exists := ctx.Get("user_id")
	if !exists {
		ctx.JSON(http.StatusUnauthorized, gin.H{"error": "user not authenticated"})
		return
	}

	processingID, err := c.mlService.ProcessVideo(req.VideoPath, userID.(int))
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(http.StatusAccepted, gin.H{
		"processing_id": processingID,
		"status":        "pending",
	})
}

// GetVideoStatus handles video status requests
func (c *MLController) GetVideoStatus(ctx *gin.Context) {
	processingID := ctx.Param("processing_id")
	if processingID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "processing_id is required"})
		return
	}

	status, err := c.mlService.GetVideoStatus(processingID)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{
		"status": status,
	})
}

// GetDetectionStats handles detection statistics requests
func (c *MLController) GetDetectionStats(ctx *gin.Context) {
	stats, err := c.mlService.GetDetectionStats()
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(http.StatusOK, stats)
}
