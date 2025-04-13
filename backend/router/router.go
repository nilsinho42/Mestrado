package router

import (
	"database/sql"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/nilsinho42/Mestrado/controllers"
	"github.com/nilsinho42/Mestrado/services"
)

// Maximum file size for uploads (100MB)
const MaxFileSize = 100 * 1024 * 1024

func SetupRouter(db *sql.DB, logger *zap.Logger, mlServiceURL string) *gin.Engine {
	router := gin.Default()

	// Set maximum multipart memory
	router.MaxMultipartMemory = MaxFileSize

	// Configure CORS to allow requests from frontend
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:3000", "http://localhost:8081"},
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Content-Length"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * 60 * 60, // 12 hours
	}))

	// Add custom logging middleware for debugging requests
	router.Use(func(c *gin.Context) {
		// Log the incoming request
		logger.Info("Received request",
			zap.String("method", c.Request.Method),
			zap.String("path", c.Request.URL.Path),
			zap.String("client", c.ClientIP()),
		)

		c.Next()

		// Log the response status
		logger.Info("Request completed",
			zap.String("method", c.Request.Method),
			zap.String("path", c.Request.URL.Path),
			zap.Int("status", c.Writer.Status()),
		)
	})

	// Initialize ML service
	mlService := services.NewMLService(services.MLServiceConfig{
		BaseURL: mlServiceURL,
		DB:      db,
	})

	// Initialize controllers
	mlController := controllers.NewMLController(mlService, logger)

	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})

	// Video processing API endpoints
	api := router.Group("/api")
	{
		// Video processing routes
		api.POST("/videos/upload", mlController.UploadVideo)
		api.GET("/videos/:id/status", mlController.GetVideoStatus)
		api.GET("/metrics", mlController.GetMetrics)
		api.GET("/metrics/:source", mlController.GetMetricsBySource)
	}

	return router
}
