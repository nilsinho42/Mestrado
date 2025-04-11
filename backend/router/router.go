package router

import (
	"database/sql"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/nilsinho42/Mestrado/controllers"
	"github.com/nilsinho42/Mestrado/services"
)

func SetupRouter(db *sql.DB, logger *zap.Logger, mlServiceURL string) *gin.Engine {
	router := gin.Default()

	// Configure CORS to allow requests from frontend
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:3000"},
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Content-Length"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * 60 * 60, // 12 hours
	}))

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
