package router

import (
	"database/sql"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/nilsinho42/Mestrado/controllers"
	"github.com/nilsinho42/Mestrado/middleware"
	"github.com/nilsinho42/Mestrado/services"
)

func SetupRouter(db *sql.DB) *gin.Engine {
	router := gin.Default()

	// Configure rate limits for specific endpoints
	rateLimits := map[string]int{
		"/api/v1/ml/videos/process":    10, // 10 requests per minute
		"/api/v1/ml/videos/:id/status": 60, // 60 requests per minute
		"/api/v1/ml/detections/stats":  30, // 30 requests per minute
		"/api/v1/auth/logout":          30, // 30 requests per minute
	}

	// Initialize rate limiter
	rateLimiter := middleware.NewEndpointRateLimiter(rateLimits)

	// Initialize ML service
	mlService := services.NewMLService(services.MLServiceConfig{
		BaseURL:     os.Getenv("ML_SERVICE_URL"),
		JWTSecret:   os.Getenv("ML_SERVICE_JWT_SECRET"),
		ServiceName: "golang_backend",
	})

	// Initialize controllers
	mlController := controllers.NewMLController(mlService)

	// Public routes
	router.GET("/health", controllers.HandleHealth)

	// Protected routes
	protected := router.Group("/api/v1")
	protected.Use(middleware.AuthMiddleware())
	protected.Use(rateLimiter.RateLimit())
	{
		// ML routes
		ml := protected.Group("/ml")
		{
			ml.POST("/videos/process", mlController.ProcessVideo)
			ml.GET("/videos/:id/status", mlController.GetVideoStatus)
			ml.GET("/detections/stats", mlController.GetDetectionStats)
		}

		// Auth routes
		protected.POST("/auth/logout", controllers.HandleLogout)
	}

	return router
}
