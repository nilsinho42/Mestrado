package router

import (
	"database/sql"
	"os"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/nilsinho42/Mestrado/controllers"
	"github.com/nilsinho42/Mestrado/middleware"
	"github.com/nilsinho42/Mestrado/services"
)

func SetupRouter(db *sql.DB) *gin.Engine {
	router := gin.Default()

	// Configure CORS
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:3000"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * 60 * 60, // 12 hours
	}))

	// Configure rate limits for specific endpoints
	rateLimits := map[string]int{
		"/api/ml/videos/process":    10, // 10 requests per minute
		"/api/ml/videos/:id/status": 60, // 60 requests per minute
		"/api/ml/detections/stats":  30, // 30 requests per minute
		"/api/auth/logout":          30, // 30 requests per minute
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
	modelController := controllers.NewModelController(db)
	cloudController := controllers.NewCloudController(db)

	// Health check endpoint (must be first)
	router.GET("/health", controllers.HandleHealth)

	// Public auth endpoints
	auth := router.Group("/api")
	{
		auth.POST("/login", controllers.HandleLogin)
		auth.POST("/register", controllers.HandleRegister)
	}

	// Protected routes
	protected := router.Group("/api")
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

		// Model routes
		models := protected.Group("/models")
		{
			models.POST("/register", modelController.RegisterModel)
			models.POST("/:id/deploy", modelController.DeployModel)
			models.GET("/list", modelController.ListModels)
			models.GET("/:id/metrics", modelController.GetModelMetrics)
			models.GET("/compare", modelController.CompareModels)
		}

		// Cloud routes
		cloud := protected.Group("/cloud")
		{
			cloud.GET("/costs", cloudController.GetCloudCosts)
			cloud.GET("/performance", cloudController.GetCloudPerformance)
		}

		// Auth routes
		protected.POST("/auth/logout", controllers.HandleLogout)
	}

	return router
}
