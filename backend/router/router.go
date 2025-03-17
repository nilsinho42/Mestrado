package router

import (
	"database/sql"

	"github.com/gin-gonic/gin"
	"github.com/nilsinho42/Mestrado/controllers"
	"github.com/nilsinho42/Mestrado/middleware"
)

func SetupRouter(db *sql.DB) *gin.Engine {
	// Initialize handlers
	controllers.InitHandlers(db, nil) // TODO: Pass logger

	r := gin.New()

	// Global middleware
	r.Use(gin.Recovery())
	r.Use(middleware.CORS())
	r.Use(gin.Logger())

	// Initialize rate limiters
	globalRateLimiter := middleware.NewRateLimiter(100, 100)  // 100 requests per second
	detectionRateLimiter := middleware.NewRateLimiter(10, 10) // 10 requests per second

	// Public routes
	public := r.Group("/api")
	{
		public.POST("/login", controllers.HandleLogin)
		public.POST("/register", controllers.HandleRegister)
	}

	// Protected routes
	protected := r.Group("/api")
	protected.Use(middleware.AuthMiddleware())
	protected.Use(globalRateLimiter.RateLimit())
	{
		// Location routes
		protected.GET("/locations", controllers.HandleListLocations)
		protected.POST("/locations", controllers.HandleCreateLocation)
		protected.GET("/locations/:id", controllers.HandleGetLocation)
		protected.PUT("/locations/:id", controllers.HandleUpdateLocation)
		protected.DELETE("/locations/:id", controllers.HandleDeleteLocation)

		// Detection routes
		detections := protected.Group("/detections")
		detections.Use(detectionRateLimiter.RateLimit())
		{
			detections.POST("/analyze", controllers.HandleAnalyzeImage)
			detections.GET("/stats", controllers.HandleGetStats)
		}

		// Model routes
		models := protected.Group("/models")
		{
			models.GET("", controllers.HandleListModels)
			models.POST("", controllers.HandleRegisterModel)
			models.GET("/compare", controllers.HandleCompareModels)
			models.GET("/:id/metrics", controllers.HandleGetModelMetrics)
			models.PUT("/:id", controllers.HandleUpdateModel)
			models.POST("/:id/deploy", controllers.HandleDeployModel)
		}

		// Cloud routes
		cloud := protected.Group("/cloud")
		{
			cloud.GET("/costs", controllers.HandleGetCloudCosts)
			cloud.GET("/performance", controllers.HandleGetCloudPerformance)
		}
	}

	return r
}
