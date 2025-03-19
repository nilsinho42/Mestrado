package controllers

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// HandleHealth handles the health check endpoint
func HandleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "healthy",
	})
}

// HandleLogout handles user logout
func HandleLogout(c *gin.Context) {
	// TODO: Implement proper logout logic (e.g., invalidate JWT token)
	c.JSON(http.StatusOK, gin.H{
		"message": "Successfully logged out",
	})
}
