package middleware

import (
	"net/http"
	"net/url"
	"strings"

	"github.com/gin-gonic/gin"
)

// ValidateImageURL checks if the provided URL is valid and points to an image
func ValidateImageURL() gin.HandlerFunc {
	return func(c *gin.Context) {
		var data struct {
			ImageURL string `json:"image_url"`
		}

		if err := c.ShouldBindJSON(&data); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
			c.Abort()
			return
		}

		// Validate URL format
		parsedURL, err := url.Parse(data.ImageURL)
		if err != nil || !parsedURL.IsAbs() {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid image URL"})
			c.Abort()
			return
		}

		// Check if URL points to an image
		ext := strings.ToLower(parsedURL.Path[strings.LastIndex(parsedURL.Path, ".")+1:])
		validExts := map[string]bool{
			"jpg":  true,
			"jpeg": true,
			"png":  true,
			"gif":  true,
			"bmp":  true,
			"webp": true,
		}

		if !validExts[ext] {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "Invalid image format. Supported formats: jpg, jpeg, png, gif, bmp, webp",
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

// ValidateCloudPlatform ensures the cloud platform is one of the supported options
func ValidateCloudPlatform() gin.HandlerFunc {
	return func(c *gin.Context) {
		var data struct {
			CloudPlatform string `json:"cloud_platform"`
		}

		if err := c.ShouldBindJSON(&data); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
			c.Abort()
			return
		}

		validPlatforms := map[string]bool{
			"aws":   true,
			"azure": true,
			"gcp":   true,
		}

		if !validPlatforms[strings.ToLower(data.CloudPlatform)] {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "Invalid cloud platform. Supported platforms: aws, azure, gcp",
			})
			c.Abort()
			return
		}

		c.Next()
	}
} 