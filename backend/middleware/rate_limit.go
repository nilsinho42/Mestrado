package middleware

import (
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
)

type RateLimiter struct {
	requests map[string]*tokenBucket
	mu       sync.RWMutex
}

type tokenBucket struct {
	tokens     float64
	lastRefill time.Time
	rate       float64
	capacity   float64
}

func NewRateLimiter(rate float64, capacity float64) *RateLimiter {
	return &RateLimiter{
		requests: make(map[string]*tokenBucket),
	}
}

func (rl *RateLimiter) RateLimit() gin.HandlerFunc {
	return func(c *gin.Context) {
		clientIP := c.ClientIP()

		rl.mu.Lock()
		bucket, exists := rl.requests[clientIP]
		if !exists {
			bucket = &tokenBucket{
				tokens:     10, // Initial tokens
				lastRefill: time.Now(),
				rate:       10, // 10 requests per second
				capacity:   10,
			}
			rl.requests[clientIP] = bucket
		}
		rl.mu.Unlock()

		// Refill tokens based on time passed
		now := time.Now()
		timePassed := now.Sub(bucket.lastRefill).Seconds()
		bucket.tokens = min(bucket.capacity, bucket.tokens+timePassed*bucket.rate)
		bucket.lastRefill = now

		// Check if request can be processed
		if bucket.tokens < 1 {
			c.JSON(http.StatusTooManyRequests, gin.H{"error": "Rate limit exceeded"})
			c.Abort()
			return
		}

		// Consume a token
		bucket.tokens--

		c.Next()
	}
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
