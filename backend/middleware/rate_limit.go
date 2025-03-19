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

type EndpointRateLimiter struct {
	limiters map[string]*RateLimiter
	default_ *RateLimiter
}

func NewRateLimiter(rate float64, capacity float64) *RateLimiter {
	return &RateLimiter{
		requests: make(map[string]*tokenBucket),
	}
}

func NewEndpointRateLimiter(limits map[string]int) *EndpointRateLimiter {
	limiters := make(map[string]*RateLimiter)
	for path, rpm := range limits {
		// Convert requests per minute to requests per second
		rate := float64(rpm) / 60.0
		limiters[path] = NewRateLimiter(rate, rate)
	}

	return &EndpointRateLimiter{
		limiters: limiters,
		default_: NewRateLimiter(1.0, 1.0), // Default: 1 request per second
	}
}

func (erl *EndpointRateLimiter) RateLimit() gin.HandlerFunc {
	return func(c *gin.Context) {
		path := c.FullPath()
		limiter, exists := erl.limiters[path]
		if !exists {
			limiter = erl.default_
		}
		limiter.RateLimit()(c)
	}
}

func (rl *RateLimiter) RateLimit() gin.HandlerFunc {
	return func(c *gin.Context) {
		clientID := c.ClientIP() + ":" + c.Request.URL.Path

		rl.mu.Lock()
		bucket, exists := rl.requests[clientID]
		if !exists {
			bucket = &tokenBucket{
				tokens:     10, // Initial tokens
				lastRefill: time.Now(),
				rate:       10, // 10 requests per second
				capacity:   10,
			}
			rl.requests[clientID] = bucket
		}
		rl.mu.Unlock()

		// Refill tokens based on time passed
		now := time.Now()
		timePassed := now.Sub(bucket.lastRefill).Seconds()
		bucket.tokens = min(bucket.capacity, bucket.tokens+timePassed*bucket.rate)
		bucket.lastRefill = now

		// Check if request can be processed
		if bucket.tokens < 1 {
			c.Header("X-RateLimit-Limit", "10")
			c.Header("X-RateLimit-Reset", now.Add(time.Second).Format(time.RFC3339))
			c.Header("Retry-After", "1")

			c.JSON(http.StatusTooManyRequests, gin.H{
				"error":       "Rate limit exceeded",
				"reset":       now.Add(time.Second).Format(time.RFC3339),
				"retry_after": 1,
			})
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
