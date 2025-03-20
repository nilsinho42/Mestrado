package controllers

import (
	"database/sql"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

type CloudController struct {
	db *sql.DB
}

func NewCloudController(db *sql.DB) *CloudController {
	return &CloudController{db: db}
}

func (c *CloudController) GetCloudCosts(ctx *gin.Context) {
	rows, err := c.db.Query(`
		SELECT platform, date, request_count, avg_latency, cost
		FROM cloud_metrics
		WHERE date >= CURRENT_DATE - '7 days'::interval
		ORDER BY platform, date`)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch cloud costs"})
		return
	}
	defer rows.Close()

	metrics := make(map[string][]gin.H)
	for rows.Next() {
		var (
			platform   string
			date       time.Time
			reqCount   int64
			avgLatency float64
			cost       float64
		)

		err := rows.Scan(&platform, &date, &reqCount, &avgLatency, &cost)
		if err != nil {
			ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to scan cloud metrics"})
			return
		}

		metrics[platform] = append(metrics[platform], gin.H{
			"date":         date,
			"requestCount": reqCount,
			"avgLatency":   avgLatency,
			"cost":         cost,
		})
	}

	ctx.JSON(http.StatusOK, metrics)
}

func (c *CloudController) GetCloudPerformance(ctx *gin.Context) {
	rows, err := c.db.Query(`
		SELECT platform,
			   AVG(avg_latency) as avg_latency,
			   SUM(request_count) as total_requests,
			   SUM(cost) as total_cost
		FROM cloud_metrics
		WHERE date >= CURRENT_DATE - '7 days'::interval
		GROUP BY platform`)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch cloud performance"})
		return
	}
	defer rows.Close()

	var performance []gin.H
	for rows.Next() {
		var (
			platform      string
			avgLatency    float64
			totalRequests int64
			totalCost     float64
		)

		err := rows.Scan(&platform, &avgLatency, &totalRequests, &totalCost)
		if err != nil {
			ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to scan performance data"})
			return
		}

		performance = append(performance, gin.H{
			"platform":      platform,
			"avgLatency":    avgLatency,
			"totalRequests": totalRequests,
			"totalCost":     totalCost,
		})
	}

	ctx.JSON(http.StatusOK, performance)
}
