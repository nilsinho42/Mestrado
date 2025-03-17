package controllers

import (
	"database/sql"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/nilsinho42/Mestrado/db"
)

// LocationController handles location-related requests
type LocationController struct {
	db *sql.DB
}

// NewLocationController creates a new location controller
func NewLocationController(db *sql.DB) *LocationController {
	return &LocationController{db: db}
}

// HandleListLocations returns a list of all locations
func HandleListLocations(c *gin.Context) {
	rows, err := database.Query(`
		SELECT id, name, description, latitude, longitude, created_at, updated_at
		FROM locations
		ORDER BY created_at DESC
	`)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch locations"})
		return
	}
	defer rows.Close()

	var locations []db.Location
	for rows.Next() {
		var loc db.Location
		err := rows.Scan(
			&loc.ID,
			&loc.Name,
			&loc.Description,
			&loc.Latitude,
			&loc.Longitude,
			&loc.CreatedAt,
			&loc.UpdatedAt,
		)
		if err != nil {
			continue
		}
		locations = append(locations, loc)
	}

	c.JSON(http.StatusOK, locations)
}

// HandleCreateLocation creates a new location
func HandleCreateLocation(c *gin.Context) {
	var location db.Location
	if err := c.ShouldBindJSON(&location); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var id int64
	err := database.QueryRow(`
		INSERT INTO locations (name, description, latitude, longitude)
		VALUES ($1, $2, $3, $4)
		RETURNING id
	`, location.Name, location.Description, location.Latitude, location.Longitude).Scan(&id)

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create location"})
		return
	}

	location.ID = id
	c.JSON(http.StatusCreated, location)
}

// HandleGetLocation returns a single location by ID
func HandleGetLocation(c *gin.Context) {
	id := c.Param("id")

	var location db.Location
	err := database.QueryRow(`
		SELECT id, name, description, latitude, longitude, created_at, updated_at
		FROM locations
		WHERE id = $1
	`, id).Scan(
		&location.ID,
		&location.Name,
		&location.Description,
		&location.Latitude,
		&location.Longitude,
		&location.CreatedAt,
		&location.UpdatedAt,
	)

	if err == sql.ErrNoRows {
		c.JSON(http.StatusNotFound, gin.H{"error": "Location not found"})
		return
	}

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch location"})
		return
	}

	c.JSON(http.StatusOK, location)
}

// HandleUpdateLocation updates an existing location
func HandleUpdateLocation(c *gin.Context) {
	id := c.Param("id")

	var location db.Location
	if err := c.ShouldBindJSON(&location); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	result, err := database.Exec(`
		UPDATE locations
		SET name = $1, description = $2, latitude = $3, longitude = $4, updated_at = CURRENT_TIMESTAMP
		WHERE id = $5
	`, location.Name, location.Description, location.Latitude, location.Longitude, id)

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to update location"})
		return
	}

	rowsAffected, _ := result.RowsAffected()
	if rowsAffected == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "Location not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Location updated successfully"})
}

// HandleDeleteLocation deletes a location
func HandleDeleteLocation(c *gin.Context) {
	id := c.Param("id")

	result, err := database.Exec("DELETE FROM locations WHERE id = $1", id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to delete location"})
		return
	}

	rowsAffected, _ := result.RowsAffected()
	if rowsAffected == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "Location not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Location deleted successfully"})
}
