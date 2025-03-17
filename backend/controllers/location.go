package controllers

import (
	"encoding/json"
	"net/http"

	"github.com/gorilla/mux"
	"gorm.io/gorm"
)

type Location struct {
	ID        string  `json:"id" gorm:"primaryKey"`
	Name      string  `json:"name"`
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
	Metrics   Metrics `json:"metrics" gorm:"embedded"`
}

type Metrics struct {
	PersonCount  int     `json:"personCount"`
	VehicleCount int     `json:"vehicleCount"`
	AvgFlow      float64 `json:"avgFlow"`
	LastUpdated  string  `json:"lastUpdated"`
}

type LocationController struct {
	db *gorm.DB
}

func NewLocationController(db *gorm.DB) *LocationController {
	return &LocationController{db: db}
}

func (c *LocationController) GetLocations(w http.ResponseWriter, r *http.Request) {
	var locations []Location
	result := c.db.Find(&locations)
	if result.Error != nil {
		http.Error(w, result.Error.Error(), http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(locations)
}

func (c *LocationController) GetLocation(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	var location Location
	result := c.db.First(&location, "id = ?", id)
	if result.Error != nil {
		if result.Error == gorm.ErrRecordNotFound {
			http.Error(w, "Location not found", http.StatusNotFound)
		} else {
			http.Error(w, result.Error.Error(), http.StatusInternalServerError)
		}
		return
	}

	json.NewEncoder(w).Encode(location)
}

func (c *LocationController) CreateLocation(w http.ResponseWriter, r *http.Request) {
	var location Location
	if err := json.NewDecoder(r.Body).Decode(&location); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	result := c.db.Create(&location)
	if result.Error != nil {
		http.Error(w, result.Error.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(location)
}

func (c *LocationController) UpdateLocationMetrics(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	var metrics Metrics
	if err := json.NewDecoder(r.Body).Decode(&metrics); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var location Location
	result := c.db.First(&location, "id = ?", id)
	if result.Error != nil {
		if result.Error == gorm.ErrRecordNotFound {
			http.Error(w, "Location not found", http.StatusNotFound)
		} else {
			http.Error(w, result.Error.Error(), http.StatusInternalServerError)
		}
		return
	}

	location.Metrics = metrics
	result = c.db.Save(&location)
	if result.Error != nil {
		http.Error(w, result.Error.Error(), http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(location)
}

func (c *LocationController) DeleteLocation(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	result := c.db.Delete(&Location{}, "id = ?", id)
	if result.Error != nil {
		http.Error(w, result.Error.Error(), http.StatusInternalServerError)
		return
	}

	if result.RowsAffected == 0 {
		http.Error(w, "Location not found", http.StatusNotFound)
		return
	}

	w.WriteHeader(http.StatusNoContent)
} 