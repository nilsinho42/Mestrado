package router

import (
	"github.com/gorilla/mux"
	"gorm.io/gorm"
	"your-project/controllers"
)

func SetupRouter(db *gorm.DB) *mux.Router {
	r := mux.NewRouter()

	// Initialize controllers
	locationController := controllers.NewLocationController(db)

	// Location routes
	r.HandleFunc("/api/locations", locationController.GetLocations).Methods("GET")
	r.HandleFunc("/api/locations/{id}", locationController.GetLocation).Methods("GET")
	r.HandleFunc("/api/locations", locationController.CreateLocation).Methods("POST")
	r.HandleFunc("/api/locations/{id}/metrics", locationController.UpdateLocationMetrics).Methods("PATCH")
	r.HandleFunc("/api/locations/{id}", locationController.DeleteLocation).Methods("DELETE")

	// Add middleware
	r.Use(corsMiddleware)
	r.Use(jsonMiddleware)

	return r
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func jsonMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		next.ServeHTTP(w, r)
	})
} 