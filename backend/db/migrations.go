package db

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// RunMigrations executes all SQL migration files in the migrations directory
func RunMigrations(db *sql.DB) error {
	// Read all migration files
	files, err := os.ReadDir("migrations")
	if err != nil {
		return fmt.Errorf("failed to read migrations directory: %v", err)
	}

	// Sort files to ensure they run in order
	var migrationFiles []string
	for _, file := range files {
		if strings.HasSuffix(file.Name(), ".sql") {
			migrationFiles = append(migrationFiles, file.Name())
		}
	}
	sort.Strings(migrationFiles)

	// Create migrations table if it doesn't exist
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS migrations (
			id SERIAL PRIMARY KEY,
			name VARCHAR(255) NOT NULL UNIQUE,
			applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create migrations table: %v", err)
	}

	// Run each migration
	for _, file := range migrationFiles {
		// Check if migration has been applied
		var exists bool
		err := db.QueryRow("SELECT EXISTS(SELECT 1 FROM migrations WHERE name = $1)", file).Scan(&exists)
		if err != nil {
			return fmt.Errorf("failed to check migration status: %v", err)
		}

		if exists {
			continue
		}

		// Read migration file
		content, err := os.ReadFile(filepath.Join("migrations", file))
		if err != nil {
			return fmt.Errorf("failed to read migration file %s: %v", file, err)
		}

		// Begin transaction
		tx, err := db.Begin()
		if err != nil {
			return fmt.Errorf("failed to begin transaction: %v", err)
		}

		// Execute migration
		_, err = tx.Exec(string(content))
		if err != nil {
			tx.Rollback()
			return fmt.Errorf("failed to execute migration %s: %v", file, err)
		}

		// Record migration
		_, err = tx.Exec("INSERT INTO migrations (name) VALUES ($1)", file)
		if err != nil {
			tx.Rollback()
			return fmt.Errorf("failed to record migration %s: %v", file, err)
		}

		// Commit transaction
		if err := tx.Commit(); err != nil {
			return fmt.Errorf("failed to commit migration %s: %v", file, err)
		}
	}

	return nil
}
