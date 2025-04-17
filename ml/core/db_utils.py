import psycopg2
import os
from dotenv import load_dotenv
import json
from typing import Dict, Any, List, Union
import logging

# Load environment variables from root directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

logger = logging.getLogger(__name__)

class Database:
    """Database utility for storing metrics and tracking results."""
    
    def __init__(self, host=None, port=None, dbname=None, user=None, password=None,
                db_url=None, disable_db=False):
        """
        Initialize database connection.
        
        Args:
            host: Database host
            port: Database port
            dbname: Database name
            user: Database user
            password: Database password
            db_url: Database URL (alternative to separate parameters)
            disable_db: If True, disable database functionality
        """
        self.conn = None
        self.cursor = None
        self.db_enabled = not disable_db
        
        if disable_db:
            logger.info("Database functionality is disabled")
            return
        
        # Try to connect to database
        try:
            if db_url:
                self.conn = psycopg2.connect(db_url)
            else:
                # Get database connection details from environment variables if not provided
                host = host or os.getenv("DB_HOST", "postgres")
                port = port or os.getenv("DB_PORT", "5432")
                dbname = dbname or os.getenv("DB_NAME", "postgres")
                user = user or os.getenv("DB_USER", "postgres")
                password = password or os.getenv("DB_PASSWORD", "postgres")
                
                self.conn = psycopg2.connect(
                    host=host,
                    port=port,
                    dbname=dbname,
                    user=user,
                    password=password
                )
            
            self.conn.autocommit = True
            self.cursor = self.conn.cursor()
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            logger.warning("Database functionality will be disabled")
            self.db_enabled = False
    
    def create_tables(self):
        """Create required tables if they don't exist."""
        if not self.db_enabled or self.cursor is None:
            logger.warning("Skipping table creation as database is disabled or connection failed")
            return
            
        try:
            # Create metrics table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id SERIAL PRIMARY KEY,
                    image_id VARCHAR(255),
                    source VARCHAR(50),
                    latency FLOAT,
                    total_processing_time FLOAT,
                    cost_image_processing FLOAT,
                    cost_video_processing FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create tracking results table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS tracking_results (
                    id SERIAL PRIMARY KEY,
                    video_id VARCHAR(255),
                    source VARCHAR(50),
                    processing_time FLOAT,
                    people_tracked INTEGER,
                    vehicles_tracked INTEGER,
                    tracking_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            logger.info("Created database tables")
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            self.db_enabled = False
    
    def save_metrics(self, metrics_data):
        """
        Save metrics to database.
        
        Args:
            metrics_data: Dictionary with metrics data
        """
        if not self.db_enabled or self.cursor is None:
            logger.debug("Skipping metrics storage as database is disabled")
            return
            
        try:
            # Extract fields
            image_id = metrics_data.get('image_id', '')
            source = metrics_data.get('source', 'unknown')
            latency = metrics_data.get('latency', 0)
            total_processing_time = metrics_data.get('total_processing_time', 0)
            cost_image_processing = metrics_data.get('cost_image_processing', 0)
            cost_video_processing = metrics_data.get('cost_video_processing', 0)
            
            # Insert into database
            self.cursor.execute("""
                INSERT INTO metrics (
                    image_id, source, latency, total_processing_time, 
                    cost_image_processing, cost_video_processing
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                image_id, source, latency, total_processing_time,
                cost_image_processing, cost_video_processing
            ))
            
            logger.debug(f"Saved metrics for {image_id}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def save_tracking_results(self, tracking_data):
        """
        Save tracking results to database.
        
        Args:
            tracking_data: Dictionary with tracking results
        """
        if not self.db_enabled or self.cursor is None:
            logger.debug("Skipping tracking results storage as database is disabled")
            return
            
        try:
            # Extract fields
            video_id = tracking_data.get('video_id', '')
            source = tracking_data.get('source', 'unknown')
            processing_time = tracking_data.get('processing_time', 0)
            people_tracked = tracking_data.get('people_tracked', 0)
            vehicles_tracked = tracking_data.get('vehicles_tracked', 0)
            
            # Convert tracking data to JSON string
            tracking_json = json.dumps(tracking_data.get('tracking_data', {}))
            
            # Insert into database
            self.cursor.execute("""
                INSERT INTO tracking_results (
                    video_id, source, processing_time, people_tracked,
                    vehicles_tracked, tracking_data
                ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            """, (
                video_id, source, processing_time, people_tracked,
                vehicles_tracked, tracking_json
            ))
            
            logger.debug(f"Saved tracking results for {video_id}")
        except Exception as e:
            logger.error(f"Error saving tracking results: {str(e)}")
    
    def get_metrics(self, source=None, limit=10):
        """
        Get metrics from database.
        
        Args:
            source: Filter by source
            limit: Maximum number of results
            
        Returns:
            List of metrics dictionaries
        """
        if not self.db_enabled or self.cursor is None:
            logger.debug("Cannot get metrics as database is disabled")
            return []
            
        try:
            if source:
                self.cursor.execute("""
                    SELECT * FROM metrics WHERE source = %s
                    ORDER BY created_at DESC LIMIT %s
                """, (source, limit))
            else:
                self.cursor.execute("""
                    SELECT * FROM metrics ORDER BY created_at DESC LIMIT %s
                """, (limit,))
            
            columns = [desc[0] for desc in self.cursor.description]
            results = []
            
            for row in self.cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return []
    
    def get_tracking_results(self, source=None, limit=10):
        """
        Get tracking results from database.
        
        Args:
            source: Filter by source
            limit: Maximum number of results
            
        Returns:
            List of tracking results dictionaries
        """
        if not self.db_enabled or self.cursor is None:
            logger.debug("Cannot get tracking results as database is disabled")
            return []
            
        try:
            if source:
                self.cursor.execute("""
                    SELECT * FROM tracking_results WHERE source = %s
                    ORDER BY created_at DESC LIMIT %s
                """, (source, limit))
            else:
                self.cursor.execute("""
                    SELECT * FROM tracking_results ORDER BY created_at DESC LIMIT %s
                """, (limit,))
            
            columns = [desc[0] for desc in self.cursor.description]
            results = []
            
            for row in self.cursor.fetchall():
                # Convert JSONB to dict
                row_dict = dict(zip(columns, row))
                row_dict['tracking_data'] = row_dict['tracking_data']
                results.append(row_dict)
            
            return results
        except Exception as e:
            logger.error(f"Error getting tracking results: {str(e)}")
            return []
    
    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed") 