import psycopg2
import os
from dotenv import load_dotenv
import json
from typing import Dict, Any, List, Union

# Load environment variables
load_dotenv()

class Database:
    def __init__(self):
        """Initialize database connection using environment variables."""
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'database': os.getenv('DB_NAME', 'ml_comparison')
        }
        
        self.connection = None
        self.cursor = None
    
    def connect(self):
        """Establish connection to the database."""
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            self.cursor = self.connection.cursor()
            return True
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
    
    def create_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            self.connect()
            
            # Create metrics table for image and video processing
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id SERIAL PRIMARY KEY,
                    image_id VARCHAR(255),
                    source VARCHAR(50) NOT NULL,
                    latency FLOAT,
                    total_processing_time FLOAT,
                    precision FLOAT,
                    recall FLOAT,
                    cost_image_processing FLOAT,
                    cost_video_processing FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create detection results table for Task A
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_results (
                    id SERIAL PRIMARY KEY,
                    image_id VARCHAR(255) NOT NULL,
                    source VARCHAR(50) NOT NULL,
                    latency FLOAT NOT NULL,
                    people_count INTEGER NOT NULL,
                    vehicles_count INTEGER NOT NULL,
                    detection_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create tracking results table for Task B
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS tracking_results (
                    id SERIAL PRIMARY KEY,
                    video_id VARCHAR(255) NOT NULL,
                    source VARCHAR(50) NOT NULL,
                    processing_time FLOAT NOT NULL,
                    people_tracked INTEGER NOT NULL,
                    vehicles_tracked INTEGER NOT NULL,
                    tracking_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.connection.commit()
            return True
            
        except psycopg2.Error as e:
            print(f"Error creating tables: {e}")
            return False
        finally:
            self.disconnect()
    
    def save_detection_results(self, results: Dict[str, Any]):
        """Save Task A detection results to the database."""
        try:
            self.connect()
            
            self.cursor.execute("""
                INSERT INTO detection_results 
                (image_id, source, latency, people_count, vehicles_count, detection_data) 
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                results['image_id'],
                results['source'],
                results['latency'],
                results['people_count'],
                results['vehicles_count'],
                json.dumps(results.get('detection_data', {}))
            ))
            
            result_id = self.cursor.fetchone()[0]
            self.connection.commit()
            return result_id
            
        except psycopg2.Error as e:
            print(f"Error saving detection results: {e}")
            return None
        finally:
            self.disconnect()
    
    def save_tracking_results(self, results: Dict[str, Any]):
        """Save Task B tracking results to the database."""
        try:
            self.connect()
            
            self.cursor.execute("""
                INSERT INTO tracking_results 
                (video_id, source, processing_time, people_tracked, vehicles_tracked, tracking_data) 
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                results['video_id'],
                results['source'],
                results['processing_time'],
                results['people_tracked'],
                results['vehicles_tracked'],
                json.dumps(results.get('tracking_data', {}))
            ))
            
            result_id = self.cursor.fetchone()[0]
            self.connection.commit()
            return result_id
            
        except psycopg2.Error as e:
            print(f"Error saving tracking results: {e}")
            return None
        finally:
            self.disconnect()
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """Save combined metrics for Task A and Task B."""
        try:
            self.connect()
            
            self.cursor.execute("""
                INSERT INTO metrics 
                (image_id, source, latency, total_processing_time, precision, recall, cost_image_processing, cost_video_processing) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                metrics.get('image_id', None),
                metrics['source'],
                metrics.get('latency', None),
                metrics.get('total_processing_time', None),
                metrics.get('precision', None),
                metrics.get('recall', None),
                metrics.get('cost_image_processing', 0.0),
                metrics.get('cost_video_processing', 0.0)
            ))
            
            metric_id = self.cursor.fetchone()[0]
            self.connection.commit()
            return metric_id
            
        except psycopg2.Error as e:
            print(f"Error saving metrics: {e}")
            return None
        finally:
            self.disconnect()
    
    def get_detection_results(self, image_id: str = None, source: str = None):
        """Retrieve detection results from the database."""
        try:
            self.connect()
            
            query = "SELECT * FROM detection_results WHERE 1=1"
            params = []
            
            if image_id:
                query += " AND image_id = %s"
                params.append(image_id)
            
            if source:
                query += " AND source = %s"
                params.append(source)
                
            query += " ORDER BY created_at DESC"
            
            self.cursor.execute(query, params)
            columns = [desc[0] for desc in self.cursor.description]
            results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            
            return results
            
        except psycopg2.Error as e:
            print(f"Error retrieving detection results: {e}")
            return []
        finally:
            self.disconnect()
    
    def get_tracking_results(self, video_id: str = None, source: str = None):
        """Retrieve tracking results from the database."""
        try:
            self.connect()
            
            query = "SELECT * FROM tracking_results WHERE 1=1"
            params = []
            
            if video_id:
                query += " AND video_id = %s"
                params.append(video_id)
            
            if source:
                query += " AND source = %s"
                params.append(source)
                
            query += " ORDER BY created_at DESC"
            
            self.cursor.execute(query, params)
            columns = [desc[0] for desc in self.cursor.description]
            results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            
            return results
            
        except psycopg2.Error as e:
            print(f"Error retrieving tracking results: {e}")
            return []
        finally:
            self.disconnect()
    
    def get_metrics(self, image_id: str = None, source: str = None):
        """Retrieve metrics from the database."""
        try:
            self.connect()
            
            query = "SELECT * FROM metrics WHERE 1=1"
            params = []
            
            if image_id:
                query += " AND image_id = %s"
                params.append(image_id)
            
            if source:
                query += " AND source = %s"
                params.append(source)
                
            query += " ORDER BY created_at DESC"
            
            self.cursor.execute(query, params)
            columns = [desc[0] for desc in self.cursor.description]
            results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            
            return results
            
        except psycopg2.Error as e:
            print(f"Error retrieving metrics: {e}")
            return []
        finally:
            self.disconnect() 