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
            # Create latency_metrics table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTSlatency_metrics (
                    pk INT GENERATED ALWAYS AS IDENTITY,
                    date TIMESTAMP DEFAULT NOW(),
                    video_id TEXT NOT NULL,
                    frames INT NOT NULL,
                    latency_azure_ms INT,
                    latency_aws_ms INT,
                    latency_gcp_ms INT,
                    latency_edge_ms INT
                );
            """)
            
            # Create processing_time_metrics results table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_time_metrics (
                    pk INT GENERATED ALWAYS AS IDENTITY,
                    date TIMESTAMP DEFAULT NOW(),
                    video_id TEXT NOT NULL,
                    frames INT NOT NULL,
                    pt_azure_sec INT,
                    pt_aws_sec INT,
                    pt_gcp_sec INT,
                    pt_edge_sec INT
                );
            """)
            
            # Create fps_metrics results table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS fps_metrics (
                    pk INT GENERATED ALWAYS AS IDENTITY,
                    date TIMESTAMP DEFAULT NOW(),
                    video_id TEXT NOT NULL,
                    frames INT NOT NULL,
                    fps_azure INT,
                    fps_aws INT,
                    fps_gcp INT,
                    fps_edge INT
                );
            """)

            # Create count_vehicles results table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS count_vehicles (
                    pk INT GENERATED ALWAYS AS IDENTITY,
                    date TIMESTAMP DEFAULT NOW(),
                    video_id TEXT NOT NULL,
                    frames INT NOT NULL,
                    cv_azure INT,
                    cv_aws INT,
                    cv_gcp INT,
                    cv_edge INT,
                    cv_expected INT
                );
            """)

            # Create count_people results table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTScount_people (
                    pk INT GENERATED ALWAYS AS IDENTITY,
                    date TIMESTAMP DEFAULT NOW(),
                    video_id TEXT NOT NULL,
                    frames INT NOT NULL,
                    cp_azure INT,
                    cp_aws INT,
                    cp_gcp INT,
                    cp_edge INT,
                    cp_expected INT
                );
            """)

            # Create precision_recall results table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS precision_recall (
                    pk INT GENERATED ALWAYS AS IDENTITY,
                    date TIMESTAMP DEFAULT NOW(),
                    video_id TEXT NOT NULL,
                    frames INT NOT NULL,
                    precision_azure INT,
                    precision_aws INT,
                    precision_gcp INT,
                    precision_edge INT,
                    recall_azure INT,
                    recall_aws INT,
                    recall_gcp INT,
                    recall_edge INT
                );
            """)
            
            # Create precision_recall results table
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_metrics (
                pk INT GENERATED ALWAYS AS IDENTITY,
                date TIMESTAMP DEFAULT NOW(),
                video_id TEXT NOT NULL,
                frames INT NOT NULL,
                cost_azure INT,
                cost_aws INT,
                cost_gcp INT,
                cost_edge INT
            );
            """)
            logger.info("Created database tables")
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            self.db_enabled = False
    
    def load_data(self, table_name: str, data: Dict[str, Union[int, float, str]]) -> None:
        """
        Load data into the specified table.

        Args:
            table_name: Name of the table to insert into.
            data: Dictionary containing column-value pairs.

        Raises:
            ValueError: If table_name or data is invalid.
        """
        if not self.db_enabled or self.cursor is None:
            logger.warning("Skipping data load as database is disabled or connection failed")
            return

        if not table_name or not isinstance(data, dict) or not data:
            raise ValueError("Invalid table name or data")

        try:
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['%s'] * len(data))
            values = list(data.values())

            sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            self.cursor.execute(sql, values)
            logger.info(f"Data inserted into {table_name}")
        except Exception as e:
            logger.error(f"Error inserting data into {table_name}: {str(e)}")
    
    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed") 