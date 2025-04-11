"""
Metrics collection and processing functionality.
Provides utilities for collecting, storing, and analyzing metrics.
"""

import json
import time
import logging
import os
import datetime
import sqlite3
import psutil
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import configparser

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Base class for collecting and storing metrics."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize metrics collector.
        
        Args:
            db_path: Path to SQLite database file (None for in-memory)
        """
        self.metrics = {}
        self.db_path = db_path
        
        if db_path:
            self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create metrics table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT,
                source TEXT NOT NULL,
                latency FLOAT,
                total_processing_time FLOAT,
                precision FLOAT,
                recall FLOAT,
                cost_image_processing FLOAT,
                cost_video_processing FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indices for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON metrics(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_id ON metrics(image_id)")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Initialized metrics database at {self.db_path}")
    
    def record_metric(self, metric_type: str, value: float, **kwargs):
        """
        Record a single metric.
        
        Args:
            metric_type: Type of metric
            value: Metric value
            **kwargs: Additional metadata
        """
        timestamp = time.time()
        
        if metric_type not in self.metrics:
            self.metrics[metric_type] = []
        
        self.metrics[metric_type].append({
            "value": value,
            "timestamp": timestamp,
            "metadata": kwargs
        })
        
        logger.debug(f"Recorded {metric_type}: {value} with metadata: {kwargs}")
    
    def get_metrics(self, metric_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get collected metrics.
        
        Args:
            metric_type: Optional type to filter by
            
        Returns:
            Dictionary of metrics
        """
        if metric_type:
            return {metric_type: self.metrics.get(metric_type, [])}
        return self.metrics
    
    def save_to_db(self, metrics_dict: Dict[str, Any]) -> int:
        """
        Save metrics to SQLite database.
        
        Args:
            metrics_dict: Dictionary of metrics to save
            
        Returns:
            ID of the inserted record
        """
        if not self.db_path:
            logger.warning("Cannot save to database: No database path specified")
            return -1
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert metrics into database
            cursor.execute("""
                INSERT INTO metrics 
                (image_id, source, latency, total_processing_time, 
                precision, recall, cost_image_processing, cost_video_processing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics_dict.get('image_id'),
                metrics_dict.get('source'),
                metrics_dict.get('latency'),
                metrics_dict.get('total_processing_time'),
                metrics_dict.get('precision'),
                metrics_dict.get('recall'),
                metrics_dict.get('cost_image_processing', 0.0),
                metrics_dict.get('cost_video_processing', 0.0)
            ))
            
            conn.commit()
            last_id = cursor.lastrowid
            
            logger.info(f"Saved metrics to database with ID {last_id}")
            return last_id
            
        except Exception as e:
            logger.error(f"Error saving metrics to database: {str(e)}")
            conn.rollback()
            return -1
            
        finally:
            conn.close()
    
    def get_from_db(self, **filters) -> List[Dict[str, Any]]:
        """
        Get metrics from database with filters.
        
        Args:
            **filters: Filters to apply (e.g., source='aws')
            
        Returns:
            List of metrics dictionaries
        """
        if not self.db_path:
            logger.warning("Cannot read from database: No database path specified")
            return []
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        try:
            # Build query based on filters
            query = "SELECT * FROM metrics"
            params = []
            
            if filters:
                query += " WHERE "
                conditions = []
                
                for key, value in filters.items():
                    if key in ['id', 'image_id', 'source']:
                        conditions.append(f"{key} = ?")
                        params.append(value)
                
                query += " AND ".join(conditions)
            
            # Add ordering
            query += " ORDER BY timestamp DESC"
            
            # Execute query
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            result = []
            for row in rows:
                result.append(dict(row))
            
            return result
            
        except Exception as e:
            logger.error(f"Error reading metrics from database: {str(e)}")
            return []
            
        finally:
            conn.close()


class CostCalculator:
    """Calculate cloud provider costs."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize cost calculator.
        
        Args:
            config_path: Path to configuration file with cost parameters
        """
        self.config = {}
        
        # Default cost parameters
        self.default_costs = {
            "aws": {
                "rekognition_image_analysis": 0.001,  # $ per image
                "rekognition_video_analysis": 0.10,   # $ per minute
                "fargate_vcpu_hour": 0.04048,         # $ per vCPU-hour
                "fargate_memory_gb_hour": 0.004445,   # $ per GB-hour
                "s3_storage_gb_month": 0.023,         # $ per GB-month
                "s3_get_request": 0.0004,             # $ per 1000 requests
                "s3_put_request": 0.005              # $ per 1000 requests
            },
            "azure": {
                "vision_image_analysis": 0.001,       # $ per transaction
                "vision_video_analysis": 0.15,        # $ per minute
                "container_apps_vcpu_hour": 0.036,    # $ per vCPU-hour
                "container_apps_memory_gb_hour": 0.004, # $ per GB-hour
                "blob_storage_gb_month": 0.0208,      # $ per GB-month
                "blob_get_request": 0.0004,           # $ per 10,000 requests
                "blob_put_request": 0.0054           # $ per 10,000 requests
            }
        }
        
        # Load configuration from file if provided
        if config_path:
            self._load_config(config_path)
        else:
            self.config = self.default_costs
    
    def _load_config(self, config_path: str):
        """Load configuration from file."""
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            
            for provider in ['aws', 'azure']:
                if provider in config:
                    self.config[provider] = {}
                    for key, value in config[provider].items():
                        self.config[provider][key] = float(value)
                else:
                    self.config[provider] = self.default_costs[provider]
            
            logger.info(f"Loaded cost configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading cost configuration: {str(e)}")
            self.config = self.default_costs
    
    def calculate_aws_cost(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate AWS cost based on metrics.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Dictionary with cost breakdown
        """
        aws_costs = self.config.get('aws', self.default_costs['aws'])
        
        # Initialize costs
        costs = {
            "image_analysis": 0.0,
            "video_analysis": 0.0,
            "compute": 0.0,
            "storage": 0.0,
            "total_cost": 0.0
        }
        
        # Image analysis cost (Task A)
        num_images = metrics.get('task_a_metrics', {}).get('aws', {}).get('images_processed', 0)
        if num_images > 0:
            costs["image_analysis"] = num_images * aws_costs['rekognition_image_analysis']
        
        # Video analysis cost (Task B)
        video_count = metrics.get('task_b_metrics', {}).get('aws', {}).get('videos_processed', 0)
        video_duration_minutes = metrics.get('task_b_metrics', {}).get('aws', {}).get('total_duration_minutes', 0)
        
        if video_count > 0 and video_duration_minutes > 0:
            costs["video_analysis"] = video_duration_minutes * aws_costs['rekognition_video_analysis']
        
        # Compute cost (Fargate)
        vcpu_hours = metrics.get('task_b_metrics', {}).get('aws', {}).get('vcpu_hours', 0)
        memory_gb_hours = metrics.get('task_b_metrics', {}).get('aws', {}).get('memory_gb_hours', 0)
        
        if vcpu_hours > 0 or memory_gb_hours > 0:
            vcpu_cost = vcpu_hours * aws_costs['fargate_vcpu_hour']
            memory_cost = memory_gb_hours * aws_costs['fargate_memory_gb_hour']
            costs["compute"] = vcpu_cost + memory_cost
        
        # Storage cost (S3)
        storage_gb = metrics.get('task_b_metrics', {}).get('aws', {}).get('storage_gb', 0)
        get_requests = metrics.get('task_b_metrics', {}).get('aws', {}).get('get_requests', 0)
        put_requests = metrics.get('task_b_metrics', {}).get('aws', {}).get('put_requests', 0)
        
        storage_cost = (storage_gb * aws_costs['s3_storage_gb_month'] / 30)  # Daily cost
        request_cost = (get_requests / 1000 * aws_costs['s3_get_request'] + 
                        put_requests / 1000 * aws_costs['s3_put_request'])
        
        costs["storage"] = storage_cost + request_cost
        
        # Calculate total cost
        costs["total_cost"] = sum(cost for key, cost in costs.items() if key != "total_cost")
        
        return costs
    
    def calculate_azure_cost(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate Azure cost based on metrics.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Dictionary with cost breakdown
        """
        azure_costs = self.config.get('azure', self.default_costs['azure'])
        
        # Initialize costs
        costs = {
            "image_analysis": 0.0,
            "video_analysis": 0.0,
            "compute": 0.0,
            "storage": 0.0,
            "total_cost": 0.0
        }
        
        # Image analysis cost (Task A)
        num_images = metrics.get('task_a_metrics', {}).get('azure', {}).get('images_processed', 0)
        if num_images > 0:
            costs["image_analysis"] = num_images * azure_costs['vision_image_analysis']
        
        # Video analysis cost (Task B)
        video_duration_minutes = metrics.get('task_b_metrics', {}).get('azure', {}).get('total_duration_minutes', 0)
        
        if video_duration_minutes > 0:
            costs["video_analysis"] = video_duration_minutes * azure_costs['vision_video_analysis']
        
        # Compute cost (Container Apps)
        vcpu_hours = metrics.get('task_b_metrics', {}).get('azure', {}).get('vcpu_hours', 0)
        memory_gb_hours = metrics.get('task_b_metrics', {}).get('azure', {}).get('memory_gb_hours', 0)
        
        if vcpu_hours > 0 or memory_gb_hours > 0:
            vcpu_cost = vcpu_hours * azure_costs['container_apps_vcpu_hour']
            memory_cost = memory_gb_hours * azure_costs['container_apps_memory_gb_hour']
            costs["compute"] = vcpu_cost + memory_cost
        
        # Storage cost (Blob Storage)
        storage_gb = metrics.get('task_b_metrics', {}).get('azure', {}).get('storage_gb', 0)
        get_requests = metrics.get('task_b_metrics', {}).get('azure', {}).get('get_requests', 0)
        put_requests = metrics.get('task_b_metrics', {}).get('azure', {}).get('put_requests', 0)
        
        storage_cost = (storage_gb * azure_costs['blob_storage_gb_month'] / 30)  # Daily cost
        request_cost = (get_requests / 10000 * azure_costs['blob_get_request'] + 
                        put_requests / 10000 * azure_costs['blob_put_request'])
        
        costs["storage"] = storage_cost + request_cost
        
        # Calculate total cost
        costs["total_cost"] = sum(cost for key, cost in costs.items() if key != "total_cost")
        
        return costs
    
    def calculate_local_cost(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate local processing cost based on metrics.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Dictionary with cost breakdown (mostly zeros, but consistent format)
        """
        # Initialize costs
        costs = {
            "image_analysis": 0.0,
            "video_analysis": 0.0,
            "compute": 0.0,
            "storage": 0.0,
            "total_cost": 0.0
        }
        
        # Local compute cost (approximation based on CPU/GPU usage)
        processing_time_seconds = metrics.get('task_b_metrics', {}).get('local', {}).get('total_processing_time', 0)
        cpu_percent = metrics.get('task_b_metrics', {}).get('local', {}).get('cpu_percent', 0)
        memory_mb = metrics.get('task_b_metrics', {}).get('local', {}).get('memory_mb', 0)
        
        # Very rough approximation - adjust as needed for your hardware
        electricity_cost_per_kwh = 0.15  # $0.15 per kWh
        system_power_watts = 200  # Assuming a 200W system
        
        # Cost = power (kW) * time (h) * cost per kWh
        if processing_time_seconds > 0 and cpu_percent > 0:
            hours = processing_time_seconds / 3600
            power_usage = system_power_watts * (cpu_percent / 100)  # Adjust power based on CPU usage
            costs["compute"] = (power_usage / 1000) * hours * electricity_cost_per_kwh
        
        # Storage cost (negligible for local)
        storage_gb = metrics.get('task_b_metrics', {}).get('local', {}).get('storage_gb', 0)
        # Assuming $0.05 per GB-month for local storage
        costs["storage"] = storage_gb * 0.05 / 30  # Daily cost
        
        # Calculate total cost
        costs["total_cost"] = sum(cost for key, cost in costs.items() if key != "total_cost")
        
        return costs


# Factory function to create cost calculator
def create_cost_calculator(config_path: Optional[str] = None) -> CostCalculator:
    """
    Create a cost calculator.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        CostCalculator instance
    """
    return CostCalculator(config_path)
