import os
import configparser
import boto3
import requests
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List, Optional, Union, Tuple

class CostCalculator:
    def __init__(self, config_path: str = 'cost_config.ini'):
        """Initialize cost calculator with the given config path."""
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path))
        
        # Initialize AWS CloudWatch client for metrics
        if 'AWS_ACCESS_KEY_ID' in os.environ:
            self.cloudwatch = boto3.client('cloudwatch')
            self.logs = boto3.client('logs')
        else:
            self.cloudwatch = None
            self.logs = None
        
        # Azure Monitor API endpoint for logs
        self.azure_monitor_endpoint = 'https://management.azure.com'
        self.azure_monitor_token = None
        
    def get_aws_cost_for_task_a(self, image_count: int) -> float:
        """Calculate AWS cost for Task A (image analysis)."""
        cost_per_image = float(self.config.get('AWS', 'taskA_object_detection_cost_per_image', fallback='0'))
        return image_count * cost_per_image
    
    def get_azure_cost_for_task_a(self, image_count: int) -> float:
        """Calculate Azure cost for Task A (image analysis)."""
        cost_per_image = float(self.config.get('Azure', 'taskA_object_detection_cost_per_image', fallback='0'))
        return image_count * cost_per_image
    
    def get_local_cost_for_task_a(self, image_count: int) -> float:
        """Calculate local processing cost for Task A."""
        # Local processing is free
        return 0.0
    
    def get_aws_cost_for_task_b(self, 
                               video_size_gb: float, 
                               frame_count: int, 
                               lambda_execution_time_seconds: float,
                               memory_mb: int = 1024) -> Dict[str, float]:
        """Calculate AWS cost for Task B (video processing).
        
        Args:
            video_size_gb: Size of the video in GB
            frame_count: Number of frames processed
            lambda_execution_time_seconds: Total Lambda execution time in seconds
            memory_mb: Lambda memory allocation in MB
            
        Returns:
            Dictionary of cost components and total
        """
        storage_cost_per_gb = float(self.config.get('AWS', 'taskB_video_storage_cost_per_gb', fallback='0'))
        object_detection_cost_per_image = float(self.config.get('AWS', 'taskB_object_detection_cost_per_image', fallback='0'))
        lambda_request_cost = float(self.config.get('AWS', 'taskB_lambda_request_cost', fallback='0'))
        lambda_cpu_cost_per_second = float(self.config.get('AWS', 'taskB_lambda_cpu_cost_per_second', fallback='0'))
        lambda_memory_cost_per_second = float(self.config.get('AWS', 'taskB_lambda_memory_cost_per_second', fallback='0'))
        
        # Calculate individual cost components
        storage_cost = video_size_gb * storage_cost_per_gb
        detection_cost = frame_count * object_detection_cost_per_image
        lambda_cost = (lambda_execution_time_seconds * lambda_cpu_cost_per_second) + \
                      (lambda_execution_time_seconds * lambda_memory_cost_per_second * (memory_mb / 1024)) + \
                      lambda_request_cost
        
        # Return the total cost and components
        return {
            'storage_cost': storage_cost,
            'detection_cost': detection_cost,
            'lambda_cost': lambda_cost,
            'total_cost': storage_cost + detection_cost + lambda_cost
        }
    
    def get_azure_cost_for_task_b(self,
                                video_size_gb: float,
                                frame_count: int,
                                container_execution_time_seconds: float,
                                cpu_cores: float = 1.0,
                                memory_gb: float = 2.0) -> Dict[str, float]:
        """Calculate Azure cost for Task B (video processing).
        
        Args:
            video_size_gb: Size of the video in GB
            frame_count: Number of frames processed
            container_execution_time_seconds: Total Container execution time in seconds
            cpu_cores: Number of CPU cores allocated
            memory_gb: Amount of memory allocated in GB
            
        Returns:
            Dictionary of cost components and total
        """
        storage_cost_per_gb = float(self.config.get('Azure', 'taskB_video_storage_cost_per_gb', fallback='0'))
        object_detection_cost_per_image = float(self.config.get('Azure', 'taskB_object_detection_cost_per_image', fallback='0'))
        container_cpu_cost_per_second = float(self.config.get('Azure', 'taskB_container_cpu_cost_per_second', fallback='0'))
        container_memory_cost_per_second = float(self.config.get('Azure', 'taskB_container_memory_cost_per_second', fallback='0'))
        
        # Calculate individual cost components
        storage_cost = video_size_gb * storage_cost_per_gb
        detection_cost = frame_count * object_detection_cost_per_image
        container_cost = (container_execution_time_seconds * container_cpu_cost_per_second * cpu_cores) + \
                         (container_execution_time_seconds * container_memory_cost_per_second * memory_gb)
        
        # Return the total cost and components
        return {
            'storage_cost': storage_cost,
            'detection_cost': detection_cost,
            'container_cost': container_cost,
            'total_cost': storage_cost + detection_cost + container_cost
        }
    
    def get_local_cost_for_task_b(self, execution_time_seconds: float) -> Dict[str, float]:
        """Calculate local processing cost for Task B."""
        # Local processing is essentially free
        processing_cost_per_second = float(self.config.get('Local', 'taskB_processing_cost_per_second', fallback='0'))
        total_cost = execution_time_seconds * processing_cost_per_second
        
        return {
            'processing_cost': total_cost,
            'total_cost': total_cost
        }
    
    def get_aws_cloudwatch_metrics(self, 
                                  log_group: str,
                                  log_stream: str,
                                  start_time: datetime,
                                  end_time: datetime) -> Dict[str, Any]:
        """Get CPU/Memory usage from AWS CloudWatch logs."""
        if not self.cloudwatch:
            return {}
        
        try:
            # Get CloudWatch Logs Insights query results
            query = f"""
            fields @timestamp, @message
            | filter @logStream = '{log_stream}'
            | parse @message "*CPU: * %" as _, cpu_percent
            | parse @message "*Memory: * MB" as _, memory_mb
            | stats avg(cpu_percent) as avg_cpu, 
                    max(cpu_percent) as max_cpu, 
                    avg(memory_mb) as avg_memory, 
                    max(memory_mb) as max_memory
            """
            
            response = self.logs.start_query(
                logGroupName=log_group,
                startTime=int(start_time.timestamp()),
                endTime=int(end_time.timestamp()),
                queryString=query
            )
            
            query_id = response['queryId']
            result = None
            
            # Wait for query to complete
            while result is None or result['status'] == 'Running':
                result = self.logs.get_query_results(queryId=query_id)
                if result['status'] == 'Complete':
                    break
                time.sleep(1)
            
            # Process and return results if available
            if result['status'] == 'Complete' and result.get('results'):
                metrics = {}
                
                for field in result['results'][0]:
                    metrics[field['field']] = float(field['value'])
                
                return metrics
            
            return {}
            
        except Exception as e:
            print(f"Error getting AWS CloudWatch metrics: {e}")
            return {}
    
    def get_azure_monitor_metrics(self,
                                resource_group: str,
                                app_name: str,
                                start_time: datetime,
                                end_time: datetime) -> Dict[str, Any]:
        """Get CPU/Memory usage from Azure Monitor logs."""
        if not self._authenticate_azure():
            return {}
        
        try:
            # Create a Kusto query for Container App metrics
            query = f"""
            ContainerAppConsoleLogs_CL
            | where ResourceGroup == '{resource_group}'
            | where AppName_s == '{app_name}'
            | where TimeGenerated between(datetime({start_time.isoformat()}) .. datetime({end_time.isoformat()}))
            | where Message has "CPU:" or Message has "Memory:"
            | extend CPU = extract("CPU: ([0-9.]+)", 1, Message)
            | extend Memory = extract("Memory: ([0-9.]+)", 1, Message)
            | project TimeGenerated, CPU, Memory
            | summarize avg(todouble(CPU)) as avg_cpu, max(todouble(CPU)) as max_cpu, 
                      avg(todouble(Memory)) as avg_memory, max(todouble(Memory)) as max_memory
            """
            
            # Make API request to Azure Monitor
            url = f"{self.azure_monitor_endpoint}/subscriptions/{os.getenv('AZURE_SUBSCRIPTION_ID')}/resourceGroups/{resource_group}/providers/Microsoft.OperationalInsights/workspaces/{os.getenv('AZURE_LOG_ANALYTICS_WORKSPACE')}/api/query"
            
            headers = {
                'Authorization': f'Bearer {self.azure_monitor_token}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'query': query,
                'timespan': f"{start_time.isoformat()}/{end_time.isoformat()}"
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                if 'tables' in result and len(result['tables']) > 0:
                    # Extract metrics from result
                    metrics = {}
                    
                    # Process the tables data to extract metrics
                    for column_idx, column in enumerate(result['tables'][0]['columns']):
                        if column['name'] in ['avg_cpu', 'max_cpu', 'avg_memory', 'max_memory']:
                            metrics[column['name']] = float(result['tables'][0]['rows'][0][column_idx])
                    
                    return metrics
            
            return {}
            
        except Exception as e:
            print(f"Error getting Azure Monitor metrics: {e}")
            return {}
    
    def _authenticate_azure(self) -> bool:
        """Authenticate with Azure to get access token."""
        if self.azure_monitor_token:
            return True
            
        try:
            # Get Azure access token using client credentials
            url = f"https://login.microsoftonline.com/{os.getenv('AZURE_TENANT_ID')}/oauth2/token"
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            data = {
                'grant_type': 'client_credentials',
                'client_id': os.getenv('AZURE_CLIENT_ID'),
                'client_secret': os.getenv('AZURE_CLIENT_SECRET'),
                'resource': 'https://management.azure.com/'
            }
            
            response = requests.post(url, headers=headers, data=data)
            
            if response.status_code == 200:
                result = response.json()
                self.azure_monitor_token = result.get('access_token')
                return True
                
            return False
            
        except Exception as e:
            print(f"Error authenticating with Azure: {e}")
            return False 