"""
Cost calculation module for cloud and edge processing.
Uses cloud provider APIs to get actual resource metrics for precise cost calculation.
"""

import logging
import os
import ast
import datetime
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import boto3
from azure.identity import DefaultAzureCredential
from azure.monitor.query import MetricsQueryClient

logger = logging.getLogger(__name__)

def get_aws_metrics(start_time: datetime,
                    end_time: datetime) -> Dict[str, float]:
    """
    Retrieve AWS ECS/Fargate CPU and memory metrics from CloudWatch.
    Returns vCPU-seconds and memory-GB-seconds.
    """
    # Standard Fargate config
    vcpu_count = 2
    memory_gb = 4

    region = "us-east-2"
    cluster_name = "deepsort-cluster"
    service_name = "meu-servico-mestrado"

    cloudwatch = boto3.client("cloudwatch", region_name=region)

    start_time = start_time - timedelta(hours=1)
    total_seconds = (end_time - start_time).total_seconds()

    response = cloudwatch.get_metric_statistics(
        Namespace="AWS/ECS",
        MetricName="CPUUtilization",
        Dimensions=[
            {"Name": "ClusterName", "Value": cluster_name},
            {"Name": "ServiceName", "Value": service_name},
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=300,
        Statistics=["Average"],
    )

    datapoints = response.get("Datapoints", [])
    if datapoints:
        print(f"CPU Utilization Retrieved {len(datapoints)} datapoints:")
        avg_cpu = sum(dp["Average"] for dp in datapoints) / len(datapoints)
        vcpu_seconds = (avg_cpu / 100.0) * vcpu_count * total_seconds
    else:
        vcpu_seconds = 0

    response = cloudwatch.get_metric_statistics(
        Namespace="AWS/ECS",
        MetricName="MemoryUtilization",
        Dimensions=[
            {"Name": "ClusterName", "Value": cluster_name},
            {"Name": "ServiceName", "Value": service_name},
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=300,
        Statistics=["Average"],
    )
    datapoints = response.get("Datapoints", [])

    if datapoints:
        print(f"Memory Utilization Retrieved {len(datapoints)} datapoints:")
        avg_mem = sum(dp["Average"] for dp in datapoints) / len(datapoints)
        memory_gb_seconds = (avg_mem / 100.0) * memory_gb * total_seconds
    else:
        memory_gb_seconds = 0

    return {
        "vcpu_seconds": vcpu_seconds,
        "memory_gb_seconds": memory_gb_seconds
    }

def get_azure_metrics(start_time: datetime,
                      end_time: datetime) -> Dict[str, float]:
    
    start_time = start_time - timedelta(hours=1)

    vcpu_count = 2
    memory_gb = 4
    duration_seconds = (end_time - start_time).total_seconds()

    credential = DefaultAzureCredential()
    metrics_client = MetricsQueryClient(credential)
    import os 

    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    app_name = os.getenv("AZURE_CONTAINER_APP")
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")

    resource_id = (
        f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/"
        f"providers/Microsoft.App/containerApps/{app_name}"
    )

    try:
        cpu_metrics = metrics_client.query_resource(
            resource_id,
            ["CpuPercentage"],
            timespan=(start_time, end_time),
            granularity=timedelta(minutes=1),
            aggregations=["Average"]
        )
    except Exception as e:
        raise ValueError(f"Could not query CPU metrics: {e}")
    
    cpu_values = []
    for metric in cpu_metrics.metrics[0].timeseries:
        for data_point in metric.data:
            # Extract the 'average' value for each data point
            if data_point.average is not None:
                # Calculate vCPU-seconds (assuming average is in cores per second)
                # You may need to adjust this calculation depending on the unit of the 'average'
                cpu_values.append(data_point.average)  # Add average to vCPU-seconds

    if not cpu_values:
        raise ValueError("No valid CPU datapoints found")

    avg_cpu_percent = sum(cpu_values) / len(cpu_values)

    try:
        memory_metrics = metrics_client.query_resource(
            resource_id,
            ["MemoryPercentage"],
            timespan=(start_time, end_time),
            granularity=timedelta(minutes=1),
            aggregations=["Average"]
        )
    except Exception as e:
        raise ValueError(f"Could not query memory metrics: {e}")

    memory_values = []
    for metric in memory_metrics.metrics[0].timeseries:
        for data_point in metric.data:
            # Extract the 'average' value for each data point
            if data_point.average is not None:
                # Calculate vCPU-seconds (assuming average is in cores per second)
                # You may need to adjust this calculation depending on the unit of the 'average'
                memory_values.append(data_point.average)

    if not memory_values:
        raise ValueError("No valid memory datapoints found")

    avg_memory_percent = sum(memory_values) / len(memory_values)

    vcpu_seconds = (avg_cpu_percent / 100) * vcpu_count * duration_seconds
    memory_gb_seconds = (avg_memory_percent / 100) * memory_gb * duration_seconds

    logger.info(f"Avg CPU: {avg_cpu_percent:.2f}%, Avg Memory: {avg_memory_percent:.2f}%")
    logger.info(f"vCPU-seconds: {vcpu_seconds:.2f}, Memory-GB-seconds: {memory_gb_seconds:.2f}")

    return {
        'vcpu_seconds': vcpu_seconds,
        'memory_gb_seconds': memory_gb_seconds
    }

def get_gcp_metrics(project_id: str, region: str, service_name: str, 
                   start_time: datetime, end_time: datetime) -> Dict[str, float]:
    """
    Retrieve Google Cloud Run CPU and memory metrics from Cloud Monitoring.
    
    Args:
        project_id: Google Cloud project ID
        region: GCP region (e.g., 'us-central1')
        service_name: Cloud Run service name
        start_time: Start time for metrics collection
        end_time: End time for metrics collection
        
    Returns:
        Dictionary with vcpu_seconds and memory_gb_seconds
    """
    from google.cloud import monitoring_v3
    from google.protobuf.timestamp_pb2 import Timestamp
    from google.protobuf.duration_pb2 import Duration

    vcpu_count = 2
    memory_gb = 4
    duration_seconds = (end_time - start_time).total_seconds()
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    start_time = start_time - timedelta(hours=1)
    start_time_pb = Timestamp()
    start_time_pb.FromDatetime(start_time)
    end_time_pb = Timestamp()
    end_time_pb.FromDatetime(end_time)

    interval = monitoring_v3.TimeInterval(start_time=start_time_pb, end_time=end_time_pb)

    aggregation = monitoring_v3.Aggregation(
        alignment_period=Duration(seconds=60),
        per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_PERCENTILE_99,
        cross_series_reducer=monitoring_v3.Aggregation.Reducer.REDUCE_MEAN
    )


    # CPU filter (plural!)
    cpu_filter = (
        f'resource.type="cloud_run_revision" AND '
        f'resource.labels.service_name="{service_name}" AND '
        f'resource.labels.location="{region}" AND '
        f'metric.type="run.googleapis.com/container/cpu/utilizations"'
    )

    cpu_values = []
    for ts in client.list_time_series(
        request={
            "name": project_name,
            "filter": cpu_filter,
            "interval": interval,
            "aggregation": aggregation,
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        }
    ):
        for point in ts.points:
            cpu_values.append(point.value.double_value)

    if not cpu_values:
        raise ValueError(f"No CPU metrics found for Cloud Run service {service_name}")

    avg_cpu = sum(cpu_values) / len(cpu_values)

    # Memory filter (plural!)
    memory_filter = (
        f'resource.type="cloud_run_revision" AND '
        f'resource.labels.service_name="{service_name}" AND '
        f'resource.labels.location="{region}" AND '
        f'metric.type="run.googleapis.com/container/memory/utilizations"'
    )

    memory_values = []
    for ts in client.list_time_series(
        request={
            "name": project_name,
            "filter": memory_filter,
            "interval": interval,
            "aggregation": aggregation,
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        }
    ):
        for point in ts.points:
            memory_values.append(point.value.double_value)

    if not memory_values:
        raise ValueError(f"No memory metrics found for Cloud Run service {service_name}")

    avg_memory = sum(memory_values) / len(memory_values)

    vcpu_seconds = avg_cpu * vcpu_count * duration_seconds
    memory_gb_seconds = avg_memory * memory_gb * duration_seconds

    return {
        'vcpu_seconds': vcpu_seconds,
        'memory_gb_seconds': memory_gb_seconds
    }

class CostCalculator:
    """
    Calculate costs for different cloud providers and edge processing.
    Uses cloud metrics for cloud providers and processing time for edge.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the cost calculator with configuration from file.
        
        Args:
            config_path: Path to configuration file with cloud costs
        """
        self.config = self._load_config(config_path)
        if not self.config:
            logger.error(f"Failed to load configuration from {config_path}")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load cost configuration from file, parsing the Python dictionary format.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary with cost configuration
        """
        config = {}
        try:
            if not os.path.exists(config_path):
                logger.error(f"Configuration file not found: {config_path}")
                return config
                
            with open(config_path, 'r') as file:
                content = file.read()
                
                # Find the dictionary in the file content
                start_idx = content.find('{')
                if start_idx != -1:
                    # Parse the dictionary using ast.literal_eval (safer than eval)
                    config = ast.literal_eval(content[start_idx:])
                    logger.info(f"Successfully loaded cost configuration from {config_path}")
                else:
                    logger.error(f"No dictionary found in configuration file: {config_path}")
                    
            return config
        except Exception as e:
            logger.error(f"Error loading cost configuration: {str(e)}")
            return {}
            
    def calculate_cost(self, provider: str, **kwargs) -> float:
        """
        Calculate cost for a specific provider.
        
        Args:
            provider: Provider name ('aws', 'azure', 'gcp', 'edge')
            **kwargs: Additional parameters depending on provider:
                - frame_count: Number of frames processed (required for cloud providers)
                - cloud_metrics: Cloud resource metrics (required for 'aws', 'azure', 'gcp')
                - processing_time: Processing time in seconds (required for 'edge')
                
        Returns:
            Total cost calculation result
        """
        provider = provider.lower()
        
        if provider not in self.config:
            logger.warning(f"No cost configuration found for provider: {provider}")
            return 0.0
            
        if provider == 'aws':
            return self._calculate_aws_cost(kwargs.get('frame_count', 0), kwargs.get('cloud_metrics', {}))
        elif provider == 'azure':
            return self._calculate_azure_cost(kwargs.get('frame_count', 0), kwargs.get('cloud_metrics', {}))
        elif provider == 'gcp':
            return self._calculate_gcp_cost(kwargs.get('frame_count', 0), kwargs.get('cloud_metrics', {}))
        elif provider == 'edge':
            return self._calculate_edge_cost(kwargs.get('processing_time', 0))
        else:
            logger.warning(f"Unsupported provider: {provider}")
            return 0.0
    
    def _calculate_aws_cost(self, frame_count: int, cloud_metrics: Dict[str, Any]) -> float:
        """
        Calculate AWS cost based on cloud metrics.
        
        Args:
            frame_count: Number of frames processed
            cloud_metrics: Dictionary with cloud resource metrics
                Required: 'vcpu_seconds', 'memory_gb_seconds'
            
        Returns:
            Total AWS cost
        """
        aws_config = self.config.get('aws', {})
        
        # Validate required metrics
        if not cloud_metrics or 'vcpu_seconds' not in cloud_metrics or 'memory_gb_seconds' not in cloud_metrics:
            logger.warning("Missing required cloud metrics for AWS cost calculation")
            return 0.0
            
        # Image detection cost
        detection_cost = frame_count * aws_config.get('image_detection_per_image', 0.001)
        
        # Compute costs (AWS Fargate)
        vcpu_seconds = cloud_metrics.get('vcpu_seconds', 0)
        memory_gb_seconds = cloud_metrics.get('memory_gb_seconds', 0)
        
        compute_cost = (
            vcpu_seconds * aws_config.get('vCPU_per_sec', 0.00001124) +
            memory_gb_seconds * aws_config.get('memory_per_GB_sec', 0.00000124)
        )
        
        # Total cost
        total_cost = detection_cost + compute_cost
        
        logger.info(f"AWS Cost Calculation - Detection: ${detection_cost:.4f}, Compute: ${compute_cost:.4f}, Total: ${total_cost:.4f}")
        
        return total_cost
        
    def _calculate_azure_cost(self, frame_count: int, cloud_metrics: Dict[str, Any]) -> float:
        """
        Calculate Azure cost based on cloud metrics.
        
        Args:
            frame_count: Number of frames processed
            cloud_metrics: Dictionary with cloud resource metrics
                Required: 'vcpu_seconds', 'memory_gb_seconds'
            
        Returns:
            Total Azure cost
        """
        azure_config = self.config.get('azure', {})
        
        # Validate required metrics
        if not cloud_metrics or 'vcpu_seconds' not in cloud_metrics or 'memory_gb_seconds' not in cloud_metrics:
            logger.warning("Missing required cloud metrics for Azure cost calculation")
            return 0.0
            
        # Image detection cost
        detection_cost = frame_count * azure_config.get('image_detection_per_image', 0.001)
        
        # Compute costs (Azure Container Apps)
        vcpu_seconds = cloud_metrics.get('vcpu_seconds', 0)
        memory_gb_seconds = cloud_metrics.get('memory_gb_seconds', 0)
        
        # Apply free tier credits if applicable
        free_tier = azure_config.get('free_tier', {})
        vcpu_seconds_billable = max(0, vcpu_seconds - free_tier.get('vCPU_sec', 0))
        memory_gb_seconds_billable = max(0, memory_gb_seconds - free_tier.get('memory_GB_sec', 0))
        
        compute_cost = (
            vcpu_seconds_billable * azure_config.get('vCPU_per_sec', 0.000024) +
            memory_gb_seconds_billable * azure_config.get('memory_per_GB_sec', 0.000003)
        )
        
        # Total cost
        total_cost = detection_cost + compute_cost
        
        logger.info(f"Azure Cost Calculation - Detection: ${detection_cost:.4f}, Compute: ${compute_cost:.4f}, "
                   f"Total: ${total_cost:.4f}")
        logger.debug(f"Azure Free Tier - vCPU seconds used: {vcpu_seconds}, billable: {vcpu_seconds_billable}, "
                    f"Memory GB seconds used: {memory_gb_seconds}, billable: {memory_gb_seconds_billable}")
        
        return total_cost
        
    def _calculate_gcp_cost(self, frame_count: int, cloud_metrics: Dict[str, Any]) -> float:
        """
        Calculate GCP cost based on cloud metrics.
        
        Args:
            frame_count: Number of frames processed
            cloud_metrics: Dictionary with cloud resource metrics
                Required: 'vcpu_seconds', 'memory_gb_seconds'
            
        Returns:
            Total GCP cost
        """
        gcp_config = self.config.get('gcp', {})
        
        # Validate required metrics
        if not cloud_metrics or 'vcpu_seconds' not in cloud_metrics or 'memory_gb_seconds' not in cloud_metrics:
            logger.warning("Missing required cloud metrics for GCP cost calculation")
            return 0.0
            
        # Image detection cost
        detection_cost = frame_count * gcp_config.get('image_detection_per_image', 0.0015)
        
        # Compute costs (GCP Cloud Run)
        vcpu_seconds = cloud_metrics.get('vcpu_seconds', 0)
        memory_gb_seconds = cloud_metrics.get('memory_gb_seconds', 0)
        
        # Apply free tier credits if applicable
        free_tier = gcp_config.get('free_tier', {})
        vcpu_seconds_billable = max(0, vcpu_seconds - free_tier.get('vCPU_sec', 0))
        memory_gb_seconds_billable = max(0, memory_gb_seconds - free_tier.get('memory_GB_sec', 0))
        
        compute_cost = (
            vcpu_seconds_billable * gcp_config.get('vCPU_per_sec', 0.000024) +
            memory_gb_seconds_billable * gcp_config.get('memory_per_GB_sec', 0.0000025)
        )
        
        # Total cost
        total_cost = detection_cost + compute_cost
        
        logger.info(f"GCP Cost Calculation - Detection: ${detection_cost:.4f}, Compute: ${compute_cost:.4f}, "
                   f"Total: ${total_cost:.4f}")
        logger.debug(f"GCP Free Tier - vCPU seconds used: {vcpu_seconds}, billable: {vcpu_seconds_billable}, "
                    f"Memory GB seconds used: {memory_gb_seconds}, billable: {memory_gb_seconds_billable}")
        
        return total_cost
        
    def _calculate_edge_cost(self, processing_time: float) -> float:
        """
        Calculate edge processing cost based on processing time.
        
        Args:
            processing_time: Processing time in seconds
            
        Returns:
            Total edge cost
        """
        edge_config = self.config.get('edge', {})
        
        if processing_time <= 0:
            logger.warning("Invalid processing time for Edge cost calculation")
            return 0.0
            
        # Cost per second of operation
        cost_per_second = edge_config.get('total_monthly_cost_per_sec', 0.00000105)
        
        # Total cost
        total_cost = processing_time * cost_per_second
        
        logger.info(f"Edge Cost Calculation - Processing time: {processing_time:.2f}s, "
                   f"Cost per second: ${cost_per_second:.10f}, Total: ${total_cost:.6f}")
        
        return total_cost


def create_cost_calculator(config_path: str) -> CostCalculator:
    """
    Create a cost calculator.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        CostCalculator instance
    """
    return CostCalculator(config_path) 