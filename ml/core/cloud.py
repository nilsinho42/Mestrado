"""
Cloud service integrations for AWS and Azure.
Provides utilities for interacting with cloud storage and vision APIs.
"""

import os
import time
import logging
import json
from typing import Dict, Any, List, Optional, Union, BinaryIO
from abc import ABC, abstractmethod
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

class CloudStorageProvider(ABC):
    """Abstract base class for cloud storage providers."""
    
    def __init__(self, name: str = "base_storage"):
        """
        Initialize the storage provider.
        
        Args:
            name: Provider name/identifier
        """
        self.name = name
        logger.info(f"Initialized {self.name} storage provider")
    
    @abstractmethod
    def upload_file(self, file_path: str, destination_path: Optional[str] = None) -> str:
        """
        Upload a file to cloud storage.
        
        Args:
            file_path: Local path to the file
            destination_path: Optional destination path in cloud storage
            
        Returns:
            URL or path to the uploaded file
        """
        pass
    
    @abstractmethod
    def download_file(self, cloud_path: str, local_path: Optional[str] = None) -> str:
        """
        Download a file from cloud storage.
        
        Args:
            cloud_path: Path to the file in cloud storage
            local_path: Optional local path to save the file
            
        Returns:
            Local path where the file was saved
        """
        pass
    
    @abstractmethod
    def list_files(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List files in cloud storage.
        
        Args:
            prefix: Optional prefix to filter files
            
        Returns:
            List of file information dictionaries
        """
        pass
    
    @abstractmethod
    def delete_file(self, cloud_path: str) -> bool:
        """
        Delete a file from cloud storage.
        
        Args:
            cloud_path: Path to the file in cloud storage
            
        Returns:
            True if deletion was successful
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get storage metrics.
        
        Returns:
            Dictionary with storage metrics
        """
        # Placeholder - override in subclasses
        return {
            "provider": self.name,
            "files_stored": 0,
            "total_size_bytes": 0,
            "requests": {
                "get": 0,
                "put": 0,
                "delete": 0,
                "list": 0
            }
        }


class AWSS3Storage(CloudStorageProvider):
    """AWS S3 storage provider implementation."""
    
    def __init__(self, bucket_name: Optional[str] = None, region: Optional[str] = None):
        """
        Initialize AWS S3 storage provider.
        
        Args:
            bucket_name: S3 bucket name
            region: AWS region
        """
        super().__init__(name="aws_s3")
        
        self.bucket_name = bucket_name or os.getenv("AWS_S3_BUCKET")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        
        if not self.bucket_name:
            raise ValueError("S3 bucket name not provided and AWS_S3_BUCKET environment variable not set")
        
        # Initialize boto3 client
        self.s3_client = None
        try:
            import boto3
            self.s3_client = boto3.client('s3', region_name=self.region)
            logger.info(f"Initialized AWS S3 client for bucket {self.bucket_name}")
        except ImportError:
            logger.error("boto3 not installed. Install with 'pip install boto3'")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AWS S3 client: {str(e)}")
            raise
        
        # Track metrics
        self.metrics = {
            "put_requests": 0,
            "get_requests": 0,
            "list_requests": 0,
            "delete_requests": 0
        }
    
    def upload_file(self, file_path: str, destination_path: Optional[str] = None) -> str:
        """
        Upload a file to S3.
        
        Args:
            file_path: Local path to the file
            destination_path: Optional destination key in S3
            
        Returns:
            S3 URL to the uploaded file
        """
        try:
            # Generate key if not provided
            if not destination_path:
                destination_path = Path(file_path).name
            
            # Upload file
            self.s3_client.upload_file(file_path, self.bucket_name, destination_path)
            
            # Generate URL
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{destination_path}"
            
            # Update metrics
            self.metrics["put_requests"] += 1
            
            logger.info(f"Uploaded {file_path} to S3: {url}")
            return url
            
        except Exception as e:
            logger.error(f"Error uploading file to S3: {str(e)}")
            raise
    
    def download_file(self, cloud_path: str, local_path: Optional[str] = None) -> str:
        """
        Download a file from S3.
        
        Args:
            cloud_path: S3 key or URL
            local_path: Optional local path to save the file
            
        Returns:
            Local path where the file was saved
        """
        try:
            # Extract key from URL if needed
            if cloud_path.startswith("http"):
                # Example: https://bucket.s3.region.amazonaws.com/key
                cloud_path = cloud_path.split(".amazonaws.com/")[1]
            
            # Generate local path if not provided
            if not local_path:
                local_path = tempfile.mktemp(suffix=Path(cloud_path).suffix)
            
            # Download file
            self.s3_client.download_file(self.bucket_name, cloud_path, local_path)
            
            # Update metrics
            self.metrics["get_requests"] += 1
            
            logger.info(f"Downloaded {cloud_path} from S3 to {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Error downloading file from S3: {str(e)}")
            raise
    
    def list_files(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List files in S3 bucket.
        
        Args:
            prefix: Optional prefix to filter files
            
        Returns:
            List of file information dictionaries
        """
        try:
            # List objects
            if prefix:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix
                )
            else:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name
                )
            
            # Update metrics
            self.metrics["list_requests"] += 1
            
            # Parse response
            files = []
            for obj in response.get("Contents", []):
                files.append({
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                    "url": f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{obj['Key']}"
                })
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files in S3: {str(e)}")
            return []
    
    def delete_file(self, cloud_path: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            cloud_path: S3 key or URL
            
        Returns:
            True if deletion was successful
        """
        try:
            # Extract key from URL if needed
            if cloud_path.startswith("http"):
                # Example: https://bucket.s3.region.amazonaws.com/key
                cloud_path = cloud_path.split(".amazonaws.com/")[1]
            
            # Delete object
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=cloud_path
            )
            
            # Update metrics
            self.metrics["delete_requests"] += 1
            
            logger.info(f"Deleted {cloud_path} from S3")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file from S3: {str(e)}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get S3 storage metrics.
        
        Returns:
            Dictionary with storage metrics
        """
        try:
            # Get bucket size
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name
            )
            
            total_size = sum(obj["Size"] for obj in response.get("Contents", []))
            file_count = len(response.get("Contents", []))
            
            return {
                "provider": self.name,
                "bucket": self.bucket_name,
                "region": self.region,
                "files_stored": file_count,
                "total_size_bytes": total_size,
                "requests": self.metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting S3 metrics: {str(e)}")
            return {
                "provider": self.name,
                "bucket": self.bucket_name,
                "region": self.region,
                "files_stored": 0,
                "total_size_bytes": 0,
                "requests": self.metrics
            }


class AzureBlobStorage(CloudStorageProvider):
    """Azure Blob Storage provider implementation."""
    
    def __init__(self, container_name: Optional[str] = None, 
                 connection_string: Optional[str] = None,
                 account_name: Optional[str] = None,
                 account_key: Optional[str] = None):
        """
        Initialize Azure Blob Storage provider.
        
        Args:
            container_name: Azure Blob container name
            connection_string: Azure Storage connection string
            account_name: Azure Storage account name (alternative to connection_string)
            account_key: Azure Storage account key (alternative to connection_string)
        """
        super().__init__(name="azure_blob")
        
        self.container_name = container_name or os.getenv("AZURE_CONTAINER_NAME")
        
        if not self.container_name:
            raise ValueError("Container name not provided and AZURE_CONTAINER_NAME environment variable not set")
        
        # Get connection info
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.account_name = account_name or os.getenv("AZURE_STORAGE_ACCOUNT")
        self.account_key = account_key or os.getenv("AZURE_STORAGE_KEY")
        
        if not self.connection_string and not (self.account_name and self.account_key):
            raise ValueError("Neither connection string nor account credentials provided")
        
        # Initialize Azure client
        self.blob_service_client = None
        self.container_client = None
        
        try:
            from azure.storage.blob import BlobServiceClient, ContainerClient
            
            if self.connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            else:
                # Create client using account name and key
                self.blob_service_client = BlobServiceClient(
                    account_url=f"https://{self.account_name}.blob.core.windows.net",
                    credential=self.account_key
                )
            
            # Get container client
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
            
            # Create container if it doesn't exist
            if not self.container_client.exists():
                self.container_client.create_container()
            
            logger.info(f"Initialized Azure Blob Storage client for container {self.container_name}")
            
        except ImportError:
            logger.error("azure-storage-blob not installed. Install with 'pip install azure-storage-blob'")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Azure Blob Storage client: {str(e)}")
            raise
        
        # Track metrics
        self.metrics = {
            "put_requests": 0,
            "get_requests": 0,
            "list_requests": 0,
            "delete_requests": 0
        }
    
    def upload_file(self, file_path: str, destination_path: Optional[str] = None) -> str:
        """
        Upload a file to Azure Blob Storage.
        
        Args:
            file_path: Local path to the file
            destination_path: Optional destination blob name
            
        Returns:
            Azure Blob URL to the uploaded file
        """
        try:
            # Generate blob name if not provided
            if not destination_path:
                destination_path = Path(file_path).name
            
            # Upload file
            with open(file_path, "rb") as data:
                blob_client = self.container_client.get_blob_client(destination_path)
                blob_client.upload_blob(data, overwrite=True)
            
            # Generate URL
            url = blob_client.url
            
            # Update metrics
            self.metrics["put_requests"] += 1
            
            logger.info(f"Uploaded {file_path} to Azure Blob Storage: {url}")
            return url
            
        except Exception as e:
            logger.error(f"Error uploading file to Azure Blob Storage: {str(e)}")
            raise
    
    def download_file(self, cloud_path: str, local_path: Optional[str] = None) -> str:
        """
        Download a file from Azure Blob Storage.
        
        Args:
            cloud_path: Blob name or URL
            local_path: Optional local path to save the file
            
        Returns:
            Local path where the file was saved
        """
        try:
            # Extract blob name from URL if needed
            if cloud_path.startswith("http"):
                # Example: https://account.blob.core.windows.net/container/blob
                cloud_path = cloud_path.split(f"{self.container_name}/")[1]
            
            # Generate local path if not provided
            if not local_path:
                local_path = tempfile.mktemp(suffix=Path(cloud_path).suffix)
            
            # Download file
            blob_client = self.container_client.get_blob_client(cloud_path)
            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            # Update metrics
            self.metrics["get_requests"] += 1
            
            logger.info(f"Downloaded {cloud_path} from Azure Blob Storage to {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Error downloading file from Azure Blob Storage: {str(e)}")
            raise
    
    def list_files(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List files in Azure Blob Storage container.
        
        Args:
            prefix: Optional prefix to filter blobs
            
        Returns:
            List of file information dictionaries
        """
        try:
            # List blobs
            if prefix:
                blobs = self.container_client.list_blobs(name_starts_with=prefix)
            else:
                blobs = self.container_client.list_blobs()
            
            # Update metrics
            self.metrics["list_requests"] += 1
            
            # Parse results
            files = []
            for blob in blobs:
                blob_client = self.container_client.get_blob_client(blob.name)
                files.append({
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified.isoformat(),
                    "url": blob_client.url
                })
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files in Azure Blob Storage: {str(e)}")
            return []
    
    def delete_file(self, cloud_path: str) -> bool:
        """
        Delete a file from Azure Blob Storage.
        
        Args:
            cloud_path: Blob name or URL
            
        Returns:
            True if deletion was successful
        """
        try:
            # Extract blob name from URL if needed
            if cloud_path.startswith("http"):
                # Example: https://account.blob.core.windows.net/container/blob
                cloud_path = cloud_path.split(f"{self.container_name}/")[1]
            
            # Delete blob
            blob_client = self.container_client.get_blob_client(cloud_path)
            blob_client.delete_blob()
            
            # Update metrics
            self.metrics["delete_requests"] += 1
            
            logger.info(f"Deleted {cloud_path} from Azure Blob Storage")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file from Azure Blob Storage: {str(e)}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get Azure Blob Storage metrics.
        
        Returns:
            Dictionary with storage metrics
        """
        try:
            # List all blobs to get size
            blobs = list(self.container_client.list_blobs())
            
            total_size = sum(blob.size for blob in blobs)
            file_count = len(blobs)
            
            return {
                "provider": self.name,
                "container": self.container_name,
                "account": self.account_name,
                "files_stored": file_count,
                "total_size_bytes": total_size,
                "requests": self.metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting Azure Blob Storage metrics: {str(e)}")
            return {
                "provider": self.name,
                "container": self.container_name,
                "account": self.account_name,
                "files_stored": 0,
                "total_size_bytes": 0,
                "requests": self.metrics
            }


# Factory function to create storage provider
def create_storage_provider(provider: str, **kwargs) -> CloudStorageProvider:
    """
    Create an appropriate storage provider based on the provider name.
    
    Args:
        provider: Provider name ('aws' or 'azure')
        **kwargs: Additional configuration for the provider
    
    Returns:
        A CloudStorageProvider instance
    """
    if provider.lower() == 'aws' or provider.lower() == 's3':
        return AWSS3Storage(**kwargs)
    
    elif provider.lower() == 'azure' or provider.lower() == 'blob':
        return AzureBlobStorage(**kwargs)
    
    else:
        logger.warning(f"Unknown provider '{provider}'. Using AWS S3 as default.")
        return AWSS3Storage(**kwargs)
