import os
import time
import boto3
from azure.storage.blob import BlobServiceClient
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudStorage:
    def __init__(self):
        """Initialize cloud storage clients for AWS and Azure."""
        # Initialize AWS S3 client
        if 'AWS_ACCESS_KEY_ID' in os.environ:
            self.s3_client = boto3.client('s3')
            self.aws_bucket = os.getenv('AWS_BUCKET_NAME', 'object-detection-data')
        else:
            self.s3_client = None
            self.aws_bucket = None
        
        # Initialize Azure Blob Storage client
        if 'AZURE_STORAGE_CONNECTION_STRING' in os.environ:
            self.blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
            self.azure_container = os.getenv('AZURE_CONTAINER_NAME', 'object-detection-data')
            
            # Create container if it doesn't exist
            try:
                container_client = self.blob_service_client.get_container_client(self.azure_container)
                if not container_client.exists():
                    container_client = self.blob_service_client.create_container(self.azure_container)
            except Exception as e:
                logger.error(f"Error initializing Azure container: {e}")
                self.blob_service_client = None
                self.azure_container = None
        else:
            self.blob_service_client = None
            self.azure_container = None
        
        logger.info(f"Initialized cloud storage. AWS Bucket: {self.aws_bucket}, Azure Container: {self.azure_container}")
    
    def upload_to_aws(self, file_path: Union[str, Path], object_key: Optional[str] = None) -> Dict[str, Any]:
        """Upload file to AWS S3.
        
        Args:
            file_path: Path to file to upload
            object_key: S3 object key (default: file name)
            
        Returns:
            Dictionary with upload information and metrics
        """
        if not self.s3_client or not self.aws_bucket:
            logger.warning("AWS S3 client not configured")
            return {'success': False, 'error': 'AWS S3 client not configured'}
        
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return {'success': False, 'error': 'File does not exist'}
        
        # Use file name as object key if not provided
        if object_key is None:
            object_key = file_path.name
        
        try:
            # Get file size in bytes
            file_size = file_path.stat().st_size
            
            # Measure upload time
            start_time = time.time()
            
            # Upload file
            self.s3_client.upload_file(
                Filename=str(file_path),
                Bucket=self.aws_bucket,
                Key=object_key
            )
            
            # Calculate upload time and speed
            upload_time = time.time() - start_time
            upload_speed_mbps = (file_size / 1024 / 1024) / upload_time if upload_time > 0 else 0
            
            logger.info(f"Uploaded to AWS S3: {object_key} ({file_size/1024/1024:.2f} MB) "
                      f"in {upload_time:.2f}s ({upload_speed_mbps:.2f} MB/s)")
            
            # Get the URL of the uploaded file
            url = f"https://{self.aws_bucket}.s3.amazonaws.com/{object_key}"
            
            return {
                'success': True,
                'service': 'aws',
                'bucket': self.aws_bucket,
                'object_key': object_key,
                'url': url,
                'file_size_bytes': file_size,
                'upload_time_seconds': upload_time,
                'upload_speed_mbps': upload_speed_mbps
            }
        
        except Exception as e:
            logger.error(f"Error uploading to AWS S3: {e}")
            return {'success': False, 'service': 'aws', 'error': str(e)}
    
    def upload_to_azure(self, file_path: Union[str, Path], blob_name: Optional[str] = None) -> Dict[str, Any]:
        """Upload file to Azure Blob Storage.
        
        Args:
            file_path: Path to file to upload
            blob_name: Blob name (default: file name)
            
        Returns:
            Dictionary with upload information and metrics
        """
        if not self.blob_service_client or not self.azure_container:
            logger.warning("Azure Blob Storage client not configured")
            return {'success': False, 'error': 'Azure Blob Storage client not configured'}
        
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return {'success': False, 'error': 'File does not exist'}
        
        # Use file name as blob name if not provided
        if blob_name is None:
            blob_name = file_path.name
        
        try:
            # Get file size in bytes
            file_size = file_path.stat().st_size
            
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.azure_container,
                blob=blob_name
            )
            
            # Measure upload time
            start_time = time.time()
            
            # Upload file
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            # Calculate upload time and speed
            upload_time = time.time() - start_time
            upload_speed_mbps = (file_size / 1024 / 1024) / upload_time if upload_time > 0 else 0
            
            logger.info(f"Uploaded to Azure Blob Storage: {blob_name} ({file_size/1024/1024:.2f} MB) "
                      f"in {upload_time:.2f}s ({upload_speed_mbps:.2f} MB/s)")
            
            # Get the URL of the uploaded file
            url = f"https://{os.getenv('AZURE_STORAGE_ACCOUNT')}.blob.core.windows.net/{self.azure_container}/{blob_name}"
            
            return {
                'success': True,
                'service': 'azure',
                'container': self.azure_container,
                'blob_name': blob_name,
                'url': url,
                'file_size_bytes': file_size,
                'upload_time_seconds': upload_time,
                'upload_speed_mbps': upload_speed_mbps
            }
        
        except Exception as e:
            logger.error(f"Error uploading to Azure Blob Storage: {e}")
            return {'success': False, 'service': 'azure', 'error': str(e)}
    
    def upload_to_all(self, file_path: Union[str, Path], object_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Upload file to all configured cloud storage services.
        
        Args:
            file_path: Path to file to upload
            object_name: Object name/key (default: file name)
            
        Returns:
            Dictionary with upload results for each service
        """
        results = {}
        
        # Upload to AWS if configured
        if self.s3_client and self.aws_bucket:
            results['aws'] = self.upload_to_aws(file_path, object_name)
        
        # Upload to Azure if configured
        if self.blob_service_client and self.azure_container:
            results['azure'] = self.upload_to_azure(file_path, object_name)
        
        return results
    
    def get_aws_file_url(self, object_key: str) -> str:
        """Get the URL for an AWS S3 object.
        
        Args:
            object_key: S3 object key
            
        Returns:
            URL string
        """
        if not self.aws_bucket:
            return ""
        
        return f"https://{self.aws_bucket}.s3.amazonaws.com/{object_key}"
    
    def get_azure_file_url(self, blob_name: str) -> str:
        """Get the URL for an Azure Blob Storage object.
        
        Args:
            blob_name: Blob name
            
        Returns:
            URL string
        """
        if not self.azure_container:
            return ""
        
        return f"https://{os.getenv('AZURE_STORAGE_ACCOUNT')}.blob.core.windows.net/{self.azure_container}/{blob_name}" 