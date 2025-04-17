"""
Utility script to download the required models for the application.
"""

import os
import sys
import requests
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# URLs for model downloads - primary sources
MODEL_URLS = {
    "yolov11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
}

def download_file(url, destination, max_retries=3):
    """Download a file from URL to destination."""
    logger.info(f"Downloading {url} to {destination}")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"Failed to download from {url}: HTTP {response.status_code}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                return False
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Save the file
            with open(destination, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Print progress
                        percent = int(100 * downloaded / total_size) if total_size > 0 else 0
                        sys.stdout.write(f"\rDownloading: {percent}% ({downloaded}/{total_size} bytes)")
                        sys.stdout.flush()
            
            print()  # New line after progress
            logger.info(f"Successfully downloaded {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return False
    
    return False

def download_via_ultralytics(model_name, destination):
    """Try to download model using ultralytics directly."""
    try:
        logger.info(f"Attempting to download {model_name} via ultralytics")
        from ultralytics import YOLO
        
        # Strip .pt extension for YOLO API
        model_type = model_name.replace('.pt', '')
        
        # Load model (this will download it)
        model = YOLO(model_type)
        
        # Check if it downloaded properly and exists
        if os.path.exists(model.ckpt_path):
            # If YOLO saved it in a different location, copy it to our destination
            if model.ckpt_path != destination:
                import shutil
                shutil.copy(model.ckpt_path, destination)
                logger.info(f"Copied model from {model.ckpt_path} to {destination}")
            
            return True
        else:
            logger.error(f"Model download via ultralytics failed: Model file not found")
            return False
            
    except Exception as e:
        logger.error(f"Failed to download via ultralytics: {e}")
        return False

def main():
    """Main function to download YOLOv11n model."""
    # Ensure we're in the correct directory
    ml_dir = Path(__file__).parent
    os.chdir(ml_dir)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # For YOLOv11n, try both direct download and ultralytics
    model_path = ml_dir / "yolov11n.pt"
    
    # Check if model already exists
    if model_path.exists():
        logger.info(f"Model YOLOv11n already exists at {model_path}")
        return
    
    # First, try to download as yolo11n.pt (as it appears in the URL)
    download_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    temp_model_path = ml_dir / "yolo11n.pt"
    
    # Try direct download
    download_success = download_file(download_url, temp_model_path)
    
    if download_success:
        # Rename to yolov11n.pt for application compatibility
        os.rename(temp_model_path, model_path)
        logger.info(f"Successfully downloaded and renamed model to {model_path}")
        return
    
    # If direct download failed, try ultralytics download
    if download_via_ultralytics("yolov11n", model_path):
        logger.info(f"Successfully downloaded {model_path} via ultralytics")
        return
    
    # If all methods failed, show error message and exit
    error_message = (
        "ERROR: Could not download YOLOv11n model. This model is required for the application to function.\n"
        "Please manually download the model from:\n"
        f"  {download_url}\n"
        "and save it as yolov11n.pt in the following location:\n"
        f"  {os.path.abspath(model_path)}"
    )
    logger.error(error_message)
    print("\n" + "="*80)
    print(error_message)
    print("="*80 + "\n")
    sys.exit(1)  # Exit with error code

if __name__ == "__main__":
    main() 