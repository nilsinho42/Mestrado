from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List
import os
from pathlib import Path

class Settings(BaseSettings):
    # Service Configuration
    SERVICE_NAME: str = "ml_service"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # JWT Configuration
    JWT_SECRET: str = Field(..., min_length=32)
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION: int = 3600  # 1 hour
    
    # ML Configuration
    MODEL_PATH: str = "models/yolov5s.pt"
    DEVICE: str = "cuda"  # or "cpu"
    
    # Storage Configuration
    VIDEO_UPLOAD_DIR: Path = Path("uploads")
    MAX_VIDEO_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_VIDEO_TYPES: List[str] = [
        "video/mp4",
        "video/avi",
        "video/mov",
        "video/quicktime"
    ]
    CLEANUP_DAYS: int = 30
    
    # Database Configuration
    SQLITE_DB_PATH: str = "processing_status.db"
    POSTGRES_DSN: str = Field(..., regex=r"^postgresql://.*")
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    # Note: Monitoring Configuration (OTLP_ENDPOINT, PUSHGATEWAY_URL) has been removed as it wasn't being used
    
    @validator("VIDEO_UPLOAD_DIR")
    def create_upload_dir(cls, v):
        os.makedirs(v, exist_ok=True)
        return v
    
    @validator("MODEL_PATH")
    def check_model_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Model file not found: {v}")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

def get_settings() -> Settings:
    """Get validated settings."""
    return Settings()

# Global settings instance
settings = get_settings() 