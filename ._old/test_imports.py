import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_yolo():
    try:
        from detectors import YOLODetector
        detector = YOLODetector()
        logger.info("YOLO detector initialized successfully")
        return True
    except Exception as e:
        logger.error(f"YOLO detector initialization failed: {e}")
        return False

def test_database():
    try:
        from ml.core.db_utils import Database
        db = Database()
        result = db.connect()
        db.disconnect()
        if result:
            logger.info("Database connection successful")
        else:
            logger.error("Database connection failed")
        return result
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return False

def test_video_pipeline():
    try:
        from video_pipeline import VideoPipeline
        pipeline = VideoPipeline()
        logger.info("Video pipeline initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Video pipeline initialization failed: {e}")
        return False

def test_cloud_storage():
    try:
        from cloud_storage import CloudStorage
        storage = CloudStorage()
        logger.info("Cloud storage initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Cloud storage initialization failed: {e}")
        return False

def test_trackers():
    try:
        from trackers import DeepSORTTracker
        tracker = DeepSORTTracker()
        logger.info("DeepSORT tracker initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Tracker initialization failed: {e}")
        return False

def test_api():
    try:
        import uvicorn
        from fastapi import FastAPI
        from run_api import app
        logger.info("API dependencies loaded successfully")
        return True
    except Exception as e:
        logger.error(f"API dependencies failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Please specify a test to run")
        sys.exit(1)

    test_name = sys.argv[1]
    test_map = {
        "yolo": test_yolo,
        "database": test_database,
        "pipeline": test_video_pipeline,
        "storage": test_cloud_storage,
        "tracker": test_trackers,
        "api": test_api
    }

    if test_name not in test_map:
        logger.error(f"Unknown test: {test_name}")
        logger.info(f"Available tests: {', '.join(test_map.keys())}")
        sys.exit(1)

    success = test_map[test_name]()
    sys.exit(0 if success else 1)