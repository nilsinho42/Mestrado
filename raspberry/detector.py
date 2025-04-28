import os
import logging
import time
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLODetector:
    """
    YOLO detector implementation for Raspberry Pi.
    """
    def __init__(self, model_path="yolov11n.pt", confidence_threshold=0.5):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to the YOLO model (use a small model like YOLOv11n for Raspberry Pi)
            confidence_threshold: Confidence threshold for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        # Load class names
        self.people_classes = ['person', 'human', 'people', 'pedestrian', 'man', 'woman', 'child', 'baby']
        self.vehicle_classes = ['car', 'vehicle', 'automobile', 'truck', 'van', 'bus', 'motorcycle', 'transportation', 'taxi', 'ambulance', 'police car']
        self.class_names = self._load_class_names()
        
        # Initialize the model using Ultralytics API
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO model from {model_path}")
        except Exception as e:
            error_msg = f"Failed to initialize YOLO model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _load_class_names(self):
        """
        Load COCO class names.
        """
        # COCO class names
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", 
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", 
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", 
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", 
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", 
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", 
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
    
    def process_image(self, image):
        """
        Process an image using YOLO detector.
        
        Args:
            image: Image as numpy array (BGR format from OpenCV)
            
        Returns:
            List of detections with bbox, confidence, class_id, and class_name
        """
        start_time = time.time()
        
        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
            # Extract detections
            detections = []
            
            if len(results) > 0:
                result = results[0]  # Get the first result
                
                # Process each detection
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names[cls_id]
                    
                    if cls_name not in self.people_classes and cls_name not in self.vehicle_classes:
                        continue

                    # Add detection to results
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf),
                        'class_id': cls_id,
                        'class_name': cls_name
                    })
            
            logger.info(f"Detection took {time.time() - start_time:.3f} seconds, found {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return [] 