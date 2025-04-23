import os
import logging
import time
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLODetector:
    """
    YOLO detector implementation for Raspberry Pi.
    """
    def __init__(self, model_path="yolov11n.pt", confidence_threshold=0.20):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to the YOLO model (use a small model like YOLOv11n for Raspberry Pi)
            confidence_threshold: Confidence threshold for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        # Initialize the model
        try:
            logger.info(f"Loading YOLO model from {model_path}...")
            
            # First try using the ultralytics YOLO
            try:
                import ultralytics
                self.model = ultralytics.YOLO(model_path)
                self.model_type = "ultralytics"
                logger.info("Loaded model using ultralytics YOLO")
            except (ImportError, Exception) as e:
                logger.warning(f"Could not load using ultralytics: {e}. Trying OpenCV DNN...")
                # Fall back to OpenCV DNN
                self.model = cv2.dnn.readNetFromONNX(model_path)
                self.model_type = "opencv"
                logger.info("Loaded model using OpenCV DNN")
                
            # Load class names
            self.class_names = self._load_class_names()
            logger.info(f"Loaded {len(self.class_names)} class names")
            
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
            if self.model_type == "ultralytics":
                return self._process_with_ultralytics(image)
            else:
                return self._process_with_opencv(image)
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []
        finally:
            logger.info(f"Detection took {time.time() - start_time:.3f} seconds")
    
    def _process_with_ultralytics(self, image):
        """Process using ultralytics YOLO."""
        # Run inference
        results = self.model(image)
        
        # Extract detections
        detections = []
        
        # Process results (different based on YOLO version)
        try:
            # Get bounding boxes
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates in (x1, y1, x2, y2) format
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get confidence
                    conf = float(box.conf[0])
                    
                    # Get class ID and name
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names[cls_id]
                    
                    # Filter based on confidence threshold
                    if conf >= self.confidence_threshold:
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class_id': cls_id,
                            'class_name': cls_name
                        })
        except Exception as e:
            logger.error(f"Error extracting detections from ultralytics results: {e}")
        
        return detections
    
    def _process_with_opencv(self, image):
        """Process using OpenCV DNN."""
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Prepare image for inference
        blob = cv2.dnn.blobFromImage(
            image, 1/255.0, (640, 640), 
            swapRB=True, crop=False
        )
        
        # Set input and forward pass
        self.model.setInput(blob)
        outputs = self.model.forward()
        
        # Process output
        detections = []
        
        # Outputs format depends on the model
        # For YOLO v8 ONNX model, outputs shape is often [1, 84, 8400]
        # Where 84 = 4 (box coords) + 80 (class scores)
        try:
            outputs = outputs[0]
            rows = outputs.shape[1]
            
            # Process each detection
            for i in range(rows):
                row = outputs[0, i, :]
                
                # Confidence score
                confidence = row[4]
                
                # Filter weak detections
                if confidence < self.confidence_threshold:
                    continue
                
                # Get class scores
                class_scores = row[5:]
                class_id = np.argmax(class_scores)
                
                # Filter by class confidence
                if class_scores[class_id] < self.confidence_threshold:
                    continue
                
                # Box coordinates (normalized)
                x, y, w, h = row[0:4]
                
                # Convert to image coordinates
                left = int((x - 0.5 * w) * width)
                top = int((y - 0.5 * h) * height)
                right = int((x + 0.5 * w) * width)
                bottom = int((y + 0.5 * h) * height)
                
                # Ensure coordinates are within image boundaries
                left = max(0, left)
                top = max(0, top)
                right = min(width - 1, right)
                bottom = min(height - 1, bottom)
                
                # Add detection to results
                detections.append({
                    'bbox': [float(left), float(top), float(right), float(bottom)],
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'class_name': self.class_names[class_id]
                })
                
        except Exception as e:
            logger.error(f"Error processing OpenCV DNN output: {e}")
        
        return detections 