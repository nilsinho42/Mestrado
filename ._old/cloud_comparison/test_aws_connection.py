import os
import cv2
import numpy as np
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_aws_connection():
    """Test AWS Rekognition connection and basic functionality."""
    try:
        # Create Rekognition client
        rekognition = boto3.client(
            'rekognition',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        
        # Create a test image (white rectangle with text)
        img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Test Image", (150, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save the image
        cv2.imwrite('test_image.jpg', img)
        
        # Read the image file
        with open('test_image.jpg', 'rb') as image:
            image_bytes = image.read()
        
        # Test object detection
        print("Testing object detection...")
        response = rekognition.detect_labels(
            Image={'Bytes': image_bytes}
        )
        
        print("\nDetected labels:")
        for label in response['Labels']:
            print(f"- {label['Name']} (Confidence: {label['Confidence']:.2f}%)")
        
        # Test image analysis
        print("\nTesting image analysis...")
        response = rekognition.detect_text(
            Image={'Bytes': image_bytes}
        )
        
        print("\nDetected text:")
        for text in response['TextDetections']:
            print(f"- {text['DetectedText']} (Confidence: {text['Confidence']:.2f}%)")
        
        print("\nAWS Rekognition connection test successful!")
        return True
        
    except Exception as e:
        print(f"Error testing AWS Rekognition: {str(e)}")
        return False

if __name__ == "__main__":
    test_aws_connection() 